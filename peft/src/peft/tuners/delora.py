import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class DeloraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Delora`].

    Args:
        r (`int`): Delora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Delora to.
        delora_lambda (`float`): The lambda parameter for Delora scaling.
        delora_dropout (`float`): The dropout probability for Delora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Delora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_delora ( `List[bool]`): Used with `delora.MergedLinear`.
        bias (`str`): Bias type for Delora. Can be 'none', 'all' or 'delora_only'
        modules_to_save (`List[str]`):List of modules apart from Delora layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Delora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Delora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    delora_lambda: int = field(default=None, metadata={"help": "Delora lambda"})
    delora_dropout: float = field(default=None, metadata={"help": "Delora dropout"})
    Wdecompose_target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to only tune the magnitude part"
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Delora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_delora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `delora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Delora. Can be 'none', 'all' or 'delora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from Delora layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.DELORA


class DeloraModel(torch.nn.Module):
    """
    Creates Decoupled Low Rank Adapter (Delora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`DeloraConfig`]): The configuration of the Delora model.

    Returns:
        `torch.nn.Module`: The Delora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, DeloraConfig >>> from peft import DeloraModel, DeloraConfig >>>
        config = DeloraConfig(
            peft_type="DELORA", task_type="SEQ_2_SEQ_LM", r=8, delora_lambda=32, target_modules=["q", "v"],
            delora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> delora_model = DeloraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`DeloraConfig`]): The configuration of the Delora model.
    """

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        print(self.peft_config)
        self.model = model
        self._find_and_replace()
        mark_only_delora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Delora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "delora_lambda": self.peft_config.delora_lambda,
            "delora_dropout": self.peft_config.delora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)

            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_delora is None:
                        print("8 bit delora")
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        kwargs.update({"enable_delora": self.peft_config.enable_delora})
                        new_module = MergedLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_delora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                elif self.peft_config.enable_delora is not None:
                    kwargs.update({"enable_delora": self.peft_config.enable_delora})
                    if isinstance(target, Conv1D):
                        in_features, out_features = (
                            target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                        )
                    else:
                        in_features, out_features = target.in_features, target.out_features
                        if kwargs["fan_in_fan_out"]:
                            warnings.warn(
                                "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                                "Setting fan_in_fan_out to False."
                            )
                            kwargs["fan_in_fan_out"] = self.peft_config.fan_in_fan_out = False
                    new_module = MergedLinear(in_features, out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "delora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, DeloraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `delora_only` to work
def mark_only_delora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "delora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "delora_only":
        for m in model.modules():
            if isinstance(m, DeloraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class DeloraLayer:
    def __init__(
        self,
        r: int,
        delora_lambda_value: int,
        delora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.delora_lambda_value = delora_lambda_value
        # Optional dropout
        if delora_dropout > 0.0:
            self.delora_dropout = nn.Dropout(p=delora_dropout)
        else:
            self.delora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Linear(nn.Linear, DeloraLayer):
    # Delora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        delora_lambda: float = 1.,
        delora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        DeloraLayer.__init__(self, r=r, delora_lambda_value=delora_lambda, delora_dropout=delora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out

        if r > 0:
            self.delora_A = nn.Linear(in_features, r, bias=False)
            self.delora_B = nn.Linear(r, out_features, bias=False)

            self.delora_lambda = nn.Parameter(torch.full((1,), delora_lambda), requires_grad=True)

            # Frozen parameters
            self.frozen_C = nn.Parameter(torch.empty_like(self.delora_A.weight).copy_(self.delora_A.weight))
            self.frozen_C.requires_grad = False
            self.frozen_D = nn.Parameter(torch.empty_like(self.delora_B.weight).copy_(self.delora_B.weight))
            self.frozen_D.requires_grad = False

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "delora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.delora_A.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.delora_B.weight, a=math.sqrt(5))
            nn.init.constant_(self.delora_lambda, self.delora_lambda_value)
            self.frozen_C.data = self.delora_A.weight.data
            self.frozen_D.data = self.delora_B.weight.data

    def get_ABCD(self):
        # Get weights
        delora_A_weight = self.delora_A.weight  # shape: (r, in_features)
        delora_B_weight = self.delora_B.weight  # shape: (out_features, r)

        # Get norms
        delora_A_norm = delora_A_weight.norm(dim=1)  # shape: (r,)
        delora_B_norm = delora_B_weight.norm(dim=0)  # shape: (r,)
        frozen_C_norm = self.frozen_C.norm(dim=1)  # shape: (r,)
        frozen_D_norm = self.frozen_D.norm(dim=0)  # shape: (r,)

        # AB normalization
        diag12 = torch.div(self.delora_lambda / self.r, torch.mul(delora_A_norm, delora_B_norm))
        diag12 = torch.diag_embed(diag12)

        diag34 = torch.div(self.delora_lambda / self.r, torch.mul(frozen_C_norm, frozen_D_norm))
        diag34 = torch.diag_embed(diag34)

        # Get ABCD
        ABCD = delora_B_weight @ diag12 @ delora_A_weight
        ABCD = ABCD - self.frozen_D @ diag34 @ self.frozen_C

        # W scaling
        Wnorm = self.weight.data.norm(dim=0)  # shape: (in_features,)
        ABCD = torch.mul(ABCD, Wnorm.unsqueeze(0))  # shape: (out_features, in_features)

        return ABCD

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.delora_A.train(mode)
        self.delora_B.train(mode)
        self.delora_lambda.requires_grad = mode

        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.get_ABCD().to(self.weight.device, dtype=self.weight.dtype)
            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= self.get_ABCD()
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.delora_A.eval()
        self.delora_B.eval()

    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype
        if self.disable_adapters:
            if self.r > 0 and self.merged:
                self.weight.data -= self.get_ABCD()
                self.merged = False

            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                result += F.linear(self.delora_dropout(x), self.get_ABCD(), bias=None)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result


class MergedLinear(nn.Linear, DeloraLayer):
    # Delora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        delora_lambda: int = 1,
        delora_dropout: float = 0.0,
        enable_delora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        raise NotImplementedError


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, DeloraLayer):
        # Delora implemented in a dense layer
        def __init__(
            self,
            in_features,
            out_features,
            r: int = 0,
            delora_lambda: int = 1,
            delora_dropout: float = 0.0,
            Wdecompose: bool = False,
            **kwargs,
        ):
            raise NotImplementedError

    class MergedLinear8bitLt(bnb.nn.Linear8bitLt, DeloraLayer):
        # Delora implemented in a dense layer
        def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            delora_lambda: int = 1,
            delora_dropout: float = 0.0,
            enable_delora: List[bool] = [False],
            **kwargs,
        ):
            raise NotImplementedError
