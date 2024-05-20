# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, List, Optional

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils import transpose


class REDLayer(BaseTunerLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = ("red_scaling", "red_bias")

    def __init__(self, base_layer: nn.Module, layer_type, **kwargs) -> None:
        self.base_layer = base_layer
        self.red_scaling = nn.ParameterDict({})
        self.red_bias = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.layer_type = layer_type

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name):
         # Actual trainable parameters
        if self.layer_type == "all" or self.layer_type == "scaling":
            weight_s = torch.ones((1, self.out_features, 1, 1))
            self.red_scaling[adapter_name] = nn.Parameter(weight_s)
        if self.layer_type == "all" or self.layer_type == "bias":
            weight_b = torch.zeros((1, self.out_features, 1, 1))
            self.red_bias[adapter_name] = nn.Parameter(weight_b)


        self.to(self.get_base_layer().weight.device)
        self.set_adapter(self.active_adapters)

    def reset_red_parameters(self, adapter_name):
        if adapter_name in self.red_scaling.keys():
            # initialize learned vector with torch.ones
            nn.init.constant_(self.red_scaling[adapter_name], 1.0)
        if adapter_name in self.red_bias.keys():
            # initialize learned vector with torch.zero
            nn.init.constant_(self.red_scaling[adapter_name], 0.0)

 # 这部分代码之后再改（merge 和 unmerge）
class Linear(nn.Module, REDLayer):
    # RED implemented in a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,  # whether target module is a conv1d layer. useful while unloading later
        layer_type: str = "all", 
        **kwargs,
    ) -> None:
        super().__init__()
        REDLayer.__init__(self, base_layer, layer_type)
        self.fan_in_fan_out = fan_in_fan_out
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self._active_adapter = adapter_name
        self.update_layer(adapter_name)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.red_l.keys():
                base_layer = self.get_base_layer()
                red_l = transpose(self.red_l[active_adapter].data, self.fan_in_fan_out)
                orig_dtype = base_layer.weight.data.dtype
                if safe_merge:
                    orig_weights = base_layer.weight.data
                    orig_weights = torch.mul(orig_weights, red_l)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights.to(orig_dtype)
                else:
                    base_layer.weight.data = torch.mul(base_layer.weight.data, red_l).to(orig_dtype)

                if base_layer.bias is not None:
                    scaling = self.red_l[active_adapter].reshape(base_layer.bias.shape)
                    orig_dtype = base_layer.bias.data.dtype
                    base_layer.bias.data = torch.mul(base_layer.bias.data, scaling.data).to(orig_dtype)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        warnings.warn("Unmerge result can be inaccurate for RED.")
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.red_l.keys():
                base_layer = self.get_base_layer()
                # Add tolerace to avoid division by zero
                red_l = transpose(self.red_l[active_adapter].data, self.fan_in_fan_out) + 1e-8
                orig_dtype = base_layer.weight.data.dtype
                base_layer.weight.data = torch.div(base_layer.weight.data, red_l).to(orig_dtype)

                if base_layer.bias is not None:
                    scaling = self.red_l[active_adapter].reshape(base_layer.bias.shape)
                    orig_dtype = base_layer.bias.data.dtype
                    base_layer.bias.data = torch.div(base_layer.bias.data, scaling.data + 1e-8).to(orig_dtype)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        dtype = previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            scaling = 1
            bias = 0
            if self.layer_type == "all" or self.layer_type == "scaling":
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.red_scaling.keys():
                        continue
                    dtype = self.red_scaling[active_adapter].dtype
                    scaling *= self.red_scaling[active_adapter].flatten()

            if self.layer_type == "all" or self.layer_type == "bias":
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.red_bias.keys():
                        continue
                    dtype = self.red_bias[active_adapter].dtype
                bias = self.red_bias[active_adapter].flatten()

            result = self.base_layer(x, *args, **kwargs)
            result = result.to(dtype) * scaling + bias
        result = result.to(previous_dtype)
        return result

# 这部分代码之后再改（merge 和 unmerge）
class Conv2d(nn.Module, REDLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        layer_type: str = "all",
        **kwargs,
    ) -> None:
        super().__init__()
        REDLayer.__init__(self, base_layer, layer_type)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name

        self.update_layer(adapter_name)

    # def update_layer(self, adapter_name):
    #     # Actual trainable parameters
    #     if self.layer_type == "all" or self.layer_type == "scaling":
    #         weight_s = torch.ones((1, self.out_features, 1, 1))
    #         self.red_scaling[adapter_name] = nn.Parameter(weight_s)
    #     if self.layer_type == "all" or self.layer_type == "bias":
    #         weight_b = torch.zeros((1, self.out_features, 1, 1))
    #         self.red_bias[adapter_name] = nn.Parameter(weight_b)

    #     self.to(self.get_base_layer().weight.device)
    #     self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.red_l.keys():
                base_layer = self.get_base_layer()
                red_scaling = self.red_l[active_adapter].data
                red_scaling = red_scaling.permute(1, 0, 2, 3)

                if safe_merge:
                    output_weight = torch.mul(base_layer.weight.data, red_scaling).clone()

                    if not torch.isfinite(output_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = output_weight
                else:
                    base_layer.weight.data = torch.mul(base_layer.weight.data, red_scaling)

                if base_layer.bias is not None:
                    scaling = self.red_l[active_adapter].reshape(base_layer.bias.shape)
                    base_layer.bias.data = torch.mul(base_layer.bias.data, scaling.data)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        warnings.warn("Unmerge result can be inaccurate for RED.")
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.red_l.keys():
                base_layer = self.get_base_layer()
                # divide by RED vector. Add tolerace to avoid division by zero
                red_scaling = self.red_l[active_adapter].data

                red_scaling = red_scaling.permute(1, 0, 2, 3)
                base_layer.weight.data = torch.div(base_layer.weight.data, red_scaling + 1e-8)

                if base_layer.bias is not None:
                    scaling = self.red_l[active_adapter].reshape(base_layer.bias.shape)
                    base_layer.bias.data = torch.mul(base_layer.bias.data, scaling.data)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        dtype = previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            scaling = 1
            bias = 0
            if self.layer_type == "all" or self.layer_type == "scaling":
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.red_scaling.keys():
                        continue
                    dtype = self.red_scaling[active_adapter].dtype
                    scaling = self.red_scaling[active_adapter].flatten()

            if self.layer_type == "bias" or self.layer_type == "scaling":
                bias = self.red_bias[active_adapter].flatten()

            result = self.base_layer(x, *args, **kwargs)
            result = result.to(dtype) * scaling + bias
            
        result = result.to(previous_dtype)
        return result
