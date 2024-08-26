# Shamelessly adapted from code (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import torch

from .dequant import dequantize_tensor
from gguf import GGMLQuantizationType
from functools import partial
import comfy
import copy

class QuantizedTensor():
    def __init__(self, data=None, tensor_type=None, tensor_shape=None, patches=[], **kwargs):
        self.tensor_type:GGMLQuantizationType = tensor_type
        self.tensor_shape:torch.Size          = tensor_shape
        self.patches:list                     = patches.copy()
        self._tensor:torch.Tensor             = None
        self._set_data(data)

    def _set_data(self, data):
        self._tensor = data if isinstance(data, torch.Tensor) or data is None else torch.as_tensor(data)  
        
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for a in args:
            if isinstance(a, QuantizedTensor):
                return_qt = QuantizedTensor(None, a.tensor_type, a.tensor_shape, a.patches)
                break

        args = [getattr(a, '_tensor', a) for a in args]
        return_qt._set_data( func(*args, **kwargs) )
        return return_qt
    
    def wrap(self, fn, *args, **kwargs):
        x = fn(*args, **kwargs)
        return QuantizedTensor(x, self.tensor_type, self.tensor_shape, self.patches) if isinstance(x, torch.Tensor) else x
    
    def __getattr__(self, __name: str):
        if __name=='patches':
            pass
        a = getattr(self._tensor, __name)
        return partial(self.wrap, a) if hasattr(a,'__call__') else a
    
    #def requires_grad_(self, *args, **kwargs):
    #    self._tensor.requires_grad_(*args, **kwargs)
    #    return self

    #def detach(self, *args, **kwargs):
    #    self._tensor.detach(*args, **kwargs)
    #    return self
    
    #def to(self, *args, **kwargs):
    #    self._tensor = self._tensor.to(*args, **kwargs)
    #    return self


class GGMLLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """
    dequant_dtype = None
    patch_dtype = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight:QuantizedTensor = None
        self.bias:QuantizedTensor   = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for k,v in state_dict.items():
            if k[len(prefix):] == "weight":
                self.weight = v
            elif k[len(prefix):] == "bias":
                self.bias = v
            else:
                missing_keys.append(k)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
        if self.weight is not None:
            weight = torch.zeros_like(self.weight, device=torch.device("meta"))
            destination[f"{prefix}weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[f"{prefix}bias"] = bias
        return

        # This would return the actual state dict
        weight, bias = self.get_weights()
        if weight is not None:
            destination[f"{prefix}weight"] = weight
        if bias is not None:
            destination[f"{prefix}bias"] = weight

    def _apply(self, fn):
        if self.weight is not None:
            try:
                self.weight = fn(self.weight)
            except TypeError:
                pass # why?
        if self.bias is not None:
            self.bias = fn(self.bias)
        super()._apply(fn)
        return self

    def get_weight(self, tensor, dtype):
        if tensor is None: return None

        # consolidate and load patches to GPU in async
        patch_list = []
        device = tensor.device
        t_move = lambda x: x.to(device) if torch.is_tensor(x) else x
        for function, patches, key in getattr(tensor, "patches", []):
            patch_list += move_patch_to_cuda(patches, device)

        # dequantize tensor while patches load
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)

        # apply patches
        if patch_list:
            if self.patch_dtype is None:
                weight = function(patch_list, weight, key)
            else:
                # for testing, may degrade image quality
                patch_dtype = dtype if self.patch_dtype == "target" else self.patch_dtype
                weight = function(patch_list, weight, key, patch_dtype)
        return weight

    def get_weights(self, dtype=torch.float16):
        weight = self.get_weight(self.weight, dtype)
        bias = self.get_weight(self.bias, dtype)
        return (weight, bias)
    

class GGMLOps(comfy.ops.manual_cast):
    """
    Dequantize weights on the fly before doing the compute
    """
    class Linear(GGMLLayer):
        comfy_cast_weights = True

        def __init__(self, *args, device=None, dtype=None, **kwargs):
            super().__init__(device=device, dtype=dtype)

        def forward(self, x:torch.Tensor):
            # lowvram hack
            device = self.weight.device
            if self.weight.device != x.device: self.to(x.device)

            if isinstance(self.weight, QuantizedTensor):
                weight, bias = self.get_weights(x.dtype)
                x = torch.nn.functional.linear(x, weight, bias)
                del weight, bias
            else:
                try:
                    x = torch.nn.functional.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)
                except:
                    raise RuntimeError()

            if self.weight.device != device: self.to(device)
            return x

def move_patch_to_cuda(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_cuda(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_cuda(x, device) for x in item]
    else:
        return item



# TODO: Temporary fix for now
import collections

class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)
        inplace_update = self.weight_inplace_update or inplace_update
        if key not in self.backup:
            self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)

        try:
            from comfy.lora import calculate_weight
        except:
            calculate_weight = self.calculate_weight

        patches = self.patches[key]
        qtype = getattr(weight, "tensor_type", None)
        if qtype not in [None, GGMLQuantizationType.F32, GGMLQuantizationType.F16]:
            if device_to is not None:
                out_weight = weight.to(device_to, copy=True)
            else:
                out_weight = weight.clone()
            
            if self.patch_on_device:
                patches = move_patch_to_cuda(patches, self.load_device)
            out_weight.patches.append((calculate_weight, patches, key))
            
        else:
            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def clone(self, *args, **kwargs):
        n = GGUFModelPatcher(self.model, self.load_device, self.offload_device, self.size, weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n.patch_on_device = getattr(self, "patch_on_device", False)
        return n