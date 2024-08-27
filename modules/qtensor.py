from functools import partial
from .dequant import dequantize_tensor
import torch
from gguf import GGMLQuantizationType

class QuantizedTensor():
    def __init__(self, data=None, tensor_type=None, tensor_shape=None, patches=[], **kwargs):
        self.tensor_type:GGMLQuantizationType = tensor_type
        self.tensor_shape:torch.Size          = tensor_shape
        self.patches:list                     = patches.copy()
        self._tensor:torch.Tensor             = None
        self._set_data(data)

    def dequantized(self, dtype, dequant_dtype=None):
        return dequantize_tensor(self._tensor, dtype, dequant_dtype)

    @property
    def tensor_description(self):
        return torch.tensor((int(self.tensor_type), *self.tensor_shape), device="cpu")
    
    @classmethod
    def load_from(cls, description, tnsr):
        return QuantizedTensor( data=tnsr, tensor_type=int(description[0]), tensor_shape=torch.Size(description[1:]))

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
        a = getattr(self._tensor, __name)
        return partial(self.wrap, a) if hasattr(a,'__call__') else a