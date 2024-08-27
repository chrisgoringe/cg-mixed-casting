from .modules.mods import QuantizedTensor, GGMLOps, GGUFModelPatcher, GGMLLayer
from .modules.dequant import dequantize_tensor
from .modules.castings import Castings
from .modules.size_tracker import SizeTracker
from safetensors.torch import load_file, save_file
import torch
import json, os, yaml, logging
from gguf import quants, GGMLQuantizationType
import folder_paths
import comfy
from tqdm import tqdm
from functools import partial

def mixed_gguf_sd_loader(path:str, config:dict=None, recast:bool=False):
    '''
    Load a non-quantized file using a metadata casting dictionary 
    '''
    data:dict[str, torch.Tensor] = load_file(path)

    described = {}
    keys      = []
    for key in data:
        if key.endswith('_tensor_description'):
            described[key[:-19]] = data.get(key)
        else:
            keys.append(key)

    if described: print("Loading prequantised file.")

    castings = Castings( config )

    sd = {}
    for key in tqdm(keys, desc="Quantising"):
        tnsr = data.pop(key)
        cast_to = castings.getcast(key)
        if key in described:
            description = described[key]
            qtnsr:QuantizedTensor = QuantizedTensor.load_from(description=description, tnsr=tnsr)
            if recast and cast_to is not None:
                print(f"Recasting {key}")
                tnsr = dequantize_tensor(qtnsr, torch.bfloat16)
            else:
                sd[key] = qtnsr
                SizeTracker.instance().track(qtnsr.tensor_type, qtnsr.tensor_shape)
        if key not in sd:
            if cast_to is None:
                sd[key] = tnsr
            elif hasattr(GGMLQuantizationType, cast_to):
                qtype = getattr(GGMLQuantizationType, cast_to)
                if tnsr.dtype==torch.bfloat16: tnsr = tnsr.to(torch.float)
                try:
                    qt = quants.quantize(tnsr.numpy(), qtype=qtype)
                except TypeError:
                    qt = quants.quantize(tnsr.to(torch.float).numpy(), qtype=qtype)
                sd[key] = QuantizedTensor( qt, tensor_shape=tnsr.shape, tensor_type=qtype )
                pass
            elif hasattr(torch, cast_to):
                sd[key] = tnsr.to(getattr(torch, cast_to))
            else:
                raise NotImplementedError(cast_to)
            SizeTracker.instance().track(cast_to or "not cast", tnsr.shape)
    return sd

def load_config(config_filepath):
    if os.path.splitext(config_filepath)[1]==".yaml":
        with open(config_filepath, 'r') as f: return yaml.safe_load(f)
    else:
        with open(config_filepath, 'r') as f: return json.load(f)


fullpathfor = partial(os.path.join, os.path.dirname(__file__))
class MixedGGUFLoader:
    NO_CONFIG = "[none]"
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet")]
        casting_names = [x for x in os.listdir(fullpathfor('configurations'))] + [MixedGGUFLoader.NO_CONFIG,]
        return {
            "required": {
                "unet_name": (unet_names,),
                "casting": (casting_names,),
                "recast": (["no","yes"],)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "func"
    CATEGORY = "advanced/loaders"

    @classmethod
    def IS_CHANGED(cls, casting, **kwargs):
        config = load_config(fullpathfor('configurations',casting))
        return json.dumps({"config":config})        

    def func(self, unet_name, casting, recast):
        ops = GGMLOps()
        ops.Linear.dequant_dtype = None
        ops.Linear.patch_dtype = None

        path = folder_paths.get_full_path("unet", unet_name)
        config = load_config(fullpathfor('configurations',casting)) if casting!=MixedGGUFLoader.NO_CONFIG else None
        
        sd = mixed_gguf_sd_loader(path, config=config, recast=(recast=='yes'))

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}
        )
        mfac =  SizeTracker.instance().total_bits(16) / SizeTracker.instance().unreduced_bits(16)
        logging.info(SizeTracker.instance().cast_summary())
        logging.info('Quantization reduced model to {:>5.2f}% of size'.format(100*mfac))
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = False
        return (model,)
    
class MixedGGUFSaver:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
            "model": ("MODEL",{}),
            "save_as": ("STRING", {"default":"model"}),
        }}
    FUNCTION = "func"
    CATEGORY = "advanced/savers"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def func(self, model, save_as:str):
        if not save_as.endswith(".safetensors"): save_as += ".safetensors"
        filepath = os.path.join(folder_paths.get_output_directory(),*save_as.split("/"))
        old_mode = GGMLLayer.return_sd_mode
        GGMLLayer.return_sd_mode = GGMLLayer.QUANTIZED_SD
        sd = model.model.diffusion_model.state_dict()
        GGMLLayer.return_sd_mode = old_mode
        save_file(sd, filepath)
        return ()