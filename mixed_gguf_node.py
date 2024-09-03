from .modules.mods import QuantizedTensor, GGMLOps, GGUFModelPatcher, GGMLLayer
from .modules.castings import Castings
from safetensors.torch import load_file, save_file
import torch
import json, os, yaml, logging
from gguf import GGMLQuantizationType, GGUFReader
import folder_paths
import comfy
from tqdm import tqdm
from functools import partial
from comfy.model_management import module_size

if "unet_gguf" not in folder_paths.folder_names_and_paths:
    orig = folder_paths.folder_names_and_paths.get("diffusion_models", folder_paths.folder_names_and_paths.get("unet", [[], set()]))
    folder_paths.folder_names_and_paths["unet_gguf"] = (orig[0], {".gguf"})

def mixed_gguf_sd_loader(path:str, config:dict=None, gguf_file:str=None):
    data:dict[str, torch.Tensor] = load_file(path)
    castings = Castings( config )

    described = { key[:-19]:value for key, value in data.items() if key.endswith('_tensor_description') }
    keys = [key for key in data if not key.endswith('_tensor_description')]

    if described: print("Loading prequantised file - ignoring configuration file")

    sd = {}
    topatch = []
    for key in tqdm(keys, desc="Quantising"):
        tnsr = data.pop(key)
        if described:
            if key in described: sd[key] = QuantizedTensor.load_from_description(description=described[key], tnsr=tnsr)
            else:                sd[key] = tnsr
        else:
            cast_to = castings.getcast(key) 
            if cast_to is None:      
                sd[key] = tnsr
            elif cast_to == 'patch':
                topatch.append(key)
            elif ttype:=getattr(torch, cast_to, None): 
                sd[key] = tnsr.to(ttype)
            elif qtype:=getattr(GGMLQuantizationType, cast_to, None):
                sd[key] = QuantizedTensor.from_unquantized_tensor( tnsr, tensor_type=qtype )
            else: 
                raise NotImplementedError(cast_to)
            
    return apply_patches(sd, gguf_file, topatch)

def apply_patches(sd, gguf_file, topatch:list[str]):
    if topatch: 
        if gguf_file == MixedGGUFLoader.CHOOSE_NONE:
            raise Exception("patch used in configuration but no gguf file selected")
        reader = GGUFReader(folder_paths.get_full_path("unet", gguf_file))
        for tensor in reader.tensors:
            if (key := str(tensor.name)) in topatch:
                sd[key] = QuantizedTensor.load_from_reader_tensor(tensor)
                topatch.remove(key)
    if topatch:
        raise Exception(f"Failed to patch {topatch}")
    return sd

def load_config(config_filepath):
    with open(config_filepath, 'r') as f: return yaml.safe_load(f)

class SdModeContext:
    UNQUANTIZED_SIZE = GGMLLayer.FAKE_DEQUANTIZED_SD
    QUANTIZED_SIZE   = GGMLLayer.FAKE_SD
    QUANTIZED_SD     = GGMLLayer.QUANTIZED_SD
    def __init__(self, mode):
        self.mode = mode
    def __enter__(self):
        self.mode_was = GGMLLayer.return_sd_mode
        GGMLLayer.return_sd_mode = self.mode
    def __exit__(self, type, value, traceback):
        GGMLLayer.return_sd_mode = self.mode_was

def measure_model(m:torch.nn.Module):
    with SdModeContext(SdModeContext.QUANTIZED_SIZE): 
        quantized_size = module_size(m)
    with SdModeContext(SdModeContext.UNQUANTIZED_SIZE):
        unquantised_size = module_size(m)
    percentage = 100*quantized_size/unquantised_size
    logging.info(f"Full model size {unquantised_size:n}.  Quantised to {quantized_size:n} ({percentage:5.2f}%)")

fullpathfor = partial(os.path.join, os.path.dirname(__file__))

class MixedGGUFLoader:
    CHOOSE_NONE = "[none]"
    @classmethod
    def INPUT_TYPES(s):
        unet_names    = [x for x in folder_paths.get_filename_list("unet")]
        casting_names = [MixedGGUFLoader.CHOOSE_NONE,] + [x for x in os.listdir(fullpathfor('configurations'))]
        gguf_names    = [MixedGGUFLoader.CHOOSE_NONE,] + [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "model": (unet_names,),
                "casting": (casting_names,)
            },
            "optional": { "optional_gguf_file": (gguf_names,), },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "func"
    CATEGORY = "advanced/loaders"

    @classmethod
    def IS_CHANGED(cls, casting, **kwargs):
        config = load_config(fullpathfor('configurations',casting))
        for k in kwargs: config[k] = kwargs[k]
        return json.dumps(config)        

    def func(self, model, casting, optional_gguf_file=None):
        ops = GGMLOps()
        ops.Linear.dequant_dtype = None
        ops.Linear.patch_dtype = None

        path = folder_paths.get_full_path("unet", model)
        config = load_config(fullpathfor('configurations',casting)) if casting!=MixedGGUFLoader.CHOOSE_NONE else None
        
        sd = mixed_gguf_sd_loader(path, config=config, gguf_file=optional_gguf_file)
        model = comfy.sd.load_diffusion_model_state_dict( sd, model_options={"custom_operations": ops} )
        measure_model(model.model.diffusion_model)
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
        with SdModeContext(SdModeContext.QUANTIZED_SD):
            save_file(tensors  = model.model.diffusion_model.state_dict(), 
                      filename = os.path.join(folder_paths.get_output_directory(),*save_as.split("/")))
        return ()