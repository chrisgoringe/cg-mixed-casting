from .mods import QuantizedTensor, GGMLOps, GGUFModelPatcher
from safetensors.torch import load_file
import torch
import struct, json, os, yaml, math, logging
from gguf import quants, GGMLQuantizationType
import folder_paths
import comfy
from tqdm import tqdm
from functools import partial
try:
    import bitsandbytes
except:
    pass


def layer_iteratable_from_string(s):
    if isinstance(s, int): return [s,]
    if s.lower()=='all':    return range(57)
    if s.lower()=='double': return range(0,19)
    if s.lower()=='single': return range(19,57)

    def parse():
        for section in (x.strip() for x in str(s or "").split(',')):
            if section:
                a,b = (int(x.strip()) for x in section.split('-')) if '-' in section else (int(section), int(section))
                for i in range(a,b+1): yield i
    return parse()

class Castings:
    def __init__(self, configuration):
        self.casts = []
        for cast in configuration['casts']:
            layers = [x for x in layer_iteratable_from_string(cast.get('layers', None))]
            blocks = cast.get('blocks', None)
            cast_to = cast.get('castto', 'none')
            self.casts.append((layers, blocks, cast_to))
        if 'default' in configuration:
            self.default = configuration['default']
            self.casts.append((list(range(57)), None, configuration['default']))
        else:
            self.default = None

        self.specials = { k:configuration.get(k,None) for k in ['final_layer', 'guidance_in', 'img_in', 'time_in', 'txt_in', 'vector_in', 'norm'] }


    def get_layer_and_subtype(self, label) -> int:
        s = label.split(".")
        if s[0] in ['final_layer', 'guidance_in', 'img_in', 'time_in', 'txt_in', 'vector_in']: return s[0], None
        if '.norm.' in label: return 'norm', None
        for i, bit in enumerate(s):
            if bit=="double_blocks":
                if s[i+2].startswith('img'):   
                    return int(s[i+1]), 'img'
                elif s[i+2].startswith('txt'): 
                    return int(s[i+1]), 'txt'
                else:                          
                    return None, None
            elif bit=="single_blocks":
                return 19 + int(s[i+1]), 'x'
        return None, None

    def getcast(self, label) -> str:
        def result(s):
            if s is None or s=='none': return None
            if s=='default': return self.default
            return s
        
        layer, subtype = self.get_layer_and_subtype(label)

        if layer is None:  
            return self.default
        if isinstance(layer, str):
            return result(self.specials[layer])
        for (layers, blocks, cast_to) in self.casts:
            if (layer in layers) and (blocks is None or blocks==subtype):
                return result(cast_to)
            
class LoadTracker:
    _instance = None
    @classmethod
    def instance(cls):
        if cls._instance is None: cls._instance = LoadTracker()
        return cls._instance
    
    def __init__(self):
        self.parameters_by_type = {}

    def track(self, type, shape):
        self.parameters_by_type[type] = self.parameters_by_type.get(type,0) + math.prod(shape)

    def bits_by_type(self, type, default):

        if type in ['bfloat16', 'float16']: return 16
        if type in ['float8_e4m3fn', 'float8_e4m3fnuz', 'float8_e5m2', 'float8_e5m2fnuz']: return 8
        if type=='bnb8': return 8
        if type=='bnb4': return 4
        if type=='Q8_0': return 8
        if type=='Q5_1': return 5
        if type=='Q4_1': return 4
        return default
        
    def total_bits(self, default):
        return sum( self.bits_by_type(t, default)*self.parameters_by_type[t] for t in self.parameters_by_type )
    
    def unreduced_bits(self, default):
        return  sum( default * self.parameters_by_type[t] for t in self.parameters_by_type )
    
    def cast_summary(self):
        return "\n".join("{:>12} parameters cast to ".format(self.parameters_by_type[t]) + t for t in self.parameters_by_type)

def read_metadata(path) -> dict:
    with open(path, 'rb') as f:
        metasize = struct.unpack_from('Q', f.read(8))[0]
        metadata_string = struct.unpack_from(f"{metasize}s", f.read(metasize))[0].decode('utf-8')
        metadata_json = json.loads(metadata_string)
        return metadata_json.get('__metadata__', None)

def mixed_gguf_sd_loader(path:str, metadata:dict = None):
    '''
    Load a non-quantized file using a metadata casting dictionary 
    '''
    castings = Castings( metadata or read_metadata(path) )

    data:dict[str, torch.Tensor] = load_file(path)
    keys = [key for key in data]
    sd = {}
    bnb = {'bnb4':[], 'bnb8':[]}
    for key in tqdm(keys, desc="Quantising"):
        tnsr = data.pop(key)
        cast_to = castings.getcast(key)
        if cast_to is None:
            sd[key] = tnsr
        elif hasattr(GGMLQuantizationType, cast_to):
            qtype = getattr(GGMLQuantizationType, cast_to)
            if tnsr.dtype==torch.bfloat16: tnsr = tnsr.to(torch.float)
            qt = quants.quantize(tnsr.numpy(), qtype=qtype)
            sd[key] = QuantizedTensor( qt, tensor_shape=tnsr.shape, tensor_type=qtype )
            pass
        elif hasattr(torch, cast_to):
            sd[key] = tnsr.to(getattr(torch, cast_to))
        elif cast_to in bnb:
            sd[key] = tnsr
            bnb[cast_to].append(key)
        else:
            raise NotImplementedError(cast_to)
        LoadTracker.instance().track(cast_to or "not cast", tnsr.shape)
    return sd, bnb

def load_config(config_filepath):
    if os.path.splitext(config_filepath)[1]==".yaml":
        with open(config_filepath, 'r') as f: return yaml.safe_load(f)
    else:
        with open(config_filepath, 'r') as f: return json.load(f)


fullpathfor = partial(os.path.join, os.path.dirname(__file__))
class MixedGGUFLoader:
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet")]
        casting_names = [x for x in os.listdir(fullpathfor('configurations'))]
        return {
            "required": {
                "unet_name": (unet_names,),
                "casting": (casting_names,)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "func"
    CATEGORY = "advanced/loaders"

    @classmethod
    def IS_CHANGED(cls, unet_name, casting):
        path = folder_paths.get_full_path("unet", unet_name)
        config = load_config(fullpathfor('configurations',casting))
        return json.dumps({"p":path, "config":config})        

    def func(self, unet_name, casting=None):
        ops = GGMLOps()
        ops.Linear.dequant_dtype = None
        ops.Linear.patch_dtype = None

        path = folder_paths.get_full_path("unet", unet_name)
        config = load_config(fullpathfor('configurations',casting))
        
        sd, bnb = mixed_gguf_sd_loader(path, metadata=config)

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}
        )
        mfac =  LoadTracker.instance().total_bits(16) / LoadTracker.instance().unreduced_bits(16)

        def apply_bnb(model, key, clazz):
            bits = key.split(".")
            if (bits[-1]!='weight'): return
            parent = model.model.diffusion_model
            for b in bits[:-2]: parent = getattr(parent, b)
            child_name = bits[-2]
            child = getattr(parent, child_name)
            weight = getattr(child, 'weight')
            bias = getattr(child, 'bias', None)
            sd = { "weight": weight }
            if bias is not None: sd["bias"] = bias
            setattr(parent, child_name, clazz( weight.shape[1], weight.shape[0], bias=(bias is not None)))
            getattr(parent, child_name).load_state_dict(sd)
            parent.cuda()
            parent.cpu()
            del child, weight, bias, sd

        if bnb['bnb8']:
            for k in tqdm(bnb['bnb8'], desc="bitsandbytes 8"): apply_bnb(model, k, bitsandbytes.nn.Linear8bitLt)
        if bnb['bnb4']:
            for k in tqdm(bnb['bnb4'], desc="bitsandbytes 4"): apply_bnb(model, k, bitsandbytes.nn.Linear4bit)

        logging.info(LoadTracker.instance().cast_summary())
        logging.info('Quantization reduced model to {:>5.2f}% of size'.format(100*mfac))
        model = GGUFModelPatcher.clone(model)
        #model.model.memory_usage_factor *= mfac
        model.patch_on_device = False
        return (model,)