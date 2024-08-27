
from .mixed_gguf_node import MixedGGUFLoader, MixedGGUFSaver
NODE_CLASS_MAPPINGS = {
    "Mixed Cast Flux Loader" : MixedGGUFLoader,
    "Mixed Cast Flux Saver"  : MixedGGUFSaver
} 

VERSION = "0.2"