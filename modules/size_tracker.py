from gguf import GGMLQuantizationType
import math

class SizeTracker:
    _instance = None
    @classmethod
    def instance(cls):
        if cls._instance is None: cls._instance = SizeTracker()
        return cls._instance
    
    def __init__(self):
        self.parameters_by_type = {}

    def track(self, the_type, shape):
        if isinstance(the_type, int): the_type = GGMLQuantizationType(the_type).name
        self.parameters_by_type[the_type] = self.parameters_by_type.get(the_type,0) + math.prod(shape)

    def bits_by_type(self, the_type, default):
        if the_type in ['bfloat16', 'float16']: return 16
        if the_type in ['float8_e4m3fn', 'float8_e4m3fnuz', 'float8_e5m2', 'float8_e5m2fnuz']: return 8
        if the_type=='Q8_0': return 8
        if the_type=='Q5_1': return 5
        if the_type=='Q4_1': return 4
        return default
        
    def total_bits(self, default):
        return sum( self.bits_by_type(t, default)*self.parameters_by_type[t] for t in self.parameters_by_type )
    
    def unreduced_bits(self, default):
        return  sum( default * self.parameters_by_type[t] for t in self.parameters_by_type )
    
    def cast_summary(self):
        return "\n".join("{:>12} parameters cast to ".format(self.parameters_by_type[t]) + t for t in self.parameters_by_type)