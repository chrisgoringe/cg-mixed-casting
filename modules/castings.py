
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
        if configuration is None:
            self.casts = None
            return
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
        if self.casts is None: return None

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