import struct, json

def read_metadata(path) -> dict:
    with open(path, 'rb') as f:
        metasize = struct.unpack_from('Q', f.read(8))[0]
        metadata_string = struct.unpack_from(f"{metasize}s", f.read(metasize))[0].decode('utf-8')
        metadata_json = json.loads(metadata_string)
        return metadata_json.get('__metadata__', None)