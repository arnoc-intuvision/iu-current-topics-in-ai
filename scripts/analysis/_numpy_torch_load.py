"""Minimal numpy reader for a torch .pt (zipfile) checkpoint state_dict."""
import io, pickle, zipfile, numpy as np

DT = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'HalfStorage': np.float16,
      'LongStorage': np.int64, 'IntStorage': np.int32, 'ByteStorage': np.uint8,
      'BoolStorage': np.bool_, 'CharStorage': np.int8, 'ShortStorage': np.int16}

def load(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split('/')[0]
    class _U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
                return _rebuild
            if mod == 'torch' and name.endswith('Storage'):
                return ('storage', DT[name])
            if mod == 'collections' and name == 'OrderedDict':
                import collections; return collections.OrderedDict
            return super().find_class(mod, name)
        def persistent_load(self, pid):
            # pid = ('storage', storage_type, key, location, numel)
            _, stype, key, _loc, numel = pid
            dtype = stype[1] if isinstance(stype, tuple) else np.float32
            raw = z.read(f'{root}/data/{key}')
            return np.frombuffer(raw, dtype=dtype, count=numel).copy()
    def _rebuild(storage, offset, size, stride, *a):
        arr = storage[offset:offset + int(np.prod(size)) if size else None]
        return np.lib.stride_tricks.as_strided(
            arr, shape=tuple(size), strides=tuple(s * arr.itemsize for s in stride)).copy()
    return _U(io.BytesIO(z.read(f'{root}/data.pkl'))).load()
