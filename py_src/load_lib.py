import os
import ctypes
from pathlib import Path



def _load_lib():
    lib_root = Path(__file__).parents[1]
    lib_path = os.path.join(str(lib_root), 'build/lib')

    path_to_so_file = os.path.join(lib_path, 'generator_api.so')
    lib = ctypes.CDLL(path_to_so_file)
    return lib



_LIB = _load_lib()


