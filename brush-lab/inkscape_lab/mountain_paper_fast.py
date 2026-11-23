"""Equivalent reference equations with per-paper adsorption-factor caching."""
from pathlib import Path
import ctypes,hashlib,os,shutil,subprocess,tempfile
import numpy as np

# Use the fast backend only with the source pair checked for exact agreement.
VERIFIED_SOURCE_HASHES = {
    'mountain_paper.cpp': 'f4e4a6fd6dc4f3291dd9007c7f04f77380737475268433923b6c2c0644d67cf2',
    'mountain_paper_fast.cpp': '6de54d52efa70b2aef86a4dfc21129b8a96682639e9eecb32616a9a80b9871ea',
}


def verify_source_pair():
    source_dir = Path(__file__).parent
    hashes = {
        name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        for name in VERIFIED_SOURCE_HASHES
    }
    if hashes != VERIFIED_SOURCE_HASHES:
        raise RuntimeError('Paper backend sources changed since the equivalence check.')
    return hashes


class Entry:
 def __init__(self,capture_fine,capture_coarse):
  self.capture_fine=capture_fine;self.capture_coarse=capture_coarse
  source=Path(__file__).with_suffix('.cpp');lib=source.with_name('_mountain_paper_'+hashlib.sha256(source.read_bytes()).hexdigest()[:12]+'.so')
  if not lib.exists():
   fd,tmp=tempfile.mkstemp(suffix='.so',dir=source.parent);os.close(fd)
   subprocess.run([shutil.which(os.environ.get('CXX','g++')),'-O3','-std=c++17','-shared','-fPIC',str(source),'-o',tmp],check=True);os.replace(tmp,lib)
  self.lib=ctypes.CDLL(str(lib));self.lib.make_cache.restype=ctypes.c_void_p;self.lib.free_cache.argtypes=[ctypes.c_void_p]
  self.context=self.lib.make_cache();self.fn=self.lib.paper_step
  arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');self.fn.argtypes=[ctypes.c_void_p]+[arr]*11+[ctypes.c_int]*7+[ctypes.c_double]*15+[arr];self.fn.restype=ctypes.c_int
 def __call__(self,*args):return self.fn(self.context,*args[:-1],self.capture_fine,self.capture_coarse,args[-1])
 def __del__(self):
  if getattr(self,'context',None):self.lib.free_cache(self.context);self.context=None
def entry_native(capture_fine,capture_coarse):return Entry(capture_fine,capture_coarse)
