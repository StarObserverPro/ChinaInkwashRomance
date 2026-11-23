"""Conservative surface particle interception at the paper-entry boundary."""
from pathlib import Path
import ctypes,hashlib,os,shutil,subprocess,tempfile
import numpy as np

def entry_native(capture_fine,capture_coarse):
    if not (0<=capture_fine<=1 and 0<=capture_coarse<=1):raise ValueError('entry capture outside [0,1]')
    source=Path(__file__).with_suffix('.cpp')
    lib=source.with_name('_technique_paper_'+hashlib.sha256(source.read_bytes()).hexdigest()[:12]+'.so')
    if not lib.exists():
        compiler=shutil.which(os.environ.get('CXX','g++'))
        if compiler is None:raise RuntimeError('Paper transport needs a C++17 compiler')
        fd,name=tempfile.mkstemp(prefix=lib.stem+'-',suffix='.so',dir=source.parent);os.close(fd)
        subprocess.run([compiler,'-O3','-std=c++17','-shared','-fPIC',str(source),'-o',name],check=True)
        os.replace(name,lib)
    fn=ctypes.CDLL(str(lib)).paper_step
    arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    fn.argtypes=[arr]*11+[ctypes.c_int]*7+[ctypes.c_double]*15+[arr];fn.restype=ctypes.c_int
    def advance(*args):return fn(*args[:-1],capture_fine,capture_coarse,args[-1])
    return advance
