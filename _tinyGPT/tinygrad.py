# this is a single file representing each of the modules in the
# root directory of the tinygrad project


# tinygrad/__init__.py

from tinygrad.tensor import Tensor            # noqa: F401
from tinygrad.features.jit import TinyJit     # noqa: F401
from tinygrad.shape.symbolic import Variable  # noqa: F401
from tinygrad.dtype import dtypes             # noqa: F401
from tinygrad.ops import GlobalCounters       # noqa: F401
from tinygrad.device import Device            # noqa: F401


# tinygrad/device.py

from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Optional, Dict, Tuple, ClassVar
import importlib, inspect, functools, pathlib, time, ctypes
from tinygrad.dtype import DType, ImageDType
from tinygrad.helpers import ansilen, DEBUG, getenv, colored, BEAM, NOOPT, all_int, to_function_name, from_mv, flat_mv, diskcache_get, diskcache_put
from tinygrad.helpers import prod
from tinygrad.shape.symbolic import Variable, sym_infer, sint
from tinygrad.ops import LazyOp, get_lazyop_info, GlobalCounters
from dataclasses import dataclass

if TYPE_CHECKING:
  from tinygrad.codegen.linearizer import Linearizer
  from tinygrad.codegen.kernel import LinearizerOptions

# **************** Device ****************

class _Device:
  def __init__(self) -> None: self._devices: List[str] = [x.stem[len("ops_"):].upper() for x in (pathlib.Path(__file__).parent/"runtime").iterdir() if x.stem.startswith("ops_")]  # noqa: E501
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def _canonicalize(self, device:str) -> str: return (device.split(":", 1)[0].upper() + ((":"+device.split(":", 1)[1]) if ':' in device else '')).replace(":0", "")   # noqa: E501
  # NOTE: you can't cache canonicalize in case Device.DEFAULT changes
  def canonicalize(self, device:Optional[str]) -> str: return self._canonicalize(device) if device is not None else Device.DEFAULT
  def __getitem__(self, ix:str) -> Compiled: return self.__get_canonicalized_item(self.canonicalize(ix))
  @functools.lru_cache(maxsize=None)  # this class is a singleton, pylint: disable=method-cache-max-size-none
  def __get_canonicalized_item(self, ix:str) -> Compiled:
    x = ix.split(":")[0].upper()
    return [cls for cname, cls in inspect.getmembers(importlib.import_module(f'tinygrad.runtime.ops_{x.lower()}')) if (cname.lower() == x.lower() + "device") and x in self._devices][0](ix)  # noqa: E501
  @functools.cached_property
  def DEFAULT(self) -> str:
    device_from_env: Optional[str] = functools.reduce(lambda val, ele: ele if getenv(ele) == 1 else val, self._devices, None)   # type: ignore
    if device_from_env: return device_from_env
    for device in ["METAL", "HIP", "CUDA", "GPU", "LLVM", "CLANG"]:
      try:
        if self[device]: return device
      except Exception: pass
    raise RuntimeError("no usable devices")
Device = _Device()

# **************** base Runner + helpers ****************

class JITRunner:
  def __init__(self):
    self.op_estimate:sint = 0
    self.mem_estimate:sint = 0
  def exec(self, rawbufs:List[Buffer], var_vals:Optional[Dict[Variable, int]]=None) -> Optional[float]:
    var_vals = var_vals if var_vals is not None else {}
    from tinygrad.features.jit import CacheCollector
    et = self(rawbufs, var_vals)
    CacheCollector.add(self, rawbufs, var_vals)
    return et
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    raise NotImplementedError("override this")

def update_stats(name:str, op_estimate:sint, mem_estimate:sint, var_vals: Optional[Dict[Variable, int]], et: Optional[float], buf_count:int, jit=False, num_kernels=1, lra: Optional[Dict]=None, device:str="", first_run=False):  # noqa: E501
  if var_vals is None: var_vals = {}
  op_estimate = sym_infer(op_estimate, var_vals)
  mem_estimate = sym_infer(mem_estimate, var_vals)
  GlobalCounters.kernel_count += num_kernels
  GlobalCounters.global_ops += op_estimate
  GlobalCounters.global_mem += mem_estimate
  if et is not None: GlobalCounters.time_sum_s += et
  if DEBUG >= 2:
    ptm = (colored(f"{et*1e3:9.2f}ms", "yellow") if et > 0.01 else f"{et*1e6:9.2f}us") if et is not None else ""
    print(f"{colored(f'*** {device[:7]:7s} {GlobalCounters.kernel_count:4d}', ('magenta' if num_kernels == 1 else 'CYAN') if jit else ('green' if first_run else None))} {name+' '*(38-ansilen(name))} arg {buf_count:3d} mem {GlobalCounters.mem_used/1e9:5.2f} GB " +  # noqa: E501
          (str() if et is None else f"tm {ptm}/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({op_estimate/((et or 1e-20)*1e9):8.2f} GFLOPS, {mem_estimate/((et or 1e-20)*1e9):7.2f} GB/s)"))  # noqa: E501

# **************** Buffer / Allocator ****************

@dataclass(frozen=True, eq=True)
class BufferOptions:
  image: Optional[ImageDType] = None
  uncached: bool = False
  host: bool = False
  signal: bool = False

class Buffer:
  def __init__(self, device:str, size:int, dtype:DType, opaque:Any=None, options:Optional[BufferOptions]=None):
    assert isinstance(dtype, DType)
    if isinstance(dtype, ImageDType): options = BufferOptions(image=dtype) # TODO: image hack shouldn't be here. where should it be?
    self.device, self.size, self.dtype, self.d, self.options = device, size, dtype, Device[device], options
    self.allocator = self.d.allocator
    self._buf = opaque if opaque is not None else self.allocator.alloc(self.nbytes, options)
    # TODO: mem_used for all devices
    if not self.device.startswith("DISK"): GlobalCounters.mem_used += self.nbytes
  @property
  def nbytes(self): return self.size*self.dtype.itemsize
  def __del__(self):
    if not hasattr(self, '_buf'): return # happens when __init__ has raised exception
    if not self.device.startswith("DISK"): GlobalCounters.mem_used -= self.nbytes
    self.allocator.free(self._buf, self.nbytes, self.options)
  def __repr__(self): return f"<buf device:{self.device} size:{self.size} dtype:{self.dtype}" + (">" if self.options is None else f"{self.options=}>")
  def as_buffer(self, allow_zero_copy=False, force_zero_copy=False) -> memoryview:
    # zero copy with as_buffer (disabled by default due to use after free)
    if (force_zero_copy or allow_zero_copy) and hasattr(self.allocator, 'as_buffer'): return self.allocator.as_buffer(self._buf)
    assert not force_zero_copy, "force zero copy was passed, but copy is required"
    return self.copyout(memoryview(bytearray(self.size*self.dtype.itemsize)))
  def copyin(self, mv:memoryview):
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    self.allocator.copyin(self._buf, mv)
    return self
  def copyout(self, mv:memoryview) -> memoryview:
    mv = flat_mv(mv)
    assert len(mv) == self.nbytes, f"size mismatch, {len(mv)=} != {self.dtype=} {self.size=}"
    self.allocator.copyout(mv, self._buf)
    return mv

class BufferCopy(JITRunner):
  def copy(self, dest, src): dest.copyin(src.as_buffer(allow_zero_copy=True))  # may allocate a CPU buffer depending on allow_zero_copy
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False):
    dest, src = rawbufs[0:2]
    assert dest.size == src.size and dest.dtype == src.dtype, f"buffer copy mismatch, {dest.size} != {src.size}, {dest.dtype} != {src.dtype}"
    st = time.perf_counter()
    self.copy(dest, src)
    et = None
    if wait or DEBUG >= 2:
      dest.d.synchronize()
      et = time.perf_counter() - st
    total_sz = dest.size*dest.dtype.itemsize
    if total_sz >= 1e6: name = f"{type(self).__name__[6:].lower()} {total_sz/1e6:7.2f}M, {dest.device[:7]:>7s} <- {src.device[:7]:7s}"
    else: name = f"{type(self).__name__[6:].lower()} {total_sz:8d}, {dest.device[:7]:>7s} <- {src.device[:7]:7s}"
    update_stats(colored(name, "yellow"), 0, total_sz, {}, et, 2, jit, device=dest.device)

class BufferRead(BufferCopy):
  def copy(self, dest, src):
    if hasattr(dest.allocator, 'copy_from_fd') and src.device.startswith("DISK") and src.nbytes >= 4096 and src._buf.ud.fd is not None:
      dest.allocator.copy_from_fd(dest._buf, src._buf.ud.fd, src._buf.offset, src.nbytes)
    elif hasattr(dest.allocator, 'as_buffer'):
      # fast(ish) path, uses readinto in diskbuffers
      src.allocator.copyout(dest.allocator.as_buffer(dest._buf), src._buf)
    else: super().copy(dest, src)

class BufferXfer(BufferCopy):
  def copy(self, dest, src):
    if hasattr(dest.allocator.device, "track_cross_buffer") and hasattr(src.allocator, "track_cross_device"):
      dest.allocator.device.track_cross_buffer.append(src)
      src.allocator.track_cross_device.add(dest.allocator.device)
    dest.allocator.transfer(dest._buf, src._buf, dest.nbytes, src_dev=src.allocator.device, dest_dev=dest.allocator.device)

# TODO: size, dest, src are the same type. can we enforce this?
class Allocator:
  def alloc(self, size:int, options:Optional[BufferOptions]=None):
    assert not isinstance(size, int) or size > 0, f"alloc size must be positve, getting {size}"
    return self._alloc_with_options(size, options) if options is not None else self._alloc(size)
  def _alloc(self, size:int): raise NotImplementedError("need alloc")
  def _alloc_with_options(self, size:int, options:BufferOptions): return self._alloc(size)  # TODO: override this if you support options
  def free(self, opaque, size:int, options:Optional[BufferOptions]=None): self._free(opaque)
  def _free(self, opaque): pass  # if opaque is a Python object, you don't need a free
  def copyin(self, dest, src:memoryview): raise NotImplementedError("need copyin")
  def copyout(self, dest:memoryview, src): raise NotImplementedError("need copyout")

class LRUAllocator(Allocator):  # pylint: disable=abstract-method
  def __init__(self): self.cache: Dict[Tuple[int, Optional[BufferOptions]], Any] = defaultdict(list)
  def alloc(self, size:int, options:Optional[BufferOptions]=None):
    if len(c := self.cache[(size, options)]): return c.pop()
    try: return super().alloc(size, options)
    except (RuntimeError, MemoryError):
      self.free_cache()
      return super().alloc(size, options)
  def free_cache(self):
    for opaques in self.cache.values():
      for opaque in opaques: self._free(opaque)
      opaques.clear()
  def free(self, opaque:Any, size:int, options:Optional[BufferOptions]=None):
    if getenv("LRU", 1) and (options is None or not options.signal): self.cache[(size, options)].append(opaque)
    else: self._free(opaque)

class _MallocAllocator(LRUAllocator):
  def _alloc(self, size:int): return (ctypes.c_uint8 * size)()
  def as_buffer(self, src) -> memoryview: return flat_mv(memoryview(src))
  def copyin(self, dest, src:memoryview): ctypes.memmove(dest, from_mv(src), len(src))
  def copyout(self, dest:memoryview, src): ctypes.memmove(from_mv(dest), src, len(dest))
MallocAllocator = _MallocAllocator()

# **************** for Compiled Devices ****************

class Compiler:
  linearizer_opts: ClassVar[LinearizerOptions]
  def __init__(self, cachekey:Optional[str]=None): self.cachekey = None if getenv("DISABLE_COMPILER_CACHE") else cachekey
  def render(self, name:str, uops) -> str: raise NotImplementedError("need a render function")
  def compile(self, src:str) -> bytes: raise NotImplementedError("need a compile function")
  def compile_cached(self, src:str) -> bytes:
    if self.cachekey is None or (lib := diskcache_get(self.cachekey, src)) is None:
      lib = self.compile(src)
      if self.cachekey is not None: diskcache_put(self.cachekey, src, lib)
    return lib

class CompiledASTRunner(JITRunner):
  def __init__(self, name:str, prg:str, device:Compiled, global_size:Optional[List[int]]=None, local_size:Optional[List[int]]=None,
               variables:Optional[List[Variable]]=None, op_estimate:sint=0, mem_estimate:sint=0, precompiled:Optional[bytes]=None):
    super().__init__()
    if DEBUG >= 4: print(prg)
    if global_size is not None: global_size = global_size + [1]*(3-len(global_size))
    if local_size is not None: local_size = local_size + [1]*(3-len(local_size))
    self.name, self.display_name, self.prg, self.device, self.global_size, self.local_size, self.first_run = \
      to_function_name(name), name, prg, device, global_size, local_size, True
    assert self.device.compiler is not None, "compiler is reuired to make an AST kernel"
    lib:bytes = precompiled if precompiled is not None else self.device.compiler.compile_cached(prg)
    self.lib, self.clprg = lib, self.device.runtime(self.name, lib)
    self.vars: List[Variable] = [] if variables is None else variables
    self.op_estimate, self.mem_estimate = op_estimate, mem_estimate

  def launch_dims(self, var_vals):
    global_size = [sym_infer(sz, var_vals) for sz in self.global_size] if self.global_size is not None else self.global_size
    local_size = [sym_infer(sz, var_vals) for sz in self.local_size] if self.local_size is not None else self.local_size
    return global_size, local_size

  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False, do_update_stats=True) -> Optional[float]:
    global_size, local_size = self.launch_dims(var_vals)
    if global_size is not None and local_size is None and all_int(self.global_size): # type: ignore[arg-type]
      # TODO: this is copied from get_program
      from tinygrad.features.search import optimize_local_size
      local_size = self.local_size = optimize_local_size(self.clprg, global_size, rawbufs)
      global_size = self.global_size = [g//l if g%l == 0 else g/l for g,l in zip(global_size, local_size)]
    lra = {}
    if global_size: lra['global_size'] = global_size
    if local_size: lra['local_size'] = local_size
    et = self.clprg(*[x._buf for x in rawbufs], **lra, vals=tuple(var_vals[k] for k in self.vars), wait=wait or DEBUG>=2)
    if do_update_stats: update_stats(self.display_name, self.op_estimate, self.mem_estimate, var_vals, et, len(rawbufs), jit,
                                     lra=lra, device=self.device.dname, first_run=self.first_run)
    self.first_run = False
    return et

class MultiDeviceJITGraph(JITRunner):
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    raise NotImplementedError("override this")

class Compiled:
  def __init__(self, device:str, allocator:Allocator, compiler:Optional[Compiler], runtime, graph=None):
    self.dname, self.allocator, self.compiler, self.runtime, self.graph = device, allocator, compiler, runtime, graph
  def synchronize(self): pass  # override this in your device

  def to_program(self, k:Linearizer) -> CompiledASTRunner:
    assert self.compiler is not None, "compiler is required to run AST"
    k.linearize()
    info = get_lazyop_info(k.ast)
    ops, mem = k.uops.flops_mem()
    run_count = prod((k.global_size if k.global_size else []) + (k.local_size if k.local_size else []))
    # NOTE: we use min here to ignore the indexing FLOPS
    ret = CompiledASTRunner(k.name, self.compiler.render(to_function_name(k.name), k.uops.uops), self, k.global_size, k.local_size,
                            k.uops.vars(), min(info.flops, ops * run_count), min(info.mem_estimate, mem * run_count))
    return ret

  def get_linearizer(self, ast:LazyOp) -> Linearizer:
    assert self.compiler is not None, "compiler is required to build AST"
    if DEBUG >= 3:
      from tinygrad.features.graph import print_tree
      print_tree(ast)
    from tinygrad.codegen.linearizer import Linearizer
    k = Linearizer(ast, self.compiler.linearizer_opts)
    k.required_optimizations()
    if not NOOPT:
      if not (used_tensor_cores:=k.apply_tensor_cores(getenv("TC", 1))): k.hand_coded_optimizations()
      if BEAM >= 1:
        lins = [(("tc" if used_tensor_cores else "hc"), k)]
        if used_tensor_cores:
          lins.append(("hc", Linearizer(ast, self.compiler.linearizer_opts)))
          lins[-1][1].hand_coded_optimizations()
        kb = Linearizer(ast, self.compiler.linearizer_opts)
        kb.required_optimizations()
        from tinygrad.features.search import beam_search, time_linearizer, bufs_from_lin
        test_rawbuffers = bufs_from_lin(kb)    # allocate scratch buffers for optimization
        lins.append((f"beam{BEAM.value}", beam_search(kb, test_rawbuffers, BEAM.value, bool(getenv("BEAM_ESTIMATE", 1)))))
        timed = sorted([(nm, tk, time_linearizer(tk, test_rawbuffers, allow_test_size=False, clear_l2=True)) for nm, tk in lins], key=lambda x: x[2])
        if DEBUG >= 1: print("  <  ".join(f"{nm:6s} : {lin.colored_shape(30, dense=True)} : {tm*1e6:8.2f} us" for nm, lin, tm in timed))
        k = timed[0][1]
    return k

  @functools.lru_cache(None)    # pylint: disable=method-cache-max-size-none
  def get_runner(self, ast:LazyOp) -> CompiledASTRunner: return self.to_program(self.get_linearizer(ast))


# tinygrad/dtype.py
  
from typing import Final, Optional, ClassVar, Set, Tuple, Dict, Union
from dataclasses import dataclass
import numpy as np  # TODO: remove numpy
import functools

Scalar = Union[float, int, bool]

@dataclass(frozen=True, order=True)
class DType:
  priority: int  # this determines when things get upcasted
  itemsize: int
  name: str
  fmt: Optional[str]
  count: int
  def __repr__(self): return f"dtypes.{'_'*(c:=self.count!=1)}{INVERSE_DTYPES_DICT[self.name if not c else self.scalar().name]}{str(self.count)*c}"
  def vec(self, sz:int):
    assert sz > 1 and self.count == 1, f"can't vectorize {self} with size {sz}"
    return DType(self.priority, self.itemsize*sz, f"{INVERSE_DTYPES_DICT[self.name]}{sz}", None, sz)
  def scalar(self): return DTYPES_DICT[self.name[:-len(str(self.count))]] if self.count > 1 else self
  # TODO: someday this will be removed with the "remove numpy" project
  @property
  def np(self) -> Optional[type]: return np.dtype(self.fmt).type if self.fmt is not None else None

# dependent typing?
@dataclass(frozen=True, repr=False)
class ImageDType(DType):
  shape: Tuple[int, ...]   # arbitrary arg for the dtype, used in image for the shape
  base: DType
  def scalar(self): return self.base
  def vec(self, sz:int): return self.base.vec(sz)
  def __repr__(self): return f"dtypes.{self.name}({self.shape})"

# @dataclass(frozen=True, init=False, repr=False, eq=False)
class PtrDType(DType):
  def __init__(self, dt:DType): super().__init__(dt.priority, dt.itemsize, dt.name, dt.fmt, dt.count)
  def __repr__(self): return f"ptr.{super().__repr__()}"
  def __hash__(self): return super().__hash__()
  def __eq__(self, dt): return self.priority==dt.priority and self.itemsize==dt.itemsize and self.name==dt.name and self.count==dt.count
  def __ne__(self, dt): return not (self == dt)

def cast_scalar(scalar: Scalar, dtype:DType):
  return int(scalar) if dtypes.is_int(dtype) else float(scalar) if dtypes.is_float(dtype) else bool(scalar)

class dtypes:
  @staticmethod
  def is_float(x: DType) -> bool: return x.scalar() in (dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64)
  @staticmethod # static methds on top, or bool in the type info will refer to dtypes.bool
  def is_int(x: DType) -> bool: return x.scalar() in (dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64) or dtypes.is_unsigned(x)
  @staticmethod
  def is_unsigned(x: DType) -> bool: return x.scalar() in (dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64)
  @staticmethod
  def from_np(x: type) -> DType: return DTYPES_DICT[np.dtype(x).name]
  @staticmethod  # NOTE: isinstance(True, int) is True in python
  def from_py(x) -> DType: return dtypes.default_float if isinstance(x, float) else dtypes.bool if isinstance(x, bool) else dtypes.default_int
  @staticmethod
  def fields() -> Dict[str, DType]: return DTYPES_DICT
  bool: Final[DType] = DType(0, 1, "bool", '?', 1)
  int8: Final[DType] = DType(1, 1, "char", 'b', 1)
  uint8: Final[DType] = DType(2, 1, "unsigned char", 'B', 1)
  int16: Final[DType] = DType(3, 2, "short", 'h', 1)
  uint16: Final[DType] = DType(4, 2, "unsigned short", 'H', 1)
  int32: Final[DType] = DType(5, 4, "int", 'i', 1)
  uint32: Final[DType] = DType(6, 4, "unsigned int", 'I', 1)
  int64: Final[DType] = DType(7, 8, "long", 'l', 1)
  uint64: Final[DType] = DType(8, 8, "unsigned long", 'L', 1)
  float16: Final[DType] = DType(9, 2, "half", 'e', 1)
  # bfloat16 has higher priority than float16, so least_upper_dtype(dtypes.int64, dtypes.uint64) = dtypes.float16
  bfloat16: Final[DType] = DType(10, 2, "__bf16", None, 1)
  float32: Final[DType] = DType(11, 4, "float", 'f', 1)
  float64: Final[DType] = DType(12, 8, "double", 'd', 1)

  # dtype aliases
  half = float16; float = float32; double = float64 # noqa: E702
  uchar = uint8; ushort = uint16; uint = uint32; ulong = uint64 # noqa: E702
  char = int8; short = int16; int = int32; long = int64 # noqa: E702

  # NOTE: these are image dtypes
  @staticmethod
  def imageh(shp): return ImageDType(100, 2, "imageh", 'e', 1, shape=shp, base=dtypes.float32)
  @staticmethod
  def imagef(shp): return ImageDType(100, 4, "imagef", 'f', 1, shape=shp, base=dtypes.float32)

  default_float: ClassVar[DType] = float32
  default_int: ClassVar[DType] = int32

# https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html
# we don't support weak type and complex type
promo_lattice = { dtypes.bool: [dtypes.int8, dtypes.uint8], dtypes.int8: [dtypes.int16], dtypes.int16: [dtypes.int32], dtypes.int32: [dtypes.int64],
  dtypes.int64: [dtypes.float16, dtypes.bfloat16], dtypes.uint8: [dtypes.int16, dtypes.uint16], dtypes.uint16: [dtypes.int32, dtypes.uint32],
  dtypes.uint32: [dtypes.int64, dtypes.uint64], dtypes.uint64: [dtypes.float16, dtypes.bfloat16],
  dtypes.float16: [dtypes.float32], dtypes.bfloat16: [dtypes.float32], dtypes.float32: [dtypes.float64], }

@functools.lru_cache(None)
def _get_recursive_parents(dtype:DType) -> Set[DType]:
  return set.union(*[_get_recursive_parents(d) for d in promo_lattice[dtype]], {dtype}) if dtype != dtypes.float64 else {dtypes.float64}
@functools.lru_cache(None)
def least_upper_dtype(*ds:DType) -> DType:
  return min(set.intersection(*[_get_recursive_parents(d) for d in ds])) if not (images:=[d for d in ds if isinstance(d, ImageDType)]) else images[0]
def least_upper_float(dt:DType) -> DType: return dt if dtypes.is_float(dt) else least_upper_dtype(dt, dtypes.float32)

# HACK: staticmethods are not callable in 3.8 so we have to compare the class
DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not (k.startswith(('__', 'default')) or v.__class__ is staticmethod)}
INVERSE_DTYPES_DICT = {v.name:k for k,v in DTYPES_DICT.items()}


# tinygrad/helpers.py

from __future__ import annotations
import os, functools, platform, time, re, contextlib, operator, hashlib, pickle, sqlite3, cProfile, pstats, tempfile, pathlib, string, ctypes
import itertools, urllib.request
from tqdm import tqdm
from typing import Dict, Tuple, Union, List, ClassVar, Optional, Iterable, Any, TypeVar, TYPE_CHECKING, Callable, Sequence
if TYPE_CHECKING:  # TODO: remove this and import TypeGuard from typing once minimum python supported version is 3.10
  from typing_extensions import TypeGuard
  from tinygrad.shape.shapetracker import sint

T = TypeVar("T")
U = TypeVar("U")
# NOTE: it returns int 1 if x is empty regardless of the type of x
def prod(x:Iterable[T]) -> Union[T,int]: return functools.reduce(operator.mul, x, 1)

# NOTE: helpers is not allowed to import from anything else in tinygrad
OSX = platform.system() == "Darwin"
CI = os.getenv("CI", "") != ""

def dedup(x:Iterable[T]): return list(dict.fromkeys(x))   # retains list order
def argfix(*x):
  if x and x[0].__class__ in (tuple, list):
    if len(x) != 1: raise ValueError(f"bad arg {x}")
    return tuple(x[0])
  return x
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def all_same(items:List[T]): return all(x == items[0] for x in items)
def all_int(t: Sequence[Any]) -> TypeGuard[Tuple[int, ...]]: return all(isinstance(s, int) for s in t)
def colored(st, color:Optional[str], background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st  # replace the termcolor library with one line  # noqa: E501
def ansistrip(s:str): return re.sub('\x1b\\[(K|.*?m)', '', s)
def ansilen(s:str): return len(ansistrip(s))
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x
def flatten(l:Iterable[Iterable[T]]): return [item for sublist in l for item in sublist]
def fully_flatten(l): return [item for sublist in l for item in (fully_flatten(sublist) if isinstance(sublist, (tuple, list)) else [sublist])]
def fromimport(mod, frm): return getattr(__import__(mod, fromlist=[frm]), frm)
def strip_parens(fst:str): return fst[1:-1] if fst[0] == '(' and fst[-1] == ')' and fst[1:-1].find('(') <= fst[1:-1].find(')') else fst
def round_up(num, amt:int): return (num+amt-1)//amt * amt
def merge_dicts(ds:Iterable[Dict[T,U]]) -> Dict[T,U]:
  assert len(kvs:=set([(k,v) for d in ds for k,v in d.items()])) == len(set(kv[0] for kv in kvs)), f"cannot merge, {kvs} contains different values for the same key"  # noqa: E501
  return {k:v for d in ds for k,v in d.items()}
def partition(lst:List[T], fxn:Callable[[T],bool]) -> Tuple[List[T], List[T]]:
  a:List[T] = []
  b:List[T] = []
  for s in lst: (a if fxn(s) else b).append(s)
  return a,b
def unwrap(x:Optional[T]) -> T:
  assert x is not None
  return x
def unwrap2(x:Tuple[T,Any]) -> T:
  ret, err = x
  assert err is None, str(err)
  return ret
def get_child(obj, key):
  for k in key.split('.'):
    if k.isnumeric(): obj = obj[int(k)]
    elif isinstance(obj, dict): obj = obj[k]
    else: obj = getattr(obj, k)
  return obj

# returns the axes to create new_shape if new_shape can be created by combining axis from old_shape
def get_contraction(old_shape:Tuple[sint, ...], new_shape:Tuple[sint, ...]) -> Optional[List[List[int]]]:
  acc_old, acc_new = list(itertools.accumulate(old_shape, operator.mul)), list(itertools.accumulate(new_shape, operator.mul))
  try: split = [acc_old.index(acc)+1 if acc != 1 else 0 for acc in acc_new]
  except ValueError: return None
  return [list(range(st,ed)) for st,ed in zip([0]+split[:-1], split[:-1]+[len(old_shape)])]

@functools.lru_cache(maxsize=None)
def to_function_name(s:str): return ''.join([c if c in (string.ascii_letters+string.digits+'_') else f'{ord(c):02X}' for c in ansistrip(s)])
@functools.lru_cache(maxsize=None)
def getenv(key:str, default=0): return type(default)(os.getenv(key, default))
def temp(x:str) -> str: return (pathlib.Path(tempfile.gettempdir()) / x).as_posix()

class GraphException(Exception): pass

class Context(contextlib.ContextDecorator):
  stack: ClassVar[List[dict[str, int]]] = [{}]
  def __init__(self, **kwargs): self.kwargs = kwargs
  def __enter__(self):
    Context.stack[-1] = {k:o.value for k,o in ContextVar._cache.items()} # Store current state.
    for k,v in self.kwargs.items(): ContextVar._cache[k].value = v # Update to new temporary state.
    Context.stack.append(self.kwargs) # Store the temporary state so we know what to undo later.
  def __exit__(self, *args):
    for k in Context.stack.pop(): ContextVar._cache[k].value = Context.stack[-1].get(k, ContextVar._cache[k].value)

class ContextVar:
  _cache: ClassVar[Dict[str, ContextVar]] = {}
  value: int
  def __new__(cls, key, default_value):
    if key in ContextVar._cache: return ContextVar._cache[key]
    instance = ContextVar._cache[key] = super().__new__(cls)
    instance.value = getenv(key, default_value)
    return instance
  def __bool__(self): return bool(self.value)
  def __ge__(self, x): return self.value >= x
  def __gt__(self, x): return self.value > x
  def __lt__(self, x): return self.value < x

DEBUG, IMAGE, WINO, BEAM, NOOPT = ContextVar("DEBUG", 0), ContextVar("IMAGE", 0), ContextVar("WINO", 0), ContextVar("BEAM", 0), ContextVar("NOOPT", 0)
GRAPH, GRAPHPATH = ContextVar("GRAPH", 0), getenv("GRAPHPATH", "/tmp/net")

class Timing(contextlib.ContextDecorator):
  def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
  def __enter__(self): self.st = time.perf_counter_ns()
  def __exit__(self, *exc):
    self.et = time.perf_counter_ns() - self.st
    if self.enabled: print(f"{self.prefix}{self.et*1e-6:6.2f} ms"+(self.on_exit(self.et) if self.on_exit else ""))

def _format_fcn(fcn): return f"{fcn[0]}:{fcn[1]}:{fcn[2]}"
class Profiling(contextlib.ContextDecorator):
  def __init__(self, enabled=True, sort='cumtime', frac=0.2, fn=None, ts=1):
    self.enabled, self.sort, self.frac, self.fn, self.time_scale = enabled, sort, frac, fn, 1e3/ts
  def __enter__(self):
    self.pr = cProfile.Profile()
    if self.enabled: self.pr.enable()
  def __exit__(self, *exc):
    if self.enabled:
      self.pr.disable()
      if self.fn: self.pr.dump_stats(self.fn)
      stats = pstats.Stats(self.pr).strip_dirs().sort_stats(self.sort)
      for fcn in stats.fcn_list[0:int(len(stats.fcn_list)*self.frac)]:    # type: ignore[attr-defined]
        (_primitive_calls, num_calls, tottime, cumtime, callers) = stats.stats[fcn]    # type: ignore[attr-defined]
        scallers = sorted(callers.items(), key=lambda x: -x[1][2])
        print(f"n:{num_calls:8d}  tm:{tottime*self.time_scale:7.2f}ms  tot:{cumtime*self.time_scale:7.2f}ms",
              colored(_format_fcn(fcn), "yellow") + " "*(50-len(_format_fcn(fcn))),
              colored(f"<- {(scallers[0][1][2]/tottime)*100:3.0f}% {_format_fcn(scallers[0][0])}", "BLACK") if len(scallers) else '')

# *** universal database cache ***

_cache_dir: str = getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if OSX else "~/.cache"))
CACHEDB: str = getenv("CACHEDB", os.path.abspath(os.path.join(_cache_dir, "tinygrad", "cache.db")))
CACHELEVEL = getenv("CACHELEVEL", 2)

VERSION = 12
_db_connection = None
def db_connection():
  global _db_connection
  if _db_connection is None:
    os.makedirs(CACHEDB.rsplit(os.sep, 1)[0], exist_ok=True)
    _db_connection = sqlite3.connect(CACHEDB)
    if DEBUG >= 7: _db_connection.set_trace_callback(print)
  return _db_connection

def diskcache_get(table:str, key:Union[Dict, str, int]) -> Any:
  if CACHELEVEL == 0: return None
  if isinstance(key, (str,int)): key = {"key": key}
  conn = db_connection()
  cur = conn.cursor()
  try:
    res = cur.execute(f"SELECT val FROM '{table}_{VERSION}' WHERE {' AND '.join([f'{x}=?' for x in key.keys()])}", tuple(key.values()))
  except sqlite3.OperationalError:
    return None  # table doesn't exist
  if (val:=res.fetchone()) is not None: return pickle.loads(val[0])
  return None

_db_tables = set()
def diskcache_put(table:str, key:Union[Dict, str, int], val:Any):
  if CACHELEVEL == 0: return val
  if isinstance(key, (str,int)): key = {"key": key}
  conn = db_connection()
  cur = conn.cursor()
  if table not in _db_tables:
    TYPES = {str: "text", bool: "integer", int: "integer", float: "numeric", bytes: "blob"}
    ltypes = ', '.join(f"{k} {TYPES[type(key[k])]}" for k in key.keys())
    cur.execute(f"CREATE TABLE IF NOT EXISTS '{table}_{VERSION}' ({ltypes}, val blob, PRIMARY KEY ({', '.join(key.keys())}))")
    _db_tables.add(table)
  cur.execute(f"REPLACE INTO '{table}_{VERSION}' ({', '.join(key.keys())}, val) VALUES ({', '.join(['?']*len(key.keys()))}, ?)", tuple(key.values()) + (pickle.dumps(val), ))  # noqa: E501
  conn.commit()
  cur.close()
  return val

def diskcache(func):
  def wrapper(*args, **kwargs) -> bytes:
    table, key = f"cache_{func.__name__}", hashlib.sha256(pickle.dumps((args, kwargs))).hexdigest()
    if (ret:=diskcache_get(table, key)): return ret
    return diskcache_put(table, key, func(*args, **kwargs))
  return wrapper

# *** http support ***

def fetch(url:str, name:Optional[Union[pathlib.Path, str]]=None, allow_caching=not getenv("DISABLE_HTTP_CACHE")) -> pathlib.Path:
  if url.startswith(("/", ".")): return pathlib.Path(url)
  fp = pathlib.Path(name) if name is not None and (isinstance(name, pathlib.Path) or '/' in name) else pathlib.Path(_cache_dir) / "tinygrad" / "downloads" / (name if name else hashlib.md5(url.encode('utf-8')).hexdigest())  # noqa: E501
  if not fp.is_file() or not allow_caching:
    with urllib.request.urlopen(url, timeout=10) as r:
      assert r.status == 200
      total_length = int(r.headers.get('content-length', 0))
      progress_bar = tqdm(total=total_length, unit='B', unit_scale=True, desc=url)
      (path := fp.parent).mkdir(parents=True, exist_ok=True)
      with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
        while chunk := r.read(16384): progress_bar.update(f.write(chunk))
        f.close()
        if (file_size:=os.stat(f.name).st_size) < total_length: raise RuntimeError(f"fetch size incomplete, {file_size} < {total_length}")
        pathlib.Path(f.name).rename(fp)
  return fp

# *** Exec helpers

def cpu_time_execution(cb, enable):
  if enable: st = time.perf_counter()
  cb()
  if enable: return time.perf_counter()-st

# *** ctypes helpers

# TODO: make this work with read only memoryviews (if possible)
def from_mv(mv:memoryview, to_type=ctypes.c_char): return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type))
def to_mv(ptr, sz) -> memoryview: return memoryview(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * sz)).contents).cast("B")
def to_char_p_p(options: List[bytes], to_type=ctypes.c_char): return (ctypes.POINTER(to_type) * len(options))(*[ctypes.cast(ctypes.create_string_buffer(o), ctypes.POINTER(to_type)) for o in options])  # noqa: E501
@functools.lru_cache(maxsize=None)
def init_c_struct_t(fields: Tuple[Tuple[str, ctypes._SimpleCData], ...]):
  class CStruct(ctypes.Structure):
    _pack_, _fields_ = 1, fields
  return CStruct
def init_c_var(ctypes_var, creat_cb): return (creat_cb(ctypes_var), ctypes_var)[1]
def flat_mv(mv:memoryview): return mv if len(mv) == 0 else mv.cast("B", shape=(mv.nbytes,))

# *** Helpers for CUDA-like APIs.

def encode_args_cuda_style(bufs, vals, device_ptr_t, marks) -> Tuple[ctypes.Array, ctypes.Structure]:
  c_args = init_c_struct_t(tuple([(f'f{i}', device_ptr_t) for i in range(len(bufs))] + [(f'f{i}', ctypes.c_int) for i in range(len(bufs), len(bufs)+len(vals))]))(*bufs, *vals)  # noqa: E501
  return (ctypes.c_void_p * 5)(ctypes.c_void_p(marks[0]), ctypes.cast(ctypes.pointer(c_args), ctypes.c_void_p), ctypes.c_void_p(marks[1]), ctypes.cast(ctypes.pointer(ctypes.c_size_t(ctypes.sizeof(c_args))), ctypes.c_void_p), ctypes.c_void_p(marks[2])), c_args  # noqa: E501

def time_execution_cuda_style(cb, ev_t, evcreate, evrecord, evsync, evdestroy, evtime, enable=False) -> Optional[float]:
  if not enable: return cb()
  evs = [init_c_var(ev_t(), lambda x: evcreate(ctypes.byref(x), 0)) for _ in range(2)]
  evrecord(evs[0], None)
  cb()
  evrecord(evs[1], None)
  evsync(evs[1])
  evtime(ctypes.byref(ret := ctypes.c_float()), evs[0], evs[1])
  for ev in evs: evdestroy(ev)
  return ret.value * 1e-3


# tinygrad/helpers.py

from __future__ import annotations
import math
from typing import Union, Optional, Any, Tuple, List, Dict, cast
from tinygrad.dtype import cast_scalar, dtypes, DType, Scalar
from tinygrad.helpers import prod, getenv, all_int, all_same
from tinygrad.ops import LoadOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps, Op
from tinygrad.shape.symbolic import sint
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer
from weakref import ref, ReferenceType

lazycache: Dict[Any, ReferenceType[LazyBuffer]] = {}
def create_lazybuffer(device:str, st:ShapeTracker, dtype:DType, op:Optional[Op]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(),
                      base:Optional[LazyBuffer]=None, enable_cache=bool(getenv("LAZYCACHE", 1))):
  if st.size == 0 and op not in {LoadOps.SYNC, LoadOps.WAIT}: op, arg, srcs, base = LoadOps.CONST, 0, (), None
  if op == LoadOps.CONST: enable_cache = True

  cache_key = (device, st, dtype, op, arg, tuple(ref(x) for x in srcs)) if base is None else (st, ref(base))
  if (rret := lazycache.get(cache_key, None)): return cast(LazyBuffer, rret())  # NOTE: this should always be a live reference

  return LazyBuffer(device, st, dtype, op, arg, srcs, base=base, cache_key=cache_key if enable_cache else None)

class LazyBuffer:
  def __init__(self, device:str, st:ShapeTracker, dtype:DType,
               op:Optional[Op]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(),
               base:Optional[LazyBuffer]=None, cache_key=None):
    self.device, self.st, self.dtype, self.shape, self.size, self.cache_key = device, st, dtype, st.shape, st.size, cache_key
    self._base: Optional[LazyBuffer] = None
    if base is None:
      # properties on base
      self.op, self.arg, self.srcs = op, arg, srcs  # this is a LazyOp, except the src is LazyBuffers and not LazyOps
      self.realized: Optional[Buffer] = None
      self.output_buffer: Optional[Buffer] = None
      self.contiguous_child: Optional[Tuple[ReferenceType[LazyBuffer], ShapeTracker]] = None
      self.forced_realize = False
    else:
      # properties on view
      assert base.base == base, "base must be a base itself"
      self._base = base
    if cache_key is not None: lazycache[cache_key] = ref(self)

  def __del__(self): lazycache.pop(self.cache_key, None)

  def __repr__(self) -> str:
    return f"<LB {self.device} {self.shape} contig:{self.st.contiguous} {self.st if self.base != self else (self.op, self.realized)}>"

  # NOTE: this has to be a function to prevent self reference
  @property
  def base(self) -> LazyBuffer: return self._base if self._base is not None else self

  @staticmethod
  def loadop(op, shape:Tuple[sint,...], dtype:DType, device:str, arg=None, src:Optional[LazyBuffer]=None, enable_cache=False) -> LazyBuffer:
    return create_lazybuffer(device, ShapeTracker.from_shape(shape), dtype, op, arg, (src,) if src is not None else (), enable_cache=enable_cache)

  def const(self, val:Scalar, shape:Optional[Tuple[sint,...]]=None) -> LazyBuffer:
    shape = self.shape if shape is None else shape
    return LazyBuffer.loadop(LoadOps.CONST, tuple(), self.dtype, self.device, arg=cast_scalar(val, self.dtype)).reshape((1,)*len(shape)).expand(shape)

  def contiguous(self):
    if not self.st.contiguous or self.size != self.base.size or self.is_unrealized_const():
      ret = self.e(LoadOps.CONTIGUOUS)
      if (sti := self.st.invert(self.base.shape)) is not None: self.base.contiguous_child = ref(ret), sti
      return ret
    self.base.forced_realize = True
    return self

  def cast(self, dtype:DType, bitcast:bool=False):
    if self.dtype == dtype: return self
    # TODO: applying this makes gpt2 slower
    if getenv("CAST_BEFORE_VIEW", 1) and dtype.itemsize <= self.dtype.itemsize and self != self.base:
      return self.base.cast(dtype, bitcast)._view(self.st)
    return create_lazybuffer(self.device, ShapeTracker.from_shape(self.shape), dtype, UnaryOps.CAST, (dtype, bitcast), (self,))

  def is_unrealized_const(self): return not self.base.realized and self.base.op is LoadOps.CONST
  def is_unrealized_contiguous_const(self): return self.base == self and not self.base.realized and self.op is LoadOps.CONST

  def _copy(self, device:str) -> LazyBuffer:
    sync_size = 1 if self.device.startswith("HIP") else 0
    if self.device.startswith("EXT") or self.device.startswith("DISK"):
      # DISK/EXT don't sync
      return create_lazybuffer(device, ShapeTracker.from_shape(self.shape), self.dtype, LoadOps.COPY, None, (self,), enable_cache=False)
    sync = LazyBuffer.loadop(LoadOps.SYNC, (sync_size,), dtypes.uint32, self.device, src=self, enable_cache=True)
    wait = LazyBuffer.loadop(LoadOps.WAIT, (0,), dtypes.uint32, device, src=sync, enable_cache=True)
    return create_lazybuffer(device, ShapeTracker.from_shape(self.shape), self.dtype, LoadOps.COPY, None, (self, wait), enable_cache=False)

  def copy_to_device(self, device:str) -> LazyBuffer:
    # no COPY
    if self.device == device: return self

    # double COPY = one COPY
    if self.st.contiguous and self.size == self.base.size and not self.base.realized and self.base.op is LoadOps.COPY:
      return self.base.srcs[0].copy_to_device(device).reshape(self.st.shape)

    # const doesn't have to be copied (issues with disk tensor)
    if self.is_unrealized_const():
      return LazyBuffer.loadop(LoadOps.CONST, tuple(), self.dtype, device, arg=self.base.arg)._view(self.st)

    # if it's a shrink, do the shrink before the copy with CONTIGUOUS
    if prod(self.st.shape) < prod(self.base.st.shape): return self.contiguous()._copy(device)

    # copy the base and apply the shapetracker on the new device
    return self.base._copy(device)._view(self.st)

  def e(self, op:Union[LoadOps, UnaryOps, BinaryOps, TernaryOps], *in_srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
    srcs: List[LazyBuffer] = []
    for s in (self,)+in_srcs:
      if s == s.base and s.base.contiguous_child and (root:=s.base.contiguous_child[0]()) is not None:
        srcs.append(root._view(s.base.contiguous_child[1]))
      else:
        srcs.append(s)
    assert all_same(dts:=[x.dtype.scalar() for x in (srcs[1:] if op is TernaryOps.WHERE else srcs)]), f"all dtypes must match {dts} on {op}"
    assert all_same([x.shape for x in srcs]), f"all shapes must be the same {[x.shape for x in srcs]}"
    if op is TernaryOps.WHERE: assert srcs[0].dtype == dtypes.bool, "TernaryOps.WHERE must have the first arg be bool"
    if op is UnaryOps.NEG: assert srcs[0].dtype != dtypes.bool, "UnaryOps.NEG does not accept dtype bool"
    out_dtype = dtypes.bool if op in (BinaryOps.CMPLT, BinaryOps.CMPEQ) else srcs[-1].dtype
    return create_lazybuffer(self.device, ShapeTracker.from_shape(self.shape), out_dtype, op, arg, tuple(srcs))

  # *** reduce ops ***

  def _reduce_op(self, op:ReduceOps, axis:Tuple[int, ...]) -> LazyBuffer:
    assert all(0 <= x < len(self.shape) for x in axis), f"axis args {axis} out of range for shape {self.shape}"
    axis = tuple(x for x in axis if self.shape[x] != 1)
    if len(axis) == 0: return self
    new_shape = tuple(1 if i in axis else s for i,s in enumerate(self.shape))
    return create_lazybuffer(self.device, ShapeTracker.from_shape(new_shape), self.dtype, op, axis, (self,))

  def r(self, op:ReduceOps, axis:Tuple[int, ...]) -> LazyBuffer:
    new_shape = tuple(1 if i in axis else s for i,s in enumerate(self.shape))
    # TODO: this logic should move to the scheduler
    if self.size == 0 and 0 not in new_shape: return self.const({ReduceOps.SUM: 0.0, ReduceOps.MAX: -math.inf}[op], new_shape)
    # TODO: can we split symbolic shape if the reduce axis is not symbolic?
    if not all_int(self.shape) or (0 in self.shape) or prod(self.shape) // prod(new_shape) < getenv("REDUCEOP_SPLIT_THRESHOLD", 32768):
      return self._reduce_op(op, axis)
    heuristic, divisor, dim_to_split = max(((divisor := math.gcd(256, s))/(st or math.inf), divisor, i) for i,(s,st) in \
                                           enumerate(zip(self.shape, self.st.real_strides())) if i in axis and (st is None or isinstance(st, int)))
    if divisor < 16 or heuristic < 0.1: return self._reduce_op(op, axis)
    # choose largest divisor (>=16) to split on, penalize large strides
    def splitted_shape(dim_aft_div):
      return self.shape[:dim_to_split] + (self.shape[dim_to_split]//divisor,) + dim_aft_div + self.shape[dim_to_split+1:]
    return self.reshape(splitted_shape((divisor,)))._reduce_op(op, (dim_to_split+1,)).reshape(splitted_shape(()))._reduce_op(op, axis)

  # *** movement ops ***

  def _view(self, new_st:ShapeTracker) -> LazyBuffer:
    if self.st.size == 0 or (new_st.views[-1].mask is not None and all((x[1]-x[0]) == 0 for x in new_st.views[-1].mask)):
      return self.const(0, new_st.shape)
    if new_st.contiguous and self.base.shape == new_st.shape: return self.base
    return create_lazybuffer(self.device, new_st, self.dtype, base=self.base)

  def reshape(self, arg:Tuple[sint, ...]): return self._view(self.st.reshape(arg))
  def pad(self, arg:Tuple[Tuple[sint, sint], ...]): return self._view(self.st.pad(arg))
  def expand(self, arg:Tuple[sint, ...]): return self._view(self.st.expand(arg))
  def permute(self, arg:Tuple[int, ...]): return self._view(self.st.permute(arg))
  def shrink(self, arg:Tuple[Tuple[sint, sint], ...]): return self._view(self.st.shrink(arg))
  def stride(self, arg:Tuple[int, ...]): return self._view(self.st.stride(arg))


# tinygrad/lazy.py
  
from __future__ import annotations
import math
from typing import Union, Optional, Any, Tuple, List, Dict, cast
from tinygrad.dtype import cast_scalar, dtypes, DType, Scalar
from tinygrad.helpers import prod, getenv, all_int, all_same
from tinygrad.ops import LoadOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps, Op
from tinygrad.shape.symbolic import sint
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.device import Buffer
from weakref import ref, ReferenceType

lazycache: Dict[Any, ReferenceType[LazyBuffer]] = {}
def create_lazybuffer(device:str, st:ShapeTracker, dtype:DType, op:Optional[Op]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(),
                      base:Optional[LazyBuffer]=None, enable_cache=bool(getenv("LAZYCACHE", 1))):
  if st.size == 0 and op not in {LoadOps.SYNC, LoadOps.WAIT}: op, arg, srcs, base = LoadOps.CONST, 0, (), None
  if op == LoadOps.CONST: enable_cache = True

  cache_key = (device, st, dtype, op, arg, tuple(ref(x) for x in srcs)) if base is None else (st, ref(base))
  if (rret := lazycache.get(cache_key, None)): return cast(LazyBuffer, rret())  # NOTE: this should always be a live reference

  return LazyBuffer(device, st, dtype, op, arg, srcs, base=base, cache_key=cache_key if enable_cache else None)

class LazyBuffer:
  def __init__(self, device:str, st:ShapeTracker, dtype:DType,
               op:Optional[Op]=None, arg:Any=None, srcs:Tuple[LazyBuffer, ...]=(),
               base:Optional[LazyBuffer]=None, cache_key=None):
    self.device, self.st, self.dtype, self.shape, self.size, self.cache_key = device, st, dtype, st.shape, st.size, cache_key
    self._base: Optional[LazyBuffer] = None
    if base is None:
      # properties on base
      self.op, self.arg, self.srcs = op, arg, srcs  # this is a LazyOp, except the src is LazyBuffers and not LazyOps
      self.realized: Optional[Buffer] = None
      self.output_buffer: Optional[Buffer] = None
      self.contiguous_child: Optional[Tuple[ReferenceType[LazyBuffer], ShapeTracker]] = None
      self.forced_realize = False
    else:
      # properties on view
      assert base.base == base, "base must be a base itself"
      self._base = base
    if cache_key is not None: lazycache[cache_key] = ref(self)

  def __del__(self): lazycache.pop(self.cache_key, None)

  def __repr__(self) -> str:
    return f"<LB {self.device} {self.shape} contig:{self.st.contiguous} {self.st if self.base != self else (self.op, self.realized)}>"

  # NOTE: this has to be a function to prevent self reference
  @property
  def base(self) -> LazyBuffer: return self._base if self._base is not None else self

  @staticmethod
  def loadop(op, shape:Tuple[sint,...], dtype:DType, device:str, arg=None, src:Optional[LazyBuffer]=None, enable_cache=False) -> LazyBuffer:
    return create_lazybuffer(device, ShapeTracker.from_shape(shape), dtype, op, arg, (src,) if src is not None else (), enable_cache=enable_cache)

  def const(self, val:Scalar, shape:Optional[Tuple[sint,...]]=None) -> LazyBuffer:
    shape = self.shape if shape is None else shape
    return LazyBuffer.loadop(LoadOps.CONST, tuple(), self.dtype, self.device, arg=cast_scalar(val, self.dtype)).reshape((1,)*len(shape)).expand(shape)

  def contiguous(self):
    if not self.st.contiguous or self.size != self.base.size or self.is_unrealized_const():
      ret = self.e(LoadOps.CONTIGUOUS)
      if (sti := self.st.invert(self.base.shape)) is not None: self.base.contiguous_child = ref(ret), sti
      return ret
    self.base.forced_realize = True
    return self

  def cast(self, dtype:DType, bitcast:bool=False):
    if self.dtype == dtype: return self
    # TODO: applying this makes gpt2 slower
    if getenv("CAST_BEFORE_VIEW", 1) and dtype.itemsize <= self.dtype.itemsize and self != self.base:
      return self.base.cast(dtype, bitcast)._view(self.st)
    return create_lazybuffer(self.device, ShapeTracker.from_shape(self.shape), dtype, UnaryOps.CAST, (dtype, bitcast), (self,))

  def is_unrealized_const(self): return not self.base.realized and self.base.op is LoadOps.CONST
  def is_unrealized_contiguous_const(self): return self.base == self and not self.base.realized and self.op is LoadOps.CONST

  def _copy(self, device:str) -> LazyBuffer:
    sync_size = 1 if self.device.startswith("HIP") else 0
    if self.device.startswith("EXT") or self.device.startswith("DISK"):
      # DISK/EXT don't sync
      return create_lazybuffer(device, ShapeTracker.from_shape(self.shape), self.dtype, LoadOps.COPY, None, (self,), enable_cache=False)
    sync = LazyBuffer.loadop(LoadOps.SYNC, (sync_size,), dtypes.uint32, self.device, src=self, enable_cache=True)
    wait = LazyBuffer.loadop(LoadOps.WAIT, (0,), dtypes.uint32, device, src=sync, enable_cache=True)
    return create_lazybuffer(device, ShapeTracker.from_shape(self.shape), self.dtype, LoadOps.COPY, None, (self, wait), enable_cache=False)

  def copy_to_device(self, device:str) -> LazyBuffer:
    # no COPY
    if self.device == device: return self

    # double COPY = one COPY
    if self.st.contiguous and self.size == self.base.size and not self.base.realized and self.base.op is LoadOps.COPY:
      return self.base.srcs[0].copy_to_device(device).reshape(self.st.shape)

    # const doesn't have to be copied (issues with disk tensor)
    if self.is_unrealized_const():
      return LazyBuffer.loadop(LoadOps.CONST, tuple(), self.dtype, device, arg=self.base.arg)._view(self.st)

    # if it's a shrink, do the shrink before the copy with CONTIGUOUS
    if prod(self.st.shape) < prod(self.base.st.shape): return self.contiguous()._copy(device)

    # copy the base and apply the shapetracker on the new device
    return self.base._copy(device)._view(self.st)

  def e(self, op:Union[LoadOps, UnaryOps, BinaryOps, TernaryOps], *in_srcs:LazyBuffer, arg:Optional[Any]=None) -> LazyBuffer:
    srcs: List[LazyBuffer] = []
    for s in (self,)+in_srcs:
      if s == s.base and s.base.contiguous_child and (root:=s.base.contiguous_child[0]()) is not None:
        srcs.append(root._view(s.base.contiguous_child[1]))
      else:
        srcs.append(s)
    assert all_same(dts:=[x.dtype.scalar() for x in (srcs[1:] if op is TernaryOps.WHERE else srcs)]), f"all dtypes must match {dts} on {op}"
    assert all_same([x.shape for x in srcs]), f"all shapes must be the same {[x.shape for x in srcs]}"
    if op is TernaryOps.WHERE: assert srcs[0].dtype == dtypes.bool, "TernaryOps.WHERE must have the first arg be bool"
    if op is UnaryOps.NEG: assert srcs[0].dtype != dtypes.bool, "UnaryOps.NEG does not accept dtype bool"
    out_dtype = dtypes.bool if op in (BinaryOps.CMPLT, BinaryOps.CMPEQ) else srcs[-1].dtype
    return create_lazybuffer(self.device, ShapeTracker.from_shape(self.shape), out_dtype, op, arg, tuple(srcs))

  # *** reduce ops ***

  def _reduce_op(self, op:ReduceOps, axis:Tuple[int, ...]) -> LazyBuffer:
    assert all(0 <= x < len(self.shape) for x in axis), f"axis args {axis} out of range for shape {self.shape}"
    axis = tuple(x for x in axis if self.shape[x] != 1)
    if len(axis) == 0: return self
    new_shape = tuple(1 if i in axis else s for i,s in enumerate(self.shape))
    return create_lazybuffer(self.device, ShapeTracker.from_shape(new_shape), self.dtype, op, axis, (self,))

  def r(self, op:ReduceOps, axis:Tuple[int, ...]) -> LazyBuffer:
    new_shape = tuple(1 if i in axis else s for i,s in enumerate(self.shape))
    # TODO: this logic should move to the scheduler
    if self.size == 0 and 0 not in new_shape: return self.const({ReduceOps.SUM: 0.0, ReduceOps.MAX: -math.inf}[op], new_shape)
    # TODO: can we split symbolic shape if the reduce axis is not symbolic?
    if not all_int(self.shape) or (0 in self.shape) or prod(self.shape) // prod(new_shape) < getenv("REDUCEOP_SPLIT_THRESHOLD", 32768):
      return self._reduce_op(op, axis)
    heuristic, divisor, dim_to_split = max(((divisor := math.gcd(256, s))/(st or math.inf), divisor, i) for i,(s,st) in \
                                           enumerate(zip(self.shape, self.st.real_strides())) if i in axis and (st is None or isinstance(st, int)))
    if divisor < 16 or heuristic < 0.1: return self._reduce_op(op, axis)
    # choose largest divisor (>=16) to split on, penalize large strides
    def splitted_shape(dim_aft_div):
      return self.shape[:dim_to_split] + (self.shape[dim_to_split]//divisor,) + dim_aft_div + self.shape[dim_to_split+1:]
    return self.reshape(splitted_shape((divisor,)))._reduce_op(op, (dim_to_split+1,)).reshape(splitted_shape(()))._reduce_op(op, axis)

  # *** movement ops ***

  def _view(self, new_st:ShapeTracker) -> LazyBuffer:
    if self.st.size == 0 or (new_st.views[-1].mask is not None and all((x[1]-x[0]) == 0 for x in new_st.views[-1].mask)):
      return self.const(0, new_st.shape)
    if new_st.contiguous and self.base.shape == new_st.shape: return self.base
    return create_lazybuffer(self.device, new_st, self.dtype, base=self.base)

  def reshape(self, arg:Tuple[sint, ...]): return self._view(self.st.reshape(arg))
  def pad(self, arg:Tuple[Tuple[sint, sint], ...]): return self._view(self.st.pad(arg))
  def expand(self, arg:Tuple[sint, ...]): return self._view(self.st.expand(arg))
  def permute(self, arg:Tuple[int, ...]): return self._view(self.st.permute(arg))
  def shrink(self, arg:Tuple[Tuple[sint, sint], ...]): return self._view(self.st.shrink(arg))
  def stride(self, arg:Tuple[int, ...]): return self._view(self.st.stride(arg))


# tinygrad/mlops.py
  
import math
from typing import Tuple, Optional
from tinygrad.helpers import argsort
from tinygrad.dtype import DType
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps
from tinygrad.tensor import Function
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.symbolic import sint

class Contiguous(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer: return x.contiguous()
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output

class ContiguousBackward(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer: return x
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.contiguous()

class Cast(Function):
  def forward(self, x:LazyBuffer, dtype:DType, bitcast:bool=False) -> LazyBuffer:
    self.input_dtype, self.bitcast = x.dtype, bitcast
    return x.cast(dtype, bitcast)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.cast(self.input_dtype, self.bitcast)

# ************* unary ops *************

class Zero(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer: return x.const(0)
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.const(0)

class Neg(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer: return x.e(UnaryOps.NEG)
  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.e(UnaryOps.NEG)

class Sin(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.x = x
    return x.e(UnaryOps.SIN)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return self.x.const(math.pi / 2).e(BinaryOps.SUB, self.x).e(UnaryOps.SIN).e(BinaryOps.MUL, grad_output)

# NOTE: maximum(x, 0) behaves differently where x=0
class Relu(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.e(BinaryOps.MAX, x.const(0))
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return self.ret.const(0).e(BinaryOps.CMPLT, self.ret).cast(grad_output.dtype).e(BinaryOps.MUL, grad_output)

class Log(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.x = x
    return x.e(UnaryOps.LOG2).e(BinaryOps.MUL, x.const(math.log(2)))

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.e(BinaryOps.DIV, self.x)

class Exp(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.e(BinaryOps.MUL, x.const(1/math.log(2))).e(UnaryOps.EXP2)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return self.ret.e(BinaryOps.MUL, grad_output)

class Sqrt(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.e(UnaryOps.SQRT)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.e(BinaryOps.DIV, self.ret.e(BinaryOps.MUL, self.ret.const(2)))

# NOTE: the implicit derivative of sigmoid is not stable
# https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
# TODO: have the backend automatically find this
class Sigmoid(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.const(1).e(BinaryOps.DIV, x.const(1).e(BinaryOps.ADD, x.e(BinaryOps.MUL, x.const(-1/math.log(2))).e(UnaryOps.EXP2)))
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return self.ret.e(BinaryOps.MUL, self.ret.const(1).e(BinaryOps.SUB, self.ret)).e(BinaryOps.MUL, grad_output)

# ************* binary ops *************

class Less(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.e(BinaryOps.CMPLT, y)
  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]: return None, None

class Eq(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.e(BinaryOps.CMPEQ, y)
  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]: return None, None

class Xor(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.e(BinaryOps.XOR, y)

class Add(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.e(BinaryOps.ADD, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output if self.needs_input_grad[1] else None

class Sub(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.e(BinaryOps.SUB, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output.e(UnaryOps.NEG) if self.needs_input_grad[1] else None

class Mul(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    self.x, self.y = x, y
    return x.e(BinaryOps.MUL, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return self.y.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[0] else None, \
           self.x.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[1] else None

class Div(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    self.x, self.y = x, y
    return x.e(BinaryOps.DIV, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return grad_output.e(BinaryOps.DIV, self.y) if self.needs_input_grad[0] else None, \
           grad_output.e(UnaryOps.NEG).e(BinaryOps.MUL, self.x).e(BinaryOps.DIV, self.y.e(BinaryOps.MUL, self.y)) if self.needs_input_grad[1] else None  # noqa: E501

# ************* ternary ops *************

class Where(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer, z:LazyBuffer) -> LazyBuffer:
    self.x = x
    return self.x.e(TernaryOps.WHERE, y, z)

  def backward(self, grad_output:LazyBuffer) -> Tuple[None, Optional[LazyBuffer], Optional[LazyBuffer]]:
    return None, \
      self.x.e(TernaryOps.WHERE, grad_output, grad_output.const(0)) if self.needs_input_grad[1] else None, \
      self.x.e(TernaryOps.WHERE, grad_output.const(0), grad_output) if self.needs_input_grad[2] else None

# ************* reduce ops *************

class Sum(Function):
  def forward(self, x:LazyBuffer, axis:Tuple[int, ...]) -> LazyBuffer:
    self.input_shape = x.shape
    return x.r(ReduceOps.SUM, axis)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.expand(self.input_shape)

class Max(Function):
  def forward(self, x:LazyBuffer, axis:Tuple[int, ...]) -> LazyBuffer:
    self.x, self.ret, self.axis = x, x.r(ReduceOps.MAX, axis), axis
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    # 1s in locations where the max was chosen (can be two locations)
    max_is_1s = self.x.e(BinaryOps.CMPEQ, self.ret.expand(self.x.shape)).cast(self.x.dtype)
    div = max_is_1s.r(ReduceOps.SUM, self.axis).expand(self.x.shape)
    return max_is_1s.e(BinaryOps.DIV, div).e(BinaryOps.MUL, grad_output.expand(self.x.shape))

# ************* movement ops *************

# NOTE: this is sum in reverse
class Expand(Function):
  def forward(self, x:LazyBuffer, shape:Tuple[int, ...]) -> LazyBuffer:
    self.expanded_axis = tuple(i for i, (si, so) in enumerate(zip(x.shape, shape)) if si != so)
    return x.expand(shape)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.r(ReduceOps.SUM, self.expanded_axis)

class Reshape(Function):
  def forward(self, x:LazyBuffer, shape:Tuple[int, ...]) -> LazyBuffer:
    self.input_shape = x.shape
    return x.reshape(shape)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.reshape(self.input_shape)

class Permute(Function):
  def forward(self, x:LazyBuffer, order:Tuple[int, ...]) -> LazyBuffer:
    self.input_order = order
    return x.permute(order)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.permute(argsort(self.input_order))

class Pad(Function):
  def forward(self, x:LazyBuffer, arg:Tuple[Tuple[int, int], ...]) -> LazyBuffer:
    self.narg = tuple([(p[0], s+p[0]) for s,p in zip(x.shape, arg)])
    return x.pad(arg)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.shrink(self.narg)

class Shrink(Function):
  def forward(self, x:LazyBuffer, arg:Tuple[Tuple[sint, sint], ...]) -> LazyBuffer:
    self.narg = tuple([(p[0], s-p[1]) for s,p in zip(x.shape, arg)])
    return x.shrink(arg)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.pad(self.narg)

class Flip(Function):
  def forward(self, x:LazyBuffer, axis:Tuple[int, ...]) -> LazyBuffer:
    self.arg = tuple([-1 if i in set(axis) else 1 for i in range(len(x.shape))])
    return x.stride(self.arg)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer: return grad_output.stride(self.arg)


# tinygrad/ops.py
  
from __future__ import annotations
from typing import TYPE_CHECKING, Union, Type, Tuple, Any, List, Dict, Callable, ClassVar
import functools, hashlib
from enum import Enum, auto
from tinygrad.helpers import prod, dedup
from tinygrad.dtype import dtypes, DType
from tinygrad.shape.symbolic import Variable
from dataclasses import dataclass

# these are the llops your accelerator must implement, along with toCpu
# the Enum class doesn't work with mypy, this is static. sorry it's ugly
# NOTE: MOD, CMPLT don't have to be implemented on vectors, just scalars
# NOTE: many GPUs don't have DIV, but UnaryOps.RECIP doesn't work for integer division
class UnaryOps(Enum): EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto(); SQRT = auto(); NEG = auto() # noqa: E702
class BinaryOps(Enum):
  ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto(); CMPEQ = auto(); XOR = auto() # noqa: E702
class TernaryOps(Enum): WHERE = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class BufferOps(Enum): LOAD = auto(); CONST = auto(); STORE = auto() # noqa: E702
class LoadOps(Enum): EMPTY = auto(); CONST = auto(); COPY = auto(); CONTIGUOUS = auto(); CUSTOM = auto(); SYNC = auto(); WAIT = auto() # noqa: E702

Op = Union[UnaryOps, BinaryOps, ReduceOps, LoadOps, TernaryOps, BufferOps]
OpType = Union[Type[UnaryOps], Type[BinaryOps], Type[ReduceOps], Type[LoadOps], Type[TernaryOps], Type[BufferOps]]

if TYPE_CHECKING:
  from tinygrad.shape.shapetracker import ShapeTracker
  from tinygrad.lazy import LazyBuffer

@dataclass(frozen=True)
class MemBuffer:
  idx: int
  dtype: DType
  st: ShapeTracker

@dataclass(frozen=True)
class ConstBuffer:
  val: Union[int, float]
  dtype: DType
  st: ShapeTracker

@dataclass(frozen=True)
class ScheduleItem:
  ast: LazyOp
  out: LazyBuffer
  inputs: Tuple[LazyBuffer, ...]
  var_vals: Dict[Variable, int]

@dataclass(frozen=True, eq=False)
class LazyOp:
  op: Op
  src: Tuple[LazyOp, ...] = ()
  arg: Any = None
  def cached_compare(self, x, context):
    if id(self) == id(x): return True
    if self.op != x.op or self.arg != x.arg or len(self.src) != len(x.src): return False
    if (key := (id(self), id(x))) in context: return context[key]
    ret = context[key] = all(a.cached_compare(b, context) for a,b in zip(self.src, x.src))
    return ret
  def __eq__(self, x): return self.cached_compare(x, context={})
  def __repr__(self): return f"LazyOp(op={self.op}, src={self.src}, arg={self.arg})"
  @functools.cached_property
  def key(self) -> bytes:
    return hashlib.sha256(functools.reduce(lambda x,y: x+y, [s.key for s in self.src], str((self.op, self.arg)).encode())).digest()
  @functools.cached_property
  def hash(self): return hash((self.op, self.src, self.arg))
  def __hash__(self): return self.hash
  @functools.cached_property
  def lazyops(self) -> List[LazyOp]: return dedup([self] + [item for x in self.src for item in x.lazyops])
  def vars(self) -> List[Variable]:
    return sorted(set.union(*[x.arg.st.vars() for x in self.lazyops if x.op in BufferOps], set()), key=lambda x: str(x.expr))

# **************** independent FlopCounter ****************

@dataclass
class FlopCounter:
  shape: Tuple[int, ...]
  dtype: DType
  flops: int
  mem: Dict[int, int]
  @property
  def mem_estimate(self): return sum(self.mem.values())
  def consume_flops(self):
    self.flops, ret = 0, self.flops
    return ret

InterpretedFlopCounter: Dict[Op, Callable] = {
  BufferOps.LOAD: lambda arg: FlopCounter(arg.st.shape, arg.dtype, 0, {arg.idx: arg.dtype.itemsize*arg.st.real_size()}),
  BufferOps.CONST: lambda arg: FlopCounter(arg.st.shape, arg.dtype, 0, {}),
  BufferOps.STORE: lambda self,arg: FlopCounter(arg.st.shape, arg.dtype, self.consume_flops(), {**self.mem, arg.idx: arg.dtype.itemsize*arg.st.real_size()}),  # noqa: E501
  UnaryOps.CAST: lambda self,arg: FlopCounter(self.shape, arg[0], self.consume_flops(), self.mem),   # cast uses no flops
  **{op:lambda self: FlopCounter(self.shape, self.dtype, self.consume_flops() + prod(self.shape), self.mem) for op in UnaryOps if op != UnaryOps.CAST},  # noqa: E501
  **{op:lambda self,y,op=op: FlopCounter(self.shape,  dtypes.bool if op in (BinaryOps.CMPLT, BinaryOps.CMPEQ) else self.dtype, self.consume_flops() + y.consume_flops() + prod(self.shape), {**self.mem, **y.mem}) for op in BinaryOps},  # noqa: E501
  **{op:lambda self,axis: FlopCounter(tuple(1 if i in axis else s for i,s in enumerate(self.shape)), self.dtype, self.consume_flops() + prod(self.shape), self.mem) for op in ReduceOps},  # noqa: E501
  TernaryOps.WHERE: lambda self,y,z: FlopCounter(self.shape, y.dtype, self.consume_flops() + y.consume_flops() + z.consume_flops() + prod(self.shape), {**self.mem, **y.mem, **z.mem})}  # noqa: E501

@functools.lru_cache(None)
def get_lazyop_info(ast:LazyOp) -> FlopCounter:
  @functools.lru_cache(None) # NOTE: this cache needs to be recreated for new ASTs
  def run_ast(ast): return InterpretedFlopCounter[ast.op](*([run_ast(x) for x in ast.src]+([ast.arg] if ast.arg is not None else [])))
  return run_ast(ast)

# **************** global state Counters ****************

class GlobalCounters:
  global_ops: ClassVar[int] = 0
  global_mem: ClassVar[int] = 0
  time_sum_s: ClassVar[float] = 0.0
  kernel_count: ClassVar[int] = 0
  mem_used: ClassVar[int] = 0   # NOTE: this is not reset
  @staticmethod
  def reset(): GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.time_sum_s, GlobalCounters.kernel_count = 0,0,0.0,0


# tinygrad/realize.py
  
import sys
from collections import defaultdict
from typing import List, Dict, Optional, cast, Set, DefaultDict
from tinygrad.ops import LoadOps, ScheduleItem, BufferOps, GlobalCounters, LazyOp, ReduceOps, ConstBuffer, MemBuffer, BinaryOps, UnaryOps
from tinygrad.device import Device, Buffer, BufferCopy, BufferXfer, BufferRead, JITRunner, update_stats, Compiled, BufferOptions
from tinygrad.features.graph import print_tree, realized_lazybuffer, log_lazybuffer
from tinygrad.helpers import colored, getenv, GRAPH, cpu_time_execution, DEBUG, flatten, prod, dedup, all_int
from tinygrad.shape.symbolic import Variable
from tinygrad.dtype import ImageDType, dtypes
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.shapetracker import ShapeTracker

# *** schedule running ***

class CustomOp(JITRunner):
  def __init__(self, fxn):
    self.fxn = fxn
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False): self.fxn(*rawbufs)

class SyncOp(JITRunner):
  def __init__(self, device):
    self.device, self.dname = Device[device], device
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Variable, int], wait=False, jit=False):
    et = cpu_time_execution(self.device.synchronize, enable=wait or DEBUG >= 1)
    update_stats(colored("synchronize", "RED"), 0, 0, {}, et, 1, device=self.dname)

def lower_schedule_item(si:ScheduleItem) -> Optional[JITRunner]:
  assert all(si.out.device == x.device for x in si.inputs) or si.ast.op in {LoadOps.COPY, LoadOps.WAIT}, \
    f"all devices must be the same, {si.out.device} != {[x.device for x in si.inputs]} {print_tree(si.ast) or ''}"
  if si.ast.op is LoadOps.EMPTY: return None
  if si.ast.op in {LoadOps.SYNC, LoadOps.WAIT, LoadOps.COPY} and si.out.device.startswith("HIP") and si.inputs[0].device.startswith("HIP"):
    from tinygrad.runtime.ops_hip import HIPSyncEvent, HIPWaitEvent
    if si.ast.op is LoadOps.SYNC: return HIPSyncEvent(si.out)
    if si.ast.op is LoadOps.WAIT: return HIPWaitEvent(si.out.device)
  if si.ast.op in {LoadOps.SYNC, LoadOps.WAIT} and si.out.device.startswith("HSA") and si.inputs[0].device.startswith("HSA"):
    # Our HSA runtime handles synchronization
    if si.ast.op is LoadOps.SYNC: return None
  if si.ast.op is LoadOps.COPY:
    if hasattr(Device[si.out.device].allocator, 'transfer') and type(Device[si.out.device]) is type(Device[si.inputs[0].device]): return BufferXfer()
    if si.inputs[0].device.startswith("DISK"): return BufferRead()
    return BufferCopy()
  if si.ast.op is LoadOps.CUSTOM: return CustomOp(si.ast.arg)
  if si.ast.op is LoadOps.SYNC: return SyncOp(si.out.device) if isinstance(Device[si.out.device], Compiled) else None
  if si.ast.op is LoadOps.WAIT: return None
  return Device[si.out.device].get_runner(si.ast)

logops = open(getenv("LOGOPS", ""), "a") if getenv("LOGOPS", "") else None
def run_schedule(schedule:List[ScheduleItem]):
  while len(schedule):
    si = schedule.pop(0)
    if logops and si.ast.op not in LoadOps: logops.write(str(si.ast)+"\n")

    # get the program
    prg = lower_schedule_item(si)

    # invalidate the output buffer if there's a non contig usage of it in inputs
    if si.out.output_buffer is not None:
      for i,a in enumerate(si.inputs):
        if a.realized == si.out.output_buffer:
          if any(not x.arg.st.contiguous for x in si.ast.lazyops if x.op is BufferOps.LOAD and x.arg.idx == i+1):
            si.out.output_buffer = None
            break

    # we don't have an output buffer, we have to create it, and create to max size if it has symbolic shape
    if si.out.size > 0:
      options = BufferOptions(host=True, signal=True) if si.ast.op is LoadOps.SYNC else None
      si.out.realized = si.out.output_buffer if si.out.output_buffer is not None else \
        Buffer(si.out.device, si.out.size, si.out.dtype, "PLACEHOLDER" if getattr(prg, "skip_allocation", False) else None, options=options)
      del si.out.srcs

    # run the function (put it in JIT)
    real_buffers = [x.realized for x in (si.out,)+si.inputs if x.size != 0]
    assert all(x is not None for x in real_buffers), f"can't run, some inputs aren't realized {real_buffers}"
    if prg: prg.exec(cast(List[Buffer], real_buffers), si.var_vals)
    elif si.out.size > 0: update_stats(colored(f"empty {si.out.st.size:10d} {si.out.dtype}", "yellow"), 0, 0, {}, None, 1, device=si.out.device)
    if GRAPH: realized_lazybuffer(si.out, GlobalCounters.kernel_count)

# *** schedule creation ***

# creation can recurse a lot
sys.setrecursionlimit(10000)

# recursively create a lazyop
def _recursive_lazyop(buf:LazyBuffer, inputs:List[LazyBuffer], var_vals:Dict[Variable, int], st:ShapeTracker,
                      realizes:Set[LazyBuffer], cache, first=True) -> LazyOp:
  if (buf, st) in cache: return cache[(buf, st)]
  if buf != buf.base:
    st = buf.st + st
    buf = buf.base
  # all buffers here are base now
  assert buf.op is not None

  # consts are always fused and generated
  if buf.op is LoadOps.CONST:
    unbound_st, st_var_vals = st.simplify().unbind()
    var_vals.update(st_var_vals)
    return LazyOp(BufferOps.CONST, (), ConstBuffer(buf.arg, buf.dtype, unbound_st))

  # if we aren't fusing it, it's a load and we add it to the inputs
  if buf.realized or (buf in realizes and not first):
    if buf not in inputs: inputs.append(buf)
    unbound_st, st_var_vals = st.simplify().unbind()
    var_vals.update(st_var_vals)
    return LazyOp(BufferOps.LOAD, (), MemBuffer(inputs.index(buf)+1, buf.dtype, unbound_st))

  # if a CONTIGUOUS made it all the way here, just skip it
  if buf.op is LoadOps.CONTIGUOUS:
    assert first
    return _recursive_lazyop(buf.srcs[0], inputs, var_vals, st, realizes, cache, False)

  # if it's a reduce, we have to change the shapetracker
  if buf.op in ReduceOps:
    assert st.contiguous, "ReduceOps late fusion must be contiguous"
    st = ShapeTracker.from_shape(buf.srcs[0].shape)

  # otherwise we fuse it like normal
  cache[(buf, st)] = ret = LazyOp(buf.op, tuple(_recursive_lazyop(x, inputs, var_vals, st, realizes, cache, False) for x in buf.srcs), buf.arg)
  return ret

# recursively walk back in the graph to create the schedule
def _recursive_schedule(out:LazyBuffer, seen:Set[LazyBuffer], realizes:Set[LazyBuffer],
                        reduce_for_op: Dict[LazyBuffer, LazyBuffer]) -> List[ScheduleItem]:
  if out in seen or out.realized or out.op == LoadOps.CONST: return []
  assert out.base == out
  seen.add(out)

  inputs: List[LazyBuffer] = []
  var_vals: Dict[Variable, int] = out.st.var_vals.copy()
  if out.op in {LoadOps.CUSTOM, LoadOps.SYNC, LoadOps.WAIT, LoadOps.COPY, LoadOps.EMPTY}:
    op, inputs = LazyOp(out.op, (), out.arg), list(out.srcs)
  else:
    output_st = ShapeTracker.from_shape(reduce_for_op[out].shape if out in reduce_for_op else out.shape)
    op = _recursive_lazyop(out, inputs, var_vals, output_st, realizes, cache={})
    op = LazyOp(BufferOps.STORE, (op, ), MemBuffer(0, out.dtype, output_st.simplify().unbind()[0]))

  return flatten(_recursive_schedule(x.base, seen, realizes, reduce_for_op) for x in inputs) + [ScheduleItem(op, out, tuple(inputs), var_vals)]

# recursively search the entire graph for all LazyBuffers, insert realizes after expands
def _recurse_lb(buf:LazyBuffer, realizes:Set[LazyBuffer], allbufs:Dict[LazyBuffer, None],
                simple_pads:Set[LazyBuffer], children:DefaultDict[LazyBuffer, Dict[LazyBuffer, None]], scheduled=False):
  if buf in allbufs or buf.base.realized: return
  if GRAPH: log_lazybuffer(buf, scheduled)
  if isinstance(buf.dtype, ImageDType) and (prod(buf.shape) != prod(buf.dtype.shape) or
                                            not any(buf.shape[x]%4 == 0 for x in buf.st.unit_stride_axes())):
    if DEBUG >= 3: print(f"forcing image {buf.dtype} with shape {buf.shape} to float32")
    buf.dtype = dtypes.float32  # NOTE: this is what makes the dtype above not match
  if buf.base != buf:
    # realize all places where the buffer is expanded
    if prod(buf.base.st.shape) < prod(buf.st.shape):
      if len(buf.st.views) == 1 and buf.st.views[-1].mask and all_int(buf.base.st.shape) and \
          prod(buf.base.st.shape) >= prod([y-x for x,y in buf.st.views[-1].mask]):
        simple_pads.add(buf.base)
      else:
        realizes.add(buf.base)
    return _recurse_lb(buf.base, realizes, allbufs, simple_pads, children)
  if buf.forced_realize: realizes.add(buf)
  allbufs[buf] = None
  if buf.op in LoadOps: realizes.add(buf.base)
  if buf.op == LoadOps.COPY:
    assert buf.srcs[0].st.contiguous and buf.srcs[0].size == buf.srcs[0].base.size, "can only copy contig"
    realizes.add(buf.srcs[0].base)
  for x in buf.srcs:
    children[x.base][buf] = None
    _recurse_lb(x, realizes, allbufs, simple_pads, children)

UNSAFE_PAD_OPS = {BinaryOps.DIV, BinaryOps.CMPLT, BinaryOps.CMPEQ, UnaryOps.LOG2, UnaryOps.EXP2}
def _is_padding_okay(buf:LazyBuffer, realizes:Set[LazyBuffer]) -> bool:
  if buf in realizes or buf.realized: return True
  # NOTE: this broke to_image_idx and coder with JIT
  if buf.op in UNSAFE_PAD_OPS: return False
  return all(_is_padding_okay(x.base, realizes) for x in buf.srcs)

def create_schedule(outs:List[LazyBuffer], seen:Optional[Set[LazyBuffer]]=None) -> List[ScheduleItem]:
  if seen is None: seen = set()

  # start by just realizing the buffers passed in
  realizes: Set[LazyBuffer] = set([x.base for x in outs if not x.base.realized])
  allbufs: Dict[LazyBuffer, None] = {}
  simple_pads: Set[LazyBuffer] = set()
  children: DefaultDict[LazyBuffer, Dict[LazyBuffer, None]] = defaultdict(dict)
  for out in outs: _recurse_lb(out.base, realizes, allbufs, simple_pads, children, scheduled=True)

  # check if we have to realize pads
  for p in simple_pads:
    if not _is_padding_okay(p, realizes):
      realizes.add(p)

  # find all reduces, and pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  reduce_for_op: Dict[LazyBuffer, LazyBuffer] = {}
  for r in allbufs.keys():
    if r != r.base or r.op not in ReduceOps or r in realizes: continue

    # follow the reduce down
    child_set: Dict[LazyBuffer, ShapeTracker] = {r: r.st}
    realized_children: Dict[LazyBuffer, ShapeTracker] = {}
    forced_realize = False
    can_chase = True
    while not forced_realize and len(child_set):
      next_child_set = {}
      for tr,st in child_set.items():
        if tr in realizes:
          realized_children[tr] = st
          # can only have one output buffer
          # can only reduce contiguous
          # max one reduceop per kernel
          if len(realized_children) > 1 or not st.contiguous or st.size != r.st.size or (tr in reduce_for_op and reduce_for_op[tr] != r):
            can_chase = tr not in reduce_for_op or reduce_for_op[tr] == r
            forced_realize = True
            break
          continue
        for tr_next in children[tr].keys():
          if not tr_next.realized:
            # max one reduceop per kernel
            if tr_next.op in ReduceOps:
              forced_realize = True
              break
            st_childs = dedup([s for s in tr_next.srcs if s.base == tr])
            if len(st_childs) > 1:
              forced_realize = True
              break
            next_child_set[tr_next] = st + st_childs[0].st
      child_set = next_child_set
    if forced_realize:
      tr = r
      if can_chase:
        # can chase this down to contiguous children
        st = tr.st
        while len(children[tr]) == 1:
          tr_next = next(iter(children[tr].keys()))
          st_childs = dedup([s for s in tr_next.srcs if s.base == tr])
          if len(st_childs) > 1: break
          if st.size != st_childs[0].st.size: break
          st = st + st_childs[0].st
          if not st.contiguous or tr_next.op in ReduceOps: break
          tr = tr_next
        reduce_for_op[tr] = r
      realizes.add(tr)
    else:
      assert len(realized_children) == 1
      reduce_for_op[next(iter(realized_children.keys()))] = r

  return flatten(_recursive_schedule(x.base, seen, realizes, reduce_for_op) for x in outs)


# tinygrad/tensor.py

# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from __future__ import annotations
import time, math, itertools
from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union, Sequence, Iterable, Dict, DefaultDict, cast, get_args
from collections import defaultdict
from functools import partialmethod, reduce
import numpy as np

from tinygrad.dtype import DType, dtypes, ImageDType, Scalar, least_upper_float, least_upper_dtype, cast_scalar
from tinygrad.helpers import argfix, make_pair, getenv, IMAGE, DEBUG, WINO, flatten, prod, all_int, round_up, merge_dicts, fully_flatten, flat_mv
from tinygrad.lazy import LazyBuffer
from tinygrad.features.multi import MultiLazyBuffer
from tinygrad.ops import LoadOps
from tinygrad.device import Device, Buffer
from tinygrad.shape.symbolic import sint
from tinygrad.realize import run_schedule, create_schedule

# **** start with two base classes, Tensor and Function ****

class Function:
  def __init__(self, device:Union[str, Tuple[str, ...]], *tensors:Tensor):
    self.device = device
    self.needs_input_grad = [t.requires_grad for t in tensors]
    self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
    if self.requires_grad: self.parents = tensors

  def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise RuntimeError(f"backward not implemented for {type(self)}")

  @classmethod
  def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
    ctx = fxn(x[0].device, *x)
    ret = Tensor.__new__(Tensor)
    ret.lazydata, ret.requires_grad, ret.grad = ctx.forward(*[t.lazydata for t in x], **kwargs), ctx.requires_grad, None
    ret._ctx = ctx if ctx.requires_grad and not Tensor.no_grad else None  # used by autograd engine
    return ret

import tinygrad.mlops as mlops

def _loadop(op, shape:Tuple[sint,...], dtype:DType, device:Union[str, Tuple[str, ...]], arg=None, src:Optional[LazyBuffer]=None):
  if isinstance(device, str): return LazyBuffer.loadop(op, shape, dtype, device, arg, src)
  return MultiLazyBuffer([LazyBuffer.loadop(op, shape, dtype, d, arg, src) for d in device], None)

def _fromcpu(x: np.ndarray) -> LazyBuffer:
  ret = LazyBuffer.loadop(LoadOps.EMPTY, x.shape, dtypes.from_np(x.dtype), "EXT")
  if x.size == 0:
    ret.realized = Buffer("EXT", 0, dtypes.from_np(x.dtype), (memoryview(bytearray()), None))
  else:
    ret.realized = Buffer("EXT", prod(x.shape), dtypes.from_np(x.dtype), (flat_mv(np.require(x, requirements='C').data), x))
  return ret

def _get_winograd_matcols(mat, dims:int, shp:Tuple[sint, ...], device:Union[str, Tuple[str, ...]]) -> List[List[Tensor]]:
  return [[Tensor.cat(*[Tensor.full(shp[:dim] + (1,) + shp[dim+1:], float(m[k]), device=device) for m in mat], dim=dim)
           for k in range(len(mat[0]))] for dim in range(dims)]

# winograd conv 3 kernel f(4x4,3x3) see: http://arxiv.org/abs/1509.09308
def _apply_winograd_matrix(mat, t:Tensor, dims:int):
  # multiply mat_1 @ mat_2 @ t with foldable constants, where mat_i acts on vector t along dimension i; roughly kron(mat, mat) @ t
  # due to realize-before-expand rule in lazy.py, we must operate in this order: reshape -> expand -> arithmetic
  t_ = t.reshape(t.shape[:dims] + (1,) * dims + t.shape[dims:]).expand(t.shape[:dims] + (len(mat),) * dims + t.shape[dims:])  # add output dims
  # precalculate mat columns for each dim; prod(itertools.product(matcols)) gives the columns of kron(mat, mat, ...)
  matcols = _get_winograd_matcols(mat, dims, t_.shape[dims:], t_.device)
  # multiply each element of t_ by the corresponding stacked column of kron(mat, mat), producing only one view for each element of t
  return sum(prod(col[idx] for col, idx in zip(matcols, mat_is)) * t_[mat_is] for mat_is in itertools.product(range(len(mat[0])), repeat=dims))

class Tensor:
  __slots__ = "lazydata", "requires_grad", "grad", "_ctx"
  __deletable__ = ('_ctx',)
  training: ClassVar[bool] = False
  class train:
    def __init__(self, val=True): self.val = val
    def __enter__(self): self.prev, Tensor.training = Tensor.training, self.val
    def __exit__(self, exc_type, exc_value, traceback): Tensor.training = self.prev

  no_grad: ClassVar[bool] = False
  def __init__(self, data:Union[None, Scalar, List, Tuple, LazyBuffer, np.ndarray, bytes, MultiLazyBuffer],
               device:Optional[Union[str, tuple, list]]=None, dtype:Optional[DType]=None, requires_grad:Optional[bool]=None):
    assert dtype is None or isinstance(dtype, DType), f"invalid dtype {dtype}"
    device = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)
    # tensors have gradients, buffers do not
    self.grad: Optional[Tensor] = None

    # NOTE: this can be in three states. False and None: no gradient, True: gradient
    # None (the default) will be updated to True if it's put in an optimizer
    self.requires_grad: Optional[bool] = requires_grad

    # internal variables used for autograd graph construction
    self._ctx: Optional[Function] = None
    if isinstance(data, LazyBuffer): assert dtype is None or dtype == data.dtype, "dtype doesn't match, and casting isn't supported"
    elif isinstance(data, get_args(Scalar)): data = _loadop(LoadOps.CONST, tuple(), dtype or dtypes.from_py(data), device, data)
    elif isinstance(data, bytes): data = _fromcpu(np.frombuffer(data, np.uint8))
    elif data is None: data = _loadop(LoadOps.EMPTY, (0,), dtype or dtypes.default_float, device)
    elif isinstance(data, list):
      if (d := fully_flatten(data)) and all(isinstance(s, bool) for s in d): dtype = dtype or dtypes.bool
      elif d and all_int(d): dtype = dtype or dtypes.default_int
      else: dtype = dtype or dtypes.default_float
      # NOTE: cast at the end for the dtypes that do not have a numpy dtype
      data = _fromcpu(np.array(data, dtype.np)).cast(dtype)
    elif isinstance(data, np.ndarray):
      if data.shape == (): data = _loadop(LoadOps.CONST, tuple(), dtype or dtypes.from_np(data.dtype), device, data.item())
      else: data = _fromcpu(data.astype(dtype.np) if dtype is not None and dtype.np is not None else data)

    # data is a LazyBuffer, but it might be on the wrong device
    if not isinstance(data, (LazyBuffer, MultiLazyBuffer)): raise RuntimeError(f"can't create Tensor from {data!r} with type {type(data)}")
    if isinstance(device, tuple):
      # TODO: what if it's a MultiLazyBuffer on other devices?
      self.lazydata: Union[LazyBuffer, MultiLazyBuffer] = MultiLazyBuffer.from_sharded(data, device, None) if isinstance(data, LazyBuffer) else data
    else:
      self.lazydata = data if data.device == device else data.copy_to_device(device)

  def __repr__(self):
    return f"<Tensor {self.lazydata!r} on {self.device} with grad {(self.grad.lazydata if self.grad else None)!r}>"

  # Python has a non moving GC, so this should be okay
  def __hash__(self): return id(self)

  @property
  def device(self) -> Union[str, Tuple[str, ...]]: return self.lazydata.device

  @property
  def shape(self) -> Tuple[sint, ...]: return self.lazydata.shape

  @property
  def dtype(self) -> DType: return self.lazydata.dtype

  # ***** data handlers ****

  @staticmethod
  def corealize(lst:Iterable[Tensor]):
    run_schedule(create_schedule(flatten([x.lazydata.lbs if isinstance(x.lazydata, MultiLazyBuffer) else [x.lazydata] for x in lst])))

  def realize(self) -> Tensor:
    Tensor.corealize([self])
    return self

  def assign(self, x) -> Tensor:
    # TODO: this is a hack for writing to DISK. remove with working assign
    if isinstance(self.device, str) and self.device.startswith("DISK"):
      if x.__class__ is not Tensor: x = Tensor(x, device="EXT", dtype=self.dtype)
      self.contiguous().realize().lazydata.base.realized.copyin(x.numpy().data)
      return self
    if x.__class__ is not Tensor: x = Tensor(x, device=self.device, dtype=self.dtype)
    # NOTE: we allow cross device assign
    assert self.shape == x.shape, f"assign shape mismatch {self.shape} != {x.shape}"
    if isinstance(self.lazydata, MultiLazyBuffer):
      assert self.lazydata.axis == x.lazydata.axis
    assert not x.requires_grad  # self requires_grad is okay?
    if DEBUG >= 4: print(f"assign {self.lazydata} <- {x.lazydata}")
    if self.dtype == x.dtype and not getenv("DISALLOW_ASSIGN"):
      if isinstance(self.lazydata, MultiLazyBuffer):
        for d,s in zip(x.lazydata.lbs, self.lazydata.lbs): d.output_buffer = s.base.realized
      else:
        if self.lazydata.base.realized is not None: x.lazydata.output_buffer = self.lazydata.base.realized
    self.lazydata = x.lazydata
    return self
  def detach(self) -> Tensor: return Tensor(self.lazydata, device=self.device, requires_grad=False)

  def _data(self) -> memoryview:
    if 0 in self.shape: return memoryview(bytearray(0))
    t = self if isinstance(self.device, str) else self.to(self.device[0])   # deal with multitensor
    return cast(Buffer, t.cast(t.dtype.scalar()).contiguous().realize().lazydata.base.realized).as_buffer()

  def data(self) -> memoryview:
    assert self.dtype.fmt is not None, f"no fmt dtype for {self.dtype}"
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    return self._data().cast(self.dtype.fmt, self.shape if len(self.shape) else (1,))
  def item(self) -> Scalar:
    assert self.dtype.fmt is not None, f"no fmt dtype for {self.dtype}"
    assert self.numel() == 1, "must have one element for item"
    return self._data().cast(self.dtype.fmt)[0]
  def numpy(self) -> np.ndarray:
    assert self.dtype.np is not None, f"no np dtype for {self.dtype}"
    assert all_int(self.shape), f"no data if shape is symbolic, {self.shape=}"
    return np.frombuffer(self._data(), dtype=self.dtype.np).reshape(self.shape)

  def to(self, device:Optional[Union[str, Tuple[str, ...]]]) -> Tensor:
    device = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)
    if device is None or device == self.device: return self
    if not isinstance(device, str): return self.shard(device)
    ret = Tensor(self.lazydata, device, requires_grad=self.requires_grad)
    if self.grad: ret.grad = self.grad.to(device)
    if hasattr(self, '_ctx'): ret._ctx = self._ctx
    return ret

  def to_(self, device:Optional[Union[str, Tuple[str, ...]]]):
    real = self.to(device)
    # TODO: is this assign?
    if self.grad is not None and real.grad is not None: self.grad.lazydata = real.grad.lazydata
    self.lazydata = real.lazydata

  def shard(self, devices:Tuple[str, ...], axis:Optional[int]=None) -> Tensor:
    assert isinstance(self.lazydata, LazyBuffer), "can't shard a MultiLazyBuffer"
    canonical_devices = tuple(Device.canonicalize(x) for x in devices)
    if axis is not None and axis < 0: axis += len(self.shape)
    return Tensor(MultiLazyBuffer.from_sharded(self.lazydata, canonical_devices, axis), device=canonical_devices, requires_grad=self.requires_grad)

  def shard_(self, devices:Tuple[str, ...], axis:Optional[int]=None):
    self.lazydata = self.shard(devices, axis).lazydata
    return self

  # ***** creation llop entrypoint *****

  @staticmethod
  def _loadop(op, shape, device:Optional[str]=None, dtype:Optional[DType]=None, arg=None, **kwargs):
    return Tensor(LazyBuffer.loadop(op, shape, dtype or dtypes.default_float, Device.canonicalize(device), arg), dtype=dtype, device=device, **kwargs)

  @staticmethod
  def empty(*shape, **kwargs): return Tensor._loadop(LoadOps.EMPTY, argfix(*shape), **kwargs)

  _seed: int = int(time.time())
  @staticmethod
  def manual_seed(seed=0): Tensor._seed = seed

  @staticmethod
  def rand(*shape, **kwargs): return Tensor._loadop(LoadOps.CUSTOM, argfix(*shape), arg=custom_random, **kwargs)

  # ***** creation helper functions *****

  @staticmethod
  def full(shape:Tuple[sint, ...], fill_value:Scalar, **kwargs):
    return Tensor(fill_value, **kwargs).reshape((1, )*len(new_shape := argfix(shape))).expand(new_shape)

  @staticmethod
  def zeros(*shape, **kwargs): return Tensor.full(argfix(*shape), 0.0, **kwargs)

  @staticmethod
  def ones(*shape, **kwargs): return Tensor.full(argfix(*shape), 1.0, **kwargs)

  @staticmethod
  def arange(start, stop=None, step=1, **kwargs):
    if stop is None: stop, start = start, 0
    dtype = kwargs.pop("dtype", dtypes.default_float if any(isinstance(x, float) for x in (start, stop, step)) else dtypes.default_int)
    return (Tensor.full((math.ceil((stop-start)/step),), step, dtype=dtype, **kwargs).cumsum() + (start - step)).cast(dtype)

  @staticmethod
  def eye(dim:int, **kwargs):
    return Tensor.ones((dim,1),**kwargs).pad((None,(0,dim))).flatten().shrink(((0,dim*dim),)).reshape(dim, dim)

  def full_like(self, fill_value:Scalar, **kwargs):
    return Tensor.full(self.shape, fill_value, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)
  def zeros_like(self, **kwargs): return self.full_like(0, **kwargs)
  def ones_like(self, **kwargs): return self.full_like(1, **kwargs)

  # ***** rng hlops *****

  @staticmethod
  def randn(*shape, dtype:Optional[DType]=None, **kwargs) -> Tensor:
    # https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    src = Tensor.rand((2, *argfix(*shape)), **kwargs)
    return src[0].mul(2*math.pi).cos().mul((1 - src[1]).log().mul(-2).sqrt()).cast(dtype or dtypes.default_float)

  @staticmethod
  def randint(*shape, low=0, high=10, **kwargs) -> Tensor: return Tensor.uniform(*shape, low=low, high=high, dtype=dtypes.int32, **kwargs)

  @staticmethod
  def normal(*shape, mean=0.0, std=1.0, **kwargs) -> Tensor: return (std * Tensor.randn(*shape, **kwargs)) + mean

  @staticmethod
  def uniform(*shape, low=0.0, high=1.0, **kwargs) -> Tensor:
    dtype = kwargs.pop("dtype", dtypes.default_float)
    return ((high-low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low

  @staticmethod
  def scaled_uniform(*shape, **kwargs) -> Tensor: return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul(prod(argfix(*shape))**-0.5)

  # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
  @staticmethod
  def glorot_uniform(*shape, **kwargs) -> Tensor:
    return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul((6/(argfix(*shape)[0]+prod(argfix(*shape)[1:])))**0.5)

  # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
  @staticmethod
  def kaiming_uniform(*shape, a:float = 0.01, **kwargs) -> Tensor:
    bound = math.sqrt(3.0) * math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:]))
    return Tensor.uniform(*shape, low=-bound, high=bound, **kwargs)

  # https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_
  @staticmethod
  def kaiming_normal(*shape, a:float = 0.01, **kwargs) -> Tensor:
    std = math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:]))
    return Tensor.normal(*shape, mean=0.0, std=std, **kwargs)

  def multinomial(self:Tensor, num_samples:int = 1, replacement:bool = False) -> Tensor:
    assert 1 <= self.ndim <= 2 and num_samples > 0, f"{self.ndim=} must be 1 or 2 dim, {num_samples=} must be positive"
    assert replacement or num_samples == 1, "no replacement only supports num_samples = 1"
    weight = self.unsqueeze(0) if self.ndim == 1 else self
    cdf = (cw := weight.cumsum(1).float()) / cw[:, -1].unsqueeze(1)
    unif_samples = Tensor.rand(num_samples, cdf.shape[0], 1, device=self.device)
    indices = (unif_samples.expand((-1, -1, cdf.shape[1])) >= cdf).sum(2).permute((1, 0))
    return (indices.squeeze(0) if self.ndim == 1 else indices).cast(dtypes.default_int)

  # ***** toposort and backward pass *****

  def deepwalk(self):
    def _deepwalk(node, visited):
      visited.add(node)
      if getattr(node, "_ctx", None):
        for i in node._ctx.parents:
          if i not in visited: yield from _deepwalk(i, visited)
        yield node
    return list(_deepwalk(self, set()))

  def backward(self) -> Tensor:
    assert self.shape == tuple(), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

    # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
    # this is "implicit gradient creation"
    self.grad = Tensor(1.0, device=self.device, requires_grad=False)

    for t0 in reversed(self.deepwalk()):
      if t0.grad is None: raise RuntimeError("tensor has no grad")
      grads = t0._ctx.backward(t0.grad.lazydata)
      grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
        for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
      for t, g in zip(t0._ctx.parents, grads):
        if g is not None and t.requires_grad:
          assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
          t.grad = g if t.grad is None else (t.grad + g)
      del t0._ctx
    return self

  # ***** movement mlops *****

  def reshape(self, shape, *args) -> Tensor:
    new_shape = argfix(shape, *args)
    new_shape = tuple([-prod(self.shape) // prod(new_shape) if s == -1 else (s if s is not None else self.shape[i]) for i,s in enumerate(new_shape)])
    return mlops.Reshape.apply(self, shape=new_shape) if new_shape != self.shape else self
  def expand(self, shape, *args) -> Tensor:
    new_shape = tuple([x if x != -1 and x is not None else s for s,x in zip(self.shape, argfix(shape, *args))])
    return mlops.Expand.apply(self, shape=new_shape) if new_shape != self.shape else self
  def permute(self, order, *args) -> Tensor: return mlops.Permute.apply(self, order=argfix(order, *args))
  def flip(self, axis, *args) -> Tensor: return mlops.Flip.apply(self, axis=[x if x >= 0 else x+len(self.shape) for x in argfix(axis, *args)])
  def shrink(self, arg:Tuple[Optional[Tuple[sint, sint]], ...]) -> Tensor:
    if all(x is None or x == (0,s) for x,s in zip(arg, self.shape)): return self
    return mlops.Shrink.apply(self, arg=tuple(x if x is not None else (0,s) for x,s in zip(arg, self.shape)))
  def pad(self, arg:Tuple[Optional[Tuple[sint, sint]], ...], value:float=0.0) -> Tensor:
    if all(x is None or x == (0,0) for x in arg): return self
    ret = mlops.Pad.apply(self, arg=(narg:=tuple(x if x is not None else (0,0) for x in arg)))
    return ret if 0 == value else ret + mlops.Pad.apply(Tensor.ones_like(self), arg=narg).where(0, value)

  # ***** movement hlops *****

  # Supported Indexing Implementations:
  #   1. Int indexing (no copy)
  #     - for all dims where there's int, shrink -> reshape
  #     - negative indices are taken relative to the end of the sequence, so X[-2] returns the 2nd-to-last element
  #     - X = Tensor.rand(4,5,9); X[2,-2] shrinks the Tensor to X.shrink(((2, 3), (3, 4), (0, 9))) -> X.shape=(1,1,9)
  #     - Then we reshape (collapse) the int dim away such that for X: (1,1,9) -> (9,)
  #   2. Slice indexing (no copy)
  #     - for all dims where slice is start:end:stride, shrink -> Optional[flip] -> pad -> reshape -> shrink
  #     - first shrink the Tensor to X.shrink(((start, end),))
  #     - then we apply stride through Optional[flip] -> pad -> reshape -> shrink
  #       - flip where dim value is negative
  #       - pad 0's on dims such that reshaping [dim_size_padded] -> [dim_size_padded // stride, stride] is possible
  #       - shrink [dim_size_padded // stride, stride] -> [dim_size_padded // stride, 1]
  #       - reshape [dim_size_padded // stride, 1] -> [dim_size_padded // stride] and now you have your stride
  #   3. None indexing (no copy)
  #     - reshape (inject) a dim at the dim where there's None
  #   4. Tensor indexing (copy)
  #     - use Tensor.arange == tensor_index to create a mask
  #     - apply mask to self by mask * self for dims where index is a tensor
  #     - (mask * self).sum(dim) to reduce to correct shape
  # Tiny Things:
  #   1. Supported indices: Union[int, slice, Tensor, None, List, Tuple, Ellipsis]
  #     - for any list, List[Union[List, Tuple, int]], must have homogeneous shape
  #     - for any tuple, Tuple[Union[List, Tuple, int]], must have homogeneous shape
  #   2. Bool indexing is not supported
  #   3. Out of bounds Tensor indexing results in 0
  #     - e.g: Tensor([1, 2, 3])[Tensor([4, 3, 2])] -> [0, 0, 3] index 4 and 3 are OOB
  def __getitem__(self, indices) -> Tensor:
    # 1. indices normalization and validation
    # treat internal tuples and lists as Tensors and standardize indices to list type
    if isinstance(indices, list) and all_int(indices): indices = [Tensor(indices, self.device, requires_grad=False)]
    elif isinstance(indices, (tuple, list)):
      indices = [Tensor(list(i), self.device, requires_grad=False) if isinstance(i, (tuple, list)) else i for i in indices]
    else: indices = [indices]

    # turn scalar Tensors into const val for int indexing if possible
    indices = [self._to_const_val(i) if isinstance(i, Tensor) else i for i in indices]
    # move Tensor indices to the same device as self
    indices = [i.to(self.device) if isinstance(i, Tensor) else i for i in indices]

    # filter ellipsis and fill with slice(None) or fill rest of indices with slice(None)
    ellipsis_idx = [dim for dim, i in enumerate(indices) if i is Ellipsis]
    fill_idx = ellipsis_idx[0] if ellipsis_idx else len(indices)
    num_indices = len(indices) - len(ellipsis_idx) - sum(1 for i in indices if i is None)
    indices[fill_idx:fill_idx+1] = [slice(None)] * (len(self.shape) - num_indices)

    # use Dict[type, List[dimension]] to track elements in indices
    type_dim: DefaultDict[Union[type, None], List[int]] = defaultdict(list)

    # record None for dimension injection later and filter None and record rest of indices
    type_dim[None] = [dim for dim, i in enumerate(indices) if i is None]
    indices_filtered = [v for v in indices if v is not None]
    for dim,i in enumerate(indices_filtered): type_dim[type(i)].append(dim)

    for index_type in type_dim:
      if index_type not in [None, int, slice, Tensor]: raise IndexError(f"{index_type=} not supported")
    if len(ellipsis_idx) > 1: raise IndexError("indices can only have a single ellipsis ('...')")
    if num_indices > self.ndim: raise IndexError(f"too many {num_indices=} for {self.ndim=}")

    # 2. basic indexing, uses only movement ops (no copy)
    # currently indices_filtered: Tuple[Union[slice, int, Tensor], ...]
    # turn indices in indices_filtered to Tuple[shrink_arg, strides]
    for dim in type_dim[int]:
      if (index := indices_filtered[dim]) >= (size := self.shape[dim]) or index < -size:
        raise IndexError(f"{index=} is out of bounds on {dim=} with {size=}")
      indices_filtered[dim] = ((index, index+1), 1) if index >= 0 else ((size+index, size+index+1), 1)
    for dim in type_dim[slice]:
      if (index := indices_filtered[dim]).step == 0: raise ValueError(f"{index=} on {dim=} cannot have 0 as step")
      s, e, st = index.indices(self.shape[dim])
      indices_filtered[dim] = ((0, 0) if (st * (e - s)) < 0 else (s, e) if st > 0 else (e+1, s+1), st)
    # record tensors and skip all Tensor dims for basic indexing
    tensor_index: List[Tensor] = []
    for dim in type_dim[Tensor]:
      tensor_index.append(index := indices_filtered[dim])
      if not dtypes.is_int(index.dtype): raise IndexError(f"{index.dtype=} on {dim=} is not supported, only int tensor indexing is supported")
      indices_filtered[dim] = ((0, self.shape[dim]), 1)

    new_slice, strides = ((),()) if not indices_filtered else zip(*indices_filtered)
    ret = self.shrink(new_slice).flip(tuple(i for i, s in enumerate(strides) if s < 0))
    if any(abs(s) != 1 for s in strides):
      strides = tuple(abs(s) for s in strides)
      ret = ret.pad(tuple((0, round_up(sh, s) - sh) for s, sh in zip(strides, ret.shape)))
      ret = ret.reshape(tuple(flatten((sh // s, s) for s, sh in zip(strides, ret.shape))))
      ret = ret.shrink(tuple(flatten(((0, sh), (0, 1)) for sh in ret.shape[::2]))).reshape(ret.shape[::2])

    # inject 1 for dim where it's None and collapse dim for int
    new_shape = list(ret.shape)
    for dim in type_dim[None]: new_shape.insert(dim, 1)
    for dim in (dims_collapsed := tuple(dim + sum(1 for d in type_dim[None] if dim >= d) for dim in reversed(type_dim[int]))): new_shape.pop(dim)

    ret = ret.reshape(new_shape)
    assert all_int(ret.shape), f"does not support symbolic shape {ret.shape}"

    # 3. advanced indexing (copy)
    if type_dim[Tensor]:
      # calculate dim of current ret by subtracting dims collapsed and adding dims injected up until tensor_dim
      def calc_dim(tensor_dim:int) -> int:
        return tensor_dim - sum(1 for d in dims_collapsed if tensor_dim >= d) + sum(1 for d in type_dim[None] if tensor_dim >= d)

      # track tensor_dim and tensor_index using a dict
      # calc_dim to get dim and use that to normalize the negative tensor indices
      idx: Dict[int,Tensor] = {(dim := calc_dim(td)):(tensor<0).where(ret.shape[dim],0) + tensor for td,tensor in zip(type_dim[Tensor],tensor_index)}

      # compute sum_dim, arange, and idx
      max_idx_dim, first_dim, last_dim = max(i.ndim for i in idx.values()), min(idx.keys()), max(idx.keys())
      sum_dim = tuple(d if n==0 else d+max_idx_dim-n for n,d in enumerate(idx.keys()))
      arange = [Tensor.arange(ret.shape[d], requires_grad=False, device=self.device).reshape(ret.shape[d], *[1]*(ret.ndim+max_idx_dim-n-sd-1)) \
                for n,(sd,d) in enumerate(zip(sum_dim, idx.keys()))]
      reshaped_idx = [i.reshape(i.shape + (1,)*(ret.ndim - first_dim - (n or 1))) for n,i in enumerate(idx.values())]
      ret = ret.reshape(ret.shape[:first_dim+1] + (1,)*max_idx_dim + ret.shape[first_dim+1:])

      # iteratively eq -> mul -> sum fancy index
      try:
        for a,i,sd in zip(arange, reshaped_idx, sum_dim): ret = (a==i).mul(ret).sum(sd)
      except AssertionError as exc: raise IndexError("cannot broadcast indices") from exc

      # special permute case
      if first_dim != 0 and len(idx) != 1 and tuple(idx.keys()) != tuple(range(first_dim, last_dim+1)):
        ret_dims = list(range(ret.ndim))
        ret = ret.permute(ret_dims[first_dim:first_dim+max_idx_dim] + ret_dims[:first_dim] + ret_dims[first_dim+max_idx_dim:])
    return ret

  def __setitem__(self,indices,v): return self.__getitem__(indices).assign(v)

  # NOTE: using slice is discouraged and things should migrate to pad and shrink
  def slice(self, arg:Sequence[Optional[Tuple[int, sint]]], value:float=0) -> Tensor:
    arg_ = tuple(a if a is not None else (0, s) for s,a in zip(self.shape, arg))
    padding = tuple((max(0, -l), max(0, r-s)) for s,(l,r) in zip(self.shape, arg_))
    return self.pad(padding, value=value).shrink(tuple((l + pl, r + pl) for (l,r),(pl,_) in zip(arg_, padding)))

  def gather(self:Tensor, idx:Tensor, dim:int) -> Tensor:
    assert idx.ndim == self.ndim, "self.ndim must equal idx.ndim"
    assert all(s >= i for s,i in zip(self.shape, idx.shape)), "all dim of idx.shape must be smaller than self.shape"
    if dim < 0: dim += self.ndim
    idx = idx.to(self.device).transpose(ax1=dim, ax2=0).unsqueeze(-1)
    permarg = list(range(self.ndim))
    permarg = permarg[1:dim] + [permarg[0]] + permarg[dim+1:] + [permarg[dim]] if dim != 0 else permarg[1:] + [permarg[0]]
    return ((idx == Tensor.arange(self.shape[dim], requires_grad=False, device=self.device)) * self.permute(*permarg).shrink(
      tuple([*[(0,sh) for sh in idx.shape[1:-1]], (0,self.shape[dim])])).unsqueeze(0)).sum(-1).transpose(ax1=0, ax2=dim)

  def cat(self:Tensor, *args:Tensor, dim:int=0) -> Tensor:
    if dim < 0: dim += self.ndim
    assert all(len(y.shape) == len(self.shape) and all(y.shape[i] == s for i,s in enumerate(self.shape) if i != dim) for y in args)
    catargs = [self, *args]
    cat_dims = [s.shape[dim] for s in catargs]
    cat_dim_cumsum = [0, *itertools.accumulate(cat_dims)]
    slc:List[List[Optional[Tuple[sint, sint]]]] = [[None for _ in self.shape] for _ in catargs]
    for d,k,s in zip(cat_dims, cat_dim_cumsum[:-1], slc): s[dim] = (k, cat_dim_cumsum[-1] - k - d)
    return reduce(Tensor.__add__, [arg.pad(tuple(s)) for arg,s in zip(catargs, slc)])

  @staticmethod
  def stack(tensors:Sequence[Tensor], dim:int=0) -> Tensor:
    unsqueezed_tensors = [tensor.unsqueeze(dim) for tensor in tensors]
    # checks for shapes and number of dimensions delegated to cat
    return unsqueezed_tensors[0].cat(*unsqueezed_tensors[1:], dim=dim)

  def repeat(self, repeats:Sequence[int]) -> Tensor:
    base_shape = (1,) * (len(repeats) - self.ndim) + self.shape
    new_shape = [x for b in base_shape for x in [1, b]]
    expand_shape = [x for rs in zip(repeats, base_shape) for x in rs]
    final_shape = [r*s for r,s in zip(repeats, base_shape)]
    return self.reshape(new_shape).expand(expand_shape).reshape(final_shape)

  def _resolve_dim(self, dim:int, *, outer:bool=False) -> int:
    if not -max(1, self.ndim+outer) <= dim < max(1, self.ndim+outer):
      raise IndexError(f"{dim=} out of range {[-max(1, self.ndim+outer), max(1, self.ndim+outer)-1]}")
    return dim + self.ndim+outer if dim < 0 else dim

  def split(self, sizes:Union[int, List[int]], dim:int=0) -> Tuple[Tensor, ...]:
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    dim = self._resolve_dim(dim)
    if isinstance(sizes, int): sizes = [min(sizes, self.shape[dim]-i) for i in range(0, max(1, self.shape[dim]), max(1, sizes))]
    assert sum(sizes) == self.shape[dim], f"expect sizes to sum exactly to {self.shape[dim]}, but got {sum(sizes)}"
    return tuple(self[sl] for sl in [tuple([slice(None)]*dim + [slice(sum(sizes[:i]), sum(sizes[:i + 1]))]) for i in range(len(sizes))])

  def chunk(self, num:int, dim:int=0) -> List[Tensor]:
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    dim = self._resolve_dim(dim)
    assert num > 0, f"expect num to be greater than 0, got: {num}"
    return list(self.split(math.ceil(self.shape[dim]/num) if self.shape[dim] else [0]*num, dim=dim))

  def squeeze(self, dim:Optional[int]=None) -> Tensor:
    if dim is None: return self.reshape(tuple(dim for dim in self.shape if dim != 1))
    dim = self._resolve_dim(dim)
    return self if not self.ndim or self.shape[dim] != 1 else self.reshape(self.shape[:dim] + self.shape[dim+1:])

  def unsqueeze(self, dim:int) -> Tensor:
    dim = self._resolve_dim(dim, outer=True)
    return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

  # (padding_left, padding_right, padding_top, padding_bottom)
  def pad2d(self, padding:Sequence[int], value:float=0) -> Tensor:
    slc = [(-p0, s+p1) for p0,p1,s in zip(padding[::2], padding[1::2], self.shape[::-1])][::-1]
    return self.slice([(0,s) for s in self.shape[:-(len(padding)//2)]] + slc, value=value)

  @property
  def T(self) -> Tensor: return self.transpose()
  def transpose(self, ax1=1, ax2=0) -> Tensor:
    order = list(range(self.ndim))
    order[ax1], order[ax2] = order[ax2], order[ax1]
    return self.permute(order)
  def flatten(self, start_dim=0, end_dim=-1):
    start_dim, end_dim = start_dim + self.ndim if start_dim < 0 else start_dim, end_dim + self.ndim if end_dim < 0 else end_dim
    return self.reshape(self.shape[:start_dim] + (prod(self.shape[start_dim:end_dim+1]), ) + self.shape[end_dim+1:])
  def unflatten(self, dim:int, sizes:Tuple[int,...]):
    if dim < 0: dim += self.ndim
    return self.reshape(self.shape[:dim] + sizes + self.shape[dim+1:])

  # ***** reduce ops *****

  def _reduce(self, fxn:Type[Function], axis:Optional[Union[int, Tuple[int, ...]]]=None, keepdim=False) -> Tensor:
    axis_: Tuple[int, ...] = tuple(range(len(self.shape))) if axis is None else ((axis,) if isinstance(axis, int) else tuple(axis))
    axis_ = tuple(x if x >= 0 else x+len(self.shape) for x in axis_)
    shape = tuple(s for i,s in enumerate(self.shape) if i not in axis_)
    ret = fxn.apply(self, axis=axis_)
    return ret if keepdim else ret.reshape(shape=shape)

  def sum(self, axis=None, keepdim=False, acc_dtype:Optional[DType]=None):
    if acc_dtype is None: acc_dtype = least_upper_dtype(self.dtype, dtypes.uint) if dtypes.is_unsigned(self.dtype) else \
                                      least_upper_dtype(self.dtype, dtypes.int) if (dtypes.is_int(self.dtype) or self.dtype==dtypes.bool) else \
                                      least_upper_dtype(self.dtype, dtypes.float)
    # cast back to float16 or bfloat16 to match torch / jax behavior, but we use float for acc
    output_dtype = self.dtype if self.dtype in (dtypes.float16, dtypes.bfloat16) else acc_dtype
    return self.cast(acc_dtype)._reduce(mlops.Sum, axis, keepdim).cast(output_dtype)

  def max(self, axis=None, keepdim=False): return self._reduce(mlops.Max, axis, keepdim)
  def min(self, axis=None, keepdim=False): return -((-self).max(axis=axis, keepdim=keepdim))

  def mean(self, axis=None, keepdim=False):
    assert all_int(self.shape), "does not support symbolic shape"
    out = self.sum(axis=axis, keepdim=keepdim)
    return out.div(prod(self.shape) / prod(out.shape)) if 0 not in out.shape else out
  def var(self, axis=None, keepdim=False, correction=1):
    assert all_int(self.shape), "does not support symbolic shape"
    square_sum = ((self - self.mean(axis=axis, keepdim=True)).square()).sum(axis=axis, keepdim=keepdim)
    return square_sum.div(max(0, prod(self.shape)/prod(square_sum.shape)-correction))
  def std(self, axis=None, keepdim=False, correction=1): return self.var(axis, keepdim, correction).sqrt()

  def _softmax(self, axis):
    if len(self.shape) == 0:
      assert axis in [-1, 0], f"{axis=} out of range of [-1, 0]"
      axis = None
    m = self - self.max(axis=axis, keepdim=True)
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdim=True)

  def softmax(self, axis=-1):
    _, e, ss = self._softmax(axis)
    return e.div(ss)

  def log_softmax(self, axis=-1):
    m, _, ss = self._softmax(axis)
    return m - ss.log()

  def argmax(self, axis=None, keepdim=False):
    if axis is None:
      idx = (self == self.max(axis)) * Tensor.arange(prod(self.shape)-1,-1,-1, requires_grad=False, device=self.device).reshape(self.shape)
      return (prod(self.shape) - idx.max() - 1).cast(dtypes.default_int)
    axis = axis + len(self.shape) if axis < 0 else axis
    m = self == self.max(axis=axis, keepdim=True)
    idx = m * Tensor.arange(self.shape[axis]-1,-1,-1, requires_grad=False, device=self.device).reshape(self.shape[axis], *[1]*(self.ndim-axis-1))
    return (self.shape[axis]-idx.max(axis=axis, keepdim=keepdim)-1).cast(dtypes.default_int)
  def argmin(self, axis=None, keepdim=False): return (-self).argmax(axis=axis, keepdim=keepdim)

  @staticmethod
  def einsum(formula:str, *raw_xs) -> Tensor:
    xs:Tuple[Tensor] = argfix(*raw_xs)
    formula = formula.replace(" ", "")
    inputs_str, output = formula.split("->") if "->" in formula else (formula, sorted(formula))
    inputs = [x for x in cast(str,inputs_str).split(',')]
    assert len(xs) == len(inputs), f"number of inputs doesn't match number of operands in formula, expected {len(inputs)}, got {len(xs)}"

    # map the value of each letter in the formula
    letter_val = sorted(merge_dicts([{letter:dim for letter, dim in zip(letters, tensor.shape)} for letters, tensor in zip(inputs, xs)]).items())

    xs_:List[Tensor] = []
    lhs = [sorted(enumerate(s), key=lambda e:e[1]) for s in inputs]
    for x,(order,letters) in zip(xs, [list(zip(*l)) for l in lhs]):
      # permute to the sorted letter order, then reshape/expand to create dimensions for the missing letters
      xs_.append(x.permute(order).reshape([val if letter in letters else 1 for letter,val in letter_val]).expand([val for _,val in letter_val]))

    rhs_order, rhs_letters = tuple(zip(*sorted(enumerate(output), key=lambda e:e[1]))) or ([], [])
    # sum over all axes that's not in the output, then permute to the output order
    return reduce(lambda a,b:a*b, xs_).sum(axis=[axis for axis,(letter,_) in enumerate(letter_val) if letter not in rhs_letters]).permute(rhs_order)

  # ***** processing ops *****

  def _pool(self, k_:Tuple[sint, ...], stride:Union[Tuple[int, ...], int]=1, dilation:Union[Tuple[int, ...], int]=1) -> Tensor:
    assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
    assert all_int(self.shape) and all_int(k_), f"does not support symbolic {self.shape=}, {k_=}"
    s_, d_ = make_pair(stride, len(k_)), make_pair(dilation, len(k_))
    assert len(k_) == len(s_) == len(d_), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
    noop_, i_ = [None] * len(self.shape[:-len(k_)]), self.shape[-len(k_):]
    if any(k > s for k,s in zip(k_, s_)) or any(d != 1 for d in d_):
      o_ = [(i - d * (k-1) - 1)//s + 1 for i,d,k,s in zip(i_, d_, k_, s_)]
      # repeats such that we don't need padding
      xup = self.repeat([1]*len(noop_) + [math.ceil(k*(i+d) / i) for k,i,d in zip(k_, i_, d_)])
      # slice by dilation
      xup = xup.slice(noop_ + [(0,k*(i+d)) for k,i,d in zip(k_, i_, d_)]).reshape(noop_ + flatten((k,i+d) for k,i,d in zip(k_, i_, d_)))
      # handle stride
      xup = xup.slice(noop_ + flatten(((0,k), (0,o*s)) for k,o,s in zip(k_, o_, s_))).reshape(noop_ + flatten((k,o,s) for k,o,s in zip(k_, o_, s_)))
      xup = xup.slice(noop_ + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_, o_))).reshape(noop_ + flatten((k,o) for k,o in zip(k_, o_)))
      # permute to move reduce to the end
      return xup.permute(*range(len(noop_)), *[len(noop_)+i*2+1 for i in range(len(i_))], *[len(noop_)+i*2 for i in range(len(i_))])
    # TODO: once the shapetracker can optimize well, remove this alternative implementation. or not if the CPU implementation doesn't use ShapeTracker
    o_ = [(i+(s-k))//s for i,s,k in zip(i_, s_, k_)]
    xup = self.slice(noop_ + [(0,o*s) for o,s in zip(o_, s_)])
    xup = xup.reshape(noop_ + flatten(((o,s) for o,s in zip(o_, s_))))
    xup = xup.slice(noop_ + flatten(((0,o), (0,k)) for o,k in zip(o_, k_)))
    return xup.permute(*range(len(noop_)), *[len(noop_)+i*2 for i in range(len(i_))], *[len(noop_)+i*2+1 for i in range(len(i_))])

  # NOTE: these work for more than 2D
  def avg_pool2d(self, kernel_size=(2,2), stride=None, dilation=1): return self._pool(
        make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).mean(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))
  def max_pool2d(self, kernel_size=(2,2), stride=None, dilation=1): return self._pool(
        make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).max(axis=tuple(range(0-len(make_pair(kernel_size)), 0)))

  def conv_transpose2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding=0, output_padding=0) -> Tensor:
    HW, trailing = weight.shape[2:], list(range(3, len(weight.shape)+1))
    x, w = self, weight.unflatten(0, (groups, -1)).permute(0,2,1,*trailing).flip(trailing)
    stride = make_pair(stride, len(HW))
    if any(s>1 for s in stride):
      x = x.reshape(None, None, *flatten((k,1) for k in x.shape[2:]))
      x = x.pad((None, None, *flatten((None,(0,s-1)) for s in stride)))
      x = x.reshape(None, None, *[k*s for k,s in zip(x.shape[2::2], stride)])
      x = x.shrink((None, None, *[(0,k-(s-1)) for k,s in zip(x.shape[2:], stride)]))
    padding = flatten((((k-1)*d-p,(k-1)*d-p+op) for k,d,p,op in reversed(list(
      zip(HW, make_pair(dilation, len(HW)), make_pair(padding, len(HW)), make_pair(output_padding, len(HW)))))))
    return x.conv2d(w.flatten(end_dim=1), groups=groups, bias=bias, dilation=dilation, padding=padding)

  def conv2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding=0, acc_dtype:Optional[DType]=None) -> Tensor:
    (bs,cin_), (cout,cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
    assert groups*cin == cin_ and len(self.shape) == len(weight.shape), f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"  # noqa: E501
    if isinstance(padding, (tuple,list)): assert len(padding) == 2*len(HW) or len(padding) == len(HW), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {self.shape}"  # noqa: E501
    padding_ = [padding]*2*len(HW) if isinstance(padding, int) else (padding if len(padding) == 2*len(HW) else [p for p in padding for _ in range(2)][::-1])  # noqa: E501

    # conv2d is a pooling op (with padding)
    x = self.pad2d(padding_)._pool(HW, stride, dilation)   # (bs, groups*cin, oy, ox, H, W)
    rcout, oyx = cout//groups, x.shape[2:-len(HW)]
    if not all(x == 3 for x in HW) or stride != 1 or dilation != 1 or not WINO.value:
      # normal conv
      x = x.reshape(bs, groups, cin, 1, *oyx, *HW).expand(bs, groups, cin, rcout, *oyx, *HW).permute(0,1,3,*[4+i for i in range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])  # noqa: E501

      # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
      ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW)).sum([-1-i for i in range(1+len(oyx))], keepdim=True, acc_dtype=acc_dtype).reshape(bs, cout, *oyx)  # noqa: E501
      return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))

    HWI, HWO = (6,) * len(HW), (4,) * len(HW)  # F(4x4,3x3) winograd tiles
    winograd_G = [[1/4, 0, 0], [-1/6, -1/6, -1/6], [-1/6, 1/6, -1/6], [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]]
    winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
    winograd_At = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]] # applying At in pre-order doubles compile time

    # todo: stride == dilation
    # use padding to round up to 4x4 output tiles
    # (bs, cin_, tyx, HWI)
    d = self.pad2d(sum([[padding_[i*2], padding_[i*2+1] + (-(dim + sum(padding_[i * 2:(i + 1) * 2]) - 2) % 4)] for i, dim in enumerate(self.shape[-len(HW):])], []))._pool(HWI, HWO)  # noqa: E501
    # move HW to the front: # (HWI, bs, cin_, tyx)
    d = d.permute(*range(len(d.shape)-len(HW),len(d.shape)), *range(len(d.shape)-len(HW)))
    tyx = d.shape[-len(HWI):]  # dim of tiling

    g = weight.permute(*range(len(weight.shape)-len(HW),len(weight.shape)), *range(len(weight.shape)-len(HW)))  # move HW to the front

    # compute 6x6 winograd tiles: GgGt, BtdB
    # (HWI, groups * rcout, cin) -> (HWI, bs=1, groups, rcout, cin, tyx=(1,1))
    gfactors = _apply_winograd_matrix(winograd_G, g, len(HW)).reshape(*HWI, 1, groups, rcout, cin, *([1]*len(tyx)))
    # (HWI, bs, cin_, tyx) -> (HWI, bs, groups, 1 ,cin, *tyx)
    dfactors = _apply_winograd_matrix(winograd_Bt, d, len(HW)).reshape(*HWI, bs, groups, 1, cin, *tyx)

    # matmul; sum across cin: (HWI, bs, groups, rcout, *tyx); then HWI -> HWO: (HWO, bs, groups, rcout, *tyx)
    ret = _apply_winograd_matrix(winograd_At, (gfactors * dfactors).sum(axis=-1-len(HW), acc_dtype=acc_dtype), len(HW))

    # interleave tyx and HWO: (bs, groups, rcout, oy, HO, ox, WO)
    ret = ret.permute([*range(len(HW), len(ret.shape)-len(HW)), *[i+o for i in range(len(HW)) for o in [len(ret.shape)-len(HW),0]]])
    # merge groups and rcout, tyx and HWO: (bs, groups, cout, *yx), shrink to final
    ret = ret.reshape(bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]).shrink(tuple((0, s) for s in [bs, cout, *oyx]))

    return (ret if bias is None else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))).contiguous().contiguous_backward()

  def dot(self, w:Tensor, acc_dtype:Optional[DType]=None) -> Tensor:
    n1, n2 = len(self.shape), len(w.shape)
    assert n1 != 0 and n2 != 0, f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
    assert (L:=self.shape[-1]) == (R:=w.shape[-min(n2, 2)]), f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({L} != {R})"
    x = self.reshape(*self.shape[0:-1], *[1]*min(n1-1, n2-1, 1), self.shape[-1])
    w = w.reshape(*w.shape[0:-2], *[1]*min(n1-1, n2-1, 1), *w.shape[-min(n2, 2):]).transpose(-1, -min(n2, 2))
    return (x*w).sum(-1, acc_dtype=acc_dtype).cast(least_upper_dtype(x.dtype, w.dtype))

  def matmul(self, x:Tensor, reverse=False, acc_dtype:Optional[DType]=None) -> Tensor:
    return x.dot(self, acc_dtype=acc_dtype) if reverse else self.dot(x, acc_dtype=acc_dtype)

  def _cumsum(self, axis:int=0, _first_zero=False) -> Tensor:
    return self.transpose(axis,-1).pad2d((self.shape[axis]-int(not _first_zero),0))._pool((self.shape[axis],)).sum(-1).transpose(axis,-1)
  def cumsum(self, axis:int=0) -> Tensor:
    # TODO: someday the optimizer will find this on it's own
    # for now this is a two stage cumsum
    SPLIT = 256
    if self.shape[axis] <= SPLIT*2: return self._cumsum(axis)
    ret = self.transpose(axis,-1).pad2d((round_up(self.shape[axis], SPLIT)-self.shape[axis], 0))
    ret = ret.unflatten(-1, (-1, SPLIT))._cumsum(-1)
    base_add = ret[..., -1]._cumsum(-1, _first_zero=True)[..., :-1]
    base_add = base_add.unsqueeze(-1).expand(*base_add.shape, ret.shape[-1])
    def fix(x:Tensor): return x.flatten(start_dim=-2)[..., -self.shape[axis]:].transpose(axis,-1)
    return fix(ret) + fix(base_add)

  @staticmethod
  def _tri(r:sint, c:sint, k:int=0, **kwargs) -> Tensor:
    assert all_int((r,c)), "does not support symbolic"
    if r == 0: return Tensor.zeros((r, c), **kwargs)
    return Tensor.arange(r, **kwargs).unsqueeze(1).expand(r,c) <= Tensor.arange(-k, c-k, **kwargs).unsqueeze(0).expand(r,c)
  def triu(self, k:int=0) -> Tensor: return Tensor._tri(self.shape[-2], self.shape[-1], k=k, device=self.device).where(self, 0)
  def tril(self, k:int=0) -> Tensor: return Tensor._tri(self.shape[-2], self.shape[-1], k=k+1, device=self.device).where(0, self)

  # ***** mlops (unary) *****

  def logical_not(self): return mlops.Eq.apply(*self._broadcasted(False))
  def neg(self): return mlops.Neg.apply(self) if self.dtype != dtypes.bool else self.logical_not()
  def contiguous(self): return mlops.Contiguous.apply(self)
  def contiguous_backward(self): return mlops.ContiguousBackward.apply(self)
  def log(self): return mlops.Log.apply(self.cast(least_upper_float(self.dtype)))
  def log2(self): return self.log()/math.log(2)
  def exp(self): return mlops.Exp.apply(self.cast(least_upper_float(self.dtype)))
  def exp2(self): return mlops.Exp.apply(self*math.log(2))
  def relu(self): return mlops.Relu.apply(self)
  def sigmoid(self): return mlops.Sigmoid.apply(self.cast(least_upper_float(self.dtype)))
  def sin(self): return mlops.Sin.apply(self.cast(least_upper_float(self.dtype)))
  def sqrt(self): return mlops.Sqrt.apply(self.cast(least_upper_float(self.dtype)))
  def rsqrt(self): return self.reciprocal().sqrt()
  def cos(self): return ((math.pi/2)-self).sin()
  def tan(self): return self.sin() / self.cos()

  # ***** math functions (unary) *****

  def trunc(self: Tensor) -> Tensor: return self.cast(dtypes.int32).cast(self.dtype)
  def ceil(self: Tensor) -> Tensor: return (self > (b := self.trunc())).where(b+1, b)
  def floor(self: Tensor) -> Tensor: return (self < (b := self.trunc())).where(b-1, b)
  def round(self: Tensor) -> Tensor:
    return ((self > 0) == ((b := self.cast(dtypes.int32) / 2.0).cast(dtypes.int32) == b)).where((self - 0.5).ceil(), (self + 0.5).floor())

  def square(self): return self*self
  def clip(self, min_, max_): return self.maximum(min_).minimum(max_)
  def abs(self): return self.relu() + (-self).relu()
  def sign(self): return ((self.float()) / (self.float().abs() + 1e-12)).cast(self.dtype)
  def reciprocal(self): return 1.0/self

  # ***** activation functions (unary) *****

  def elu(self, alpha=1.0): return self.relu() - alpha*(1-self.exp()).relu()
  def celu(self, alpha=1.0): return self.maximum(0) + (alpha * ((self / alpha).exp() - 1)).minimum(0)
  def swish(self): return self * self.sigmoid()
  def silu(self): return self.swish()   # The SiLU function is also known as the swish function.
  def relu6(self): return self.relu() - (self-6).relu()
  def hardswish(self): return self * (self+3).relu6() * (1/6)
  def tanh(self): return 2.0 * ((2.0 * self).sigmoid()) - 1.0
  def sinh(self): return (self.exp() - self.neg().exp()) / 2
  def cosh(self): return (self.exp() + self.neg().exp()) / 2
  def atanh(self): return ((1 + self)/(1 - self)).log() / 2
  def asinh(self): return (self + (self.square() + 1).sqrt()).log()
  def acosh(self): return (self + (self.square() - 1).sqrt()).log()
  def hardtanh(self, min_val=-1, max_val=1): return self.clip(min_val, max_val)
  def gelu(self): return 0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())
  def quick_gelu(self): return self * (self * 1.702).sigmoid()
  def leakyrelu(self, neg_slope=0.01): return self.relu() - (-neg_slope*self).relu()
  def mish(self): return self * self.softplus().tanh()
  def softplus(self, beta=1): return (1/beta) * (1 + (self*beta).exp()).log()
  def softsign(self): return self / (1 + self.abs())

  # ***** broadcasted elementwise mlops *****

  def _broadcasted(self, y:Union[Tensor, Scalar], reverse:bool=False, match_dtype:bool=True) -> Tuple[Tensor, Tensor]:
    x: Tensor = self
    if not isinstance(y, Tensor):
      # make y a Tensor
      assert isinstance(y, (float, int, bool)), f"{type(y)=}, {y=}"
      if isinstance(self.dtype, ImageDType) or dtypes.is_float(x.dtype) or (dtypes.is_int(x.dtype) and isinstance(y, int)): y_dtype = x.dtype
      else: y_dtype = dtypes.from_py(y)
      y = Tensor(cast_scalar(y, y_dtype), self.device, y_dtype, requires_grad=False)

    if match_dtype:
      output_dtype = least_upper_dtype(x.dtype, y.dtype)
      x, y = x.cast(output_dtype), y.cast(output_dtype)

    if reverse: x, y = y, x

    # left pad shape with 1s
    if len(y.shape) < len(x.shape): y = y.reshape((1,) * (len(x.shape) - len(y.shape)) + y.shape)
    elif len(x.shape) < len(y.shape): x = x.reshape((1,) * (len(y.shape) - len(x.shape)) + x.shape)

    broadcasted_shape = tuple(0 if xi==0 or yi==0 else max(xi, yi) for xi, yi in zip(x.shape, y.shape))
    return x.expand(broadcasted_shape), y.expand(broadcasted_shape)

  def _to_const_val(self, x:Union[Tensor, Scalar]) -> Union[Tensor, Scalar]:
    # TODO: update with multi
    return x.lazydata.base.arg if isinstance(x, Tensor) and isinstance(x.lazydata, LazyBuffer) and x.lazydata.is_unrealized_contiguous_const() \
      and not x.requires_grad and self._broadcasted(x)[0].shape == self.shape else x

  def add(self, x:Union[Tensor, Scalar], reverse=False) -> Tensor:
    x = self._to_const_val(x)
    return mlops.Add.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x else self
  def sub(self, x:Union[Tensor, Scalar], reverse=False) -> Tensor:
    x = self._to_const_val(x)
    return mlops.Sub.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x else (-self if reverse else self)
  def mul(self, x:Union[Tensor, Scalar], reverse=False) -> Tensor:
    x = self._to_const_val(x)
    if x.__class__ is not Tensor and x == 0.0: return mlops.Zero.apply(self)
    if x.__class__ is not Tensor and x == -1.0: return -self
    return mlops.Mul.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x != 1.0 else self
  def div(self, x:Union[Tensor, Scalar], reverse=False) -> Tensor:
    x = self._to_const_val(x)
    if x.__class__ is not Tensor and not reverse and x != 0: return self.mul(1/x)
    if isinstance(x, Tensor) and dtypes.is_float(x.dtype): return mlops.Div.apply(*self._broadcasted(x, reverse))
    return mlops.Div.apply(*self.cast(least_upper_float(self.dtype))._broadcasted(x, reverse))
  def xor(self, x:Tensor, reverse=False) -> Tensor: return mlops.Xor.apply(*self._broadcasted(x, reverse))

  def pow(self, x:Union[Tensor, Scalar], reverse=False) -> Tensor:
    x = self._to_const_val(x)
    if not isinstance(x, Tensor) and not reverse:
      # simple pow identities
      if x < 0: return self.reciprocal().pow(-x)
      if x in [3,2,1,0]: return reduce(lambda acc,_: acc * self, range(int(x)), mlops.Zero.apply(self)+1)
      if x == 0.5: return self.sqrt()
    if not isinstance(x, Tensor) and reverse and x > 0: return self.mul(math.log(x)).exp()
    ar = self.abs().log().mul(x).exp() if not reverse or isinstance(x, Tensor) else self.mul(math.log(abs(x))).exp()
    # correct sign of negative numbers raised to a power (cos has a period of 2pi so we use it here to get the oddness of the power)
    sign = (x * math.pi).cos() if isinstance(x, Tensor) else math.cos(x * math.pi) if not reverse else (self * math.pi).cos()
    # we only need to correct the sign if the base is negative
    base_sign = ((self.sign() if not reverse else x.sign() if isinstance(x, Tensor) else math.copysign(1, x)) - 1) / -2
    # we need 0 to be positive so we need to correct base_sign when the base is 0
    base_sign = base_sign - (1.5 * (1 - (self.sign().abs() if not reverse else x.sign().abs() if isinstance(x, Tensor) else abs(int(bool(x))))))
    # inject nan if the base is negative and the power is not an integer
    to_nan = (((x - x.trunc()) * 1e10).abs().clip(0, 1) if isinstance(x, Tensor) else \
              int(bool(x - int(x))) if not reverse else ((self - self.trunc()) * 1e10).abs().clip(0, 1)) * base_sign
    inject_nan = ((((-to_nan) * 2) + 1)).log().add(1) if isinstance(to_nan, Tensor) else 1 if not to_nan else float("nan")
    return ar.mul(sign * base_sign + (1 - base_sign)).mul(inject_nan)

  def maximum(self, x:Union[Tensor, Scalar]) -> Tensor:
    return (self<x).detach().where(x, (self==x).detach().where(((self * 0.5 + x * 0.5).cast(self.dtype)), self))
  def minimum(self, x:Union[Tensor, Scalar]) -> Tensor: return -((-self).maximum(-x))

  def where(self:Tensor, input_:Union[Tensor, Scalar], other:Union[Tensor, Scalar]):
    if isinstance(input_, Tensor): input_, other = input_._broadcasted(other)
    elif isinstance(other, Tensor): other, input_ = other._broadcasted(input_)
    x_,y = self._broadcasted(input_, match_dtype=False)
    x,z = x_._broadcasted(other, match_dtype=False)
    return mlops.Where.apply(x.cast(dtypes.bool), *y._broadcasted(z))

  # ***** op wrappers (wasted lines to make the typechecker happy) *****

  def __neg__(self) -> Tensor: return self.neg()

  def __add__(self, x) -> Tensor: return self.add(x)
  def __sub__(self, x) -> Tensor: return self.sub(x)
  def __mul__(self, x) -> Tensor: return self.mul(x)
  def __pow__(self, x) -> Tensor: return self.pow(x)
  def __truediv__(self, x) -> Tensor: return self.div(x)
  def __matmul__(self, x) -> Tensor: return self.matmul(x)
  def __xor__(self, x) -> Tensor: return self.xor(x)

  def __radd__(self, x) -> Tensor: return self.add(x, True)
  def __rsub__(self, x) -> Tensor: return self.sub(x, True)
  def __rmul__(self, x) -> Tensor: return self.mul(x, True)
  def __rpow__(self, x) -> Tensor: return self.pow(x, True)
  def __rtruediv__(self, x) -> Tensor: return self.div(x, True)
  def __rmatmul__(self, x) -> Tensor: return self.matmul(x, True)
  def __rxor__(self, x) -> Tensor: return self.xor(x, True)

  def __iadd__(self, x) -> Tensor: return self.assign(self.add(x))
  def __isub__(self, x) -> Tensor: return self.assign(self.sub(x))
  def __imul__(self, x) -> Tensor: return self.assign(self.mul(x))
  def __ipow__(self, x) -> Tensor: return self.assign(self.pow(x))
  def __itruediv__(self, x) -> Tensor: return self.assign(self.div(x))
  def __imatmul__(self, x) -> Tensor: return self.assign(self.matmul(x))
  def __ixor__(self, x) -> Tensor: return self.assign(self.xor(x))

  def __lt__(self, x) -> Tensor: return mlops.Less.apply(*self._broadcasted(x, False))
  def __gt__(self, x) -> Tensor: return mlops.Less.apply(*self._broadcasted(x, True))
  def __ge__(self, x) -> Tensor: return (self<x).logical_not()
  def __le__(self, x) -> Tensor: return (self>x).logical_not()
  def __eq__(self, x) -> Tensor: return mlops.Eq.apply(*self._broadcasted(x, True))   # type: ignore[override]
  def __ne__(self, x) -> Tensor: return (self==x).logical_not()                       # type: ignore[override]

  # ***** functional nn ops *****

  def linear(self, weight:Tensor, bias:Optional[Tensor]=None):
    x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
    return x.add(bias) if bias is not None else x

  def sequential(self, ll:List[Callable[[Tensor], Tensor]]): return reduce(lambda x,f: f(x), ll, self)

  def layernorm(self, axis=-1, eps:float=1e-5) -> Tensor:
    y = (self - self.mean(axis, keepdim=True))
    return y.mul((y*y).mean(axis, keepdim=True).add(eps).rsqrt())

  def batchnorm(self, weight:Optional[Tensor], bias:Optional[Tensor], mean:Tensor, invstd:Tensor, axis:Union[int,Tuple[int,...]]=1) -> Tensor:
    axis_ = argfix(axis)
    shape = tuple(s if ax in axis_ else 1 for ax, s in enumerate(self.shape))
    x = self - mean.reshape(shape)
    if weight: x = x * weight.reshape(shape)
    ret = x.mul(invstd.reshape(shape) if len(invstd.shape) == len(axis_) else invstd)
    return (ret + bias.reshape(shape)) if bias else ret

  def dropout(self, p=0.5) -> Tensor:
    if not Tensor.training or p == 0: return self
    return self * (Tensor.rand(*self.shape, requires_grad=False, device=self.device) >= p) * (1/(1.0 - p))

  def one_hot(self, num_classes:int) -> Tensor:
    return (self[..., None] == Tensor.arange(num_classes, requires_grad=False, device=self.device)).where(1, 0)

  def scaled_dot_product_attention(self, key:Tensor, value:Tensor, attn_mask:Optional[Tensor]=None,
                                   dropout_p:float=0.0, is_causal:bool=False) -> Tensor:
    # NOTE: it works if key, value have symbolic shape
    assert all_int(self.shape), f"does not support symbolic shape {self.shape}"
    if is_causal: attn_mask = Tensor.ones(self.shape[-2], key.shape[-2], requires_grad=False, device=self.device).tril(0).cast(dtypes.bool)
    if attn_mask is not None and attn_mask.dtype == dtypes.bool: attn_mask = (attn_mask == 0).where(-float("inf"), 0)
    qk = self @ key.transpose(-2,-1) / math.sqrt(self.shape[-1])
    return ((qk+attn_mask) if attn_mask is not None else qk).softmax(-1).dropout(dropout_p) @ value

  def binary_crossentropy(self, y:Tensor) -> Tensor:
    return (-y*self.log() - (1-y)*(1-self).log()).mean()

  def binary_crossentropy_logits(self, y:Tensor) -> Tensor:
    return (self.maximum(0) - y * self + (1 + self.abs().neg().exp()).log()).mean()

  def sparse_categorical_crossentropy(self, Y:Tensor, ignore_index=-1, label_smoothing=0.0) -> Tensor:
    assert 0.0 <= label_smoothing <= 1.0, "label_smoothing must be in [0.0, 1.0]"
    # NOTE: self is a logits input
    log_probs, loss_mask = self.log_softmax(), (Y != ignore_index)
    y_counter = Tensor.arange(self.shape[-1], requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    smoothing = -1 * label_smoothing * (log_probs.mean(-1) * loss_mask).sum() / loss_mask.sum()
    return (1 - label_smoothing) * (log_probs * y).sum() / loss_mask.sum() + smoothing

  # ***** cast ops *****

  def cast(self, dtype:DType) -> Tensor:
    if self.dtype == dtype: return self
    # hack for devices that don't support bfloat16
    if self.dtype == dtypes.bfloat16: return self.bitcast(dtypes.uint16).cast(dtypes.uint32).mul(1<<16).bitcast(dtypes.float32).cast(dtype)
    return mlops.Cast.apply(self, dtype=dtype)
  def bitcast(self, dtype:DType) -> Tensor:
    assert self.dtype.itemsize == dtype.itemsize, "can't bitcast mismatched dtype itemsizes"
    return mlops.Cast.apply(self, dtype=dtype, bitcast=True) if self.dtype != dtype else self
  def float(self) -> Tensor: return self.cast(dtypes.float32)
  def half(self) -> Tensor: return self.cast(dtypes.float16)

  # ***** convenience stuff *****

  @property
  def ndim(self) -> int: return len(self.shape)
  def numel(self) -> sint: return prod(self.shape)
  def element_size(self) -> int: return self.dtype.itemsize
  def nbytes(self) -> int: return self.numel() * self.element_size()
  def is_floating_point(self) -> bool: return dtypes.is_float(self.dtype)

# register functions to move between devices
for device in Device._devices: setattr(Tensor, f"{device.lower()}", partialmethod(Tensor.to, device))

if IMAGE:
  # if IMAGE>0 we install these replacement functions in Tensor (hack!)
  from tinygrad.features.image import image_conv2d, image_dot
  setattr(Tensor, "conv2d", image_conv2d)
  setattr(Tensor, "dot", image_dot)

# TODO: remove the custom op and replace with threefry
def custom_random(out:Buffer):
  Tensor._seed += 1
  if DEBUG >= 2: print(f"*** {out.device}   rand  seed {Tensor._seed} size {out.size:<15d} dtype {out.dtype}")
  rng = np.random.default_rng(Tensor._seed)
  rng_np_buffer = rng.random(size=out.size, dtype=np.float32).astype(dtype=out.dtype.np, copy=False)
  out.copyin(rng_np_buffer.data)
