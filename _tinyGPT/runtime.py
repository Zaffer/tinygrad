# this is a single file representing each of the modules in the
# /runtime directory of the tinygrad project


# tinygrad/runtime/ops_clang.py

import ctypes, subprocess, pathlib, tempfile
from tinygrad.device import Compiled, MallocAllocator, Compiler
from tinygrad.helpers import cpu_time_execution
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import uops_to_cstyle, CStyleLanguage

CLANG_PROGRAM_HEADER = '#include <math.h>\n#define max(x,y) ((x>y)?x:y)\n#define int64 long\n#define half __fp16\n#define uchar unsigned char\n#include <stdbool.h>\n'  # noqa: E501

class ClangCompiler(Compiler):
  linearizer_opts = LinearizerOptions("CLANG", supports_float4=False, has_local=False)
  def render(self, name:str, uops) -> str: return uops_to_cstyle(CStyleLanguage(buffer_suffix=" restrict"), name, uops)
  def compile(self, src:str) -> bytes:
    # TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    with tempfile.NamedTemporaryFile(delete=True) as output_file:
      subprocess.check_output(args=('clang -shared -march=native -O2 -Wall -Werror -x c -fPIC - -o '+ \
                                    str(output_file.name)).split(), input=(CLANG_PROGRAM_HEADER+src).encode('utf-8'))
      return pathlib.Path(output_file.name).read_bytes()

class ClangProgram:
  def __init__(self, name:str, lib:bytes):
    self.name, self.lib = name, lib
    # write to disk so we can load it
    with tempfile.NamedTemporaryFile(delete=True) as cached_file_path:
      pathlib.Path(cached_file_path.name).write_bytes(lib)
      self.fxn = ctypes.CDLL(str(cached_file_path.name))[name]

  def __call__(self, *bufs, vals=(), wait=False): return cpu_time_execution(lambda: self.fxn(*bufs, *vals), enable=wait)

class ClangDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, MallocAllocator, ClangCompiler("compile_clang"), ClangProgram)


# tinygrad/runtime/ops_cuda.py
  
from __future__ import annotations
import subprocess, hashlib, tempfile, ctypes, ctypes.util, functools, re
from pathlib import Path
from typing import Tuple, Optional
import tinygrad.runtime.autogen.cuda as cuda
from tinygrad.helpers import DEBUG, getenv, from_mv, to_char_p_p, init_c_var, colored, cpu_time_execution, encode_args_cuda_style, time_execution_cuda_style # noqa: E501
from tinygrad.device import Compiled, LRUAllocator, MallocAllocator, Compiler
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import CUDARenderer

def pretty_ptx(s):
  # all expressions match `<valid_before><expr><valid_after>` and replace it with `<valid_before>color(<expr>)<valid_after>`
  s = re.sub(r'([!@<\[\s,\+\-;\n])((?:[_%$][\w%\$_]+(?:\.[xyz])?\:?)|(?:buf\d+))([<>\]\s,\+\-;\n\)])', lambda m:m[1]+colored(m[2], "blue")+m[3], s, flags=re.M) # identifiers  # noqa: E501
  s = re.sub(r'(.)((?:b|s|u|f)(?:8|16|32|64)|pred)([\.\s])', lambda m:m[1]+colored(m[2], "green")+m[3], s, flags=re.M) # types
  s = re.sub(r'^(\s*)([\w]+)(.*?;$)', lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # instructions
  s = re.sub(r'([<>\[\]\s,\+\-;])((?:0[fF][0-9a-fA-F]{8})|(?:[0-9]+)|(?:0[xX][0-9a-fA-F]+))([<>\[\]\s,\+\-;])', lambda m:m[1]+colored(m[2], "yellow")+m[3], s, flags=re.M) # numbers  # noqa: E501
  s = re.sub(r'(\.)(param|reg|global)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # space
  s = re.sub(r'(\.)(version|target|address_size|visible|entry)', lambda m:m[1]+colored(m[2], "magenta"), s, flags=re.M) # derivatives
  return s

CUDACPU = getenv("CUDACPU") == 1
if CUDACPU:
  gpuocelot_lib = ctypes.CDLL(ctypes.util.find_library("gpuocelot"))
  gpuocelot_lib.ptx_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]  # noqa: E501
  cuda.cuLaunchKernel = lambda src, gx, gy, gz, lx, ly, lz, shared, stream, unused_extra, args: gpuocelot_lib.ptx_run(src, len(args), (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]), lx, ly, lz, gx, gy, gz, shared)  # type: ignore  # noqa: E501

def check(status):
  if status != 0: raise RuntimeError(f"CUDA Error {status}, {ctypes.string_at(init_c_var(ctypes.POINTER(ctypes.c_char)(), lambda x: cuda.cuGetErrorString(status, ctypes.byref(x)))).decode()}")  # noqa: E501

def cu_time_execution(cb, enable=False) -> Optional[float]: return time_execution_cuda_style(cb, cuda.CUevent, cuda.cuEventCreate, cuda.cuEventRecord, cuda.cuEventSynchronize, cuda.cuEventDestroy_v2, cuda.cuEventElapsedTime, enable=enable) if not CUDACPU else cpu_time_execution(cb, enable=enable)  # noqa: E501

def _get_bytes(arg, get_str, get_sz, check) -> bytes:
  sz = init_c_var(ctypes.c_size_t(), lambda x: check(get_sz(arg, ctypes.byref(x))))
  return ctypes.string_at(init_c_var(ctypes.create_string_buffer(sz.value), lambda x: check(get_str(arg, x))), size=sz.value)

class CUDACompiler(Compiler):
  linearizer_opts = LinearizerOptions("CUDA", global_max=[65535, 65535, 2147483647], local_max=[64, 1024, 1024])
  def __init__(self, arch:str):
    self.arch = arch
    CUDACompiler.linearizer_opts = CUDACompiler.linearizer_opts._replace(has_tensor_cores=int(arch[3:]) >= 80)
    super().__init__(f"compile_cuda_{self.arch}")
  def render(self, name:str, uops) -> str: return CUDARenderer(name, uops)
  def compile(self, src:str) -> bytes:
    compile_options = [f'--gpu-architecture={self.arch}', "-I/usr/local/cuda/include", "-I/usr/include", "-I/opt/cuda/include/"]
    check(cuda.nvrtcCreateProgram(ctypes.byref(prog := cuda.nvrtcProgram()), src.encode(), "<null>".encode(), 0, None, None))
    status = cuda.nvrtcCompileProgram(prog, len(compile_options), to_char_p_p([o.encode() for o in compile_options]))

    if status != 0: raise RuntimeError(f"compile failed: {_get_bytes(prog, cuda.nvrtcGetProgramLog, cuda.nvrtcGetProgramLogSize, check).decode()}")
    return _get_bytes(prog, cuda.nvrtcGetPTX, cuda.nvrtcGetPTXSize, check)

class CUDAProgram:
  def __init__(self, device:CUDADevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    if DEBUG >= 5: print("\n".join([f"{i+1:>3} {line}" for i, line in enumerate(pretty_ptx(lib.decode('utf-8')).split("\n"))]))
    if DEBUG >= 6:
      try:
        fn = (Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(lib).hexdigest()}").as_posix()
        with open(fn + ".ptx", "wb") as f: f.write(lib)
        subprocess.run(["ptxas", f"-arch={device.arch}", "-o", fn, fn+".ptx"], check=True)
        print(subprocess.check_output(['nvdisasm', fn]).decode('utf-8'))
      except Exception as e: print("failed to generate SASS", str(e))

    if not CUDACPU:
      check(cuda.cuCtxSetCurrent(self.device.context))
      self.module = init_c_var(cuda.CUmodule(), lambda x: check(cuda.cuModuleLoadData(ctypes.byref(x), lib)))
      check(cuda.cuModuleGetFunction(ctypes.byref(prg := cuda.CUfunction()), self.module, name.encode("utf-8")))
    self.prg = prg if not CUDACPU else lib

  def __del__(self):
    if hasattr(self, 'module'): check(cuda.cuModuleUnload(self.module))

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if not CUDACPU: check(cuda.cuCtxSetCurrent(self.device.context))
    c_kernel_input_config = encode_args_cuda_style(bufs, vals, cuda.CUdeviceptr_v2, (1,2,0))[0] if not CUDACPU else (bufs+tuple(vals))
    return cu_time_execution(lambda: check(cuda.cuLaunchKernel(self.prg, *global_size, *local_size, 0, None, None, c_kernel_input_config)), enable=wait)  # noqa: E501

class CUDAAllocator(LRUAllocator):
  def __init__(self, device:CUDADevice):
    self.device = device
    super().__init__()
  def _alloc(self, size):
    check(cuda.cuCtxSetCurrent(self.device.context))
    return init_c_var(cuda.CUdeviceptr(), lambda x: check(cuda.cuMemAlloc_v2(ctypes.byref(x), size)))
  def _free(self, opaque): check(cuda.cuMemFree_v2(opaque))
  def copyin(self, dest, src:memoryview):
    check(cuda.cuCtxSetCurrent(self.device.context))
    check(cuda.cuMemcpyHtoD_v2(dest, from_mv(src), len(src), None))
  def copyout(self, dest:memoryview, src):
    check(cuda.cuCtxSetCurrent(self.device.context))
    check(cuda.cuMemcpyDtoH_v2(from_mv(dest), src, len(dest)))

class CUDADevice(Compiled):
  def __init__(self, device:str):
    device_id = int(device.split(":")[1]) if ":" in device else 0
    if not CUDACPU:
      check(cuda.cuInit(0))
      check(cuda.cuDeviceGet(ctypes.byref(cu_device := cuda.CUdevice()), device_id))
      self.context = init_c_var(cuda.CUcontext(), lambda x: check(cuda.cuCtxCreate_v2(ctypes.byref(x), 0, cu_device)))
      check(cuda.cuDeviceComputeCapability(ctypes.byref(major := ctypes.c_int()), ctypes.byref(minor := ctypes.c_int()), device_id))
    self.arch = f"sm_{major.value}{minor.value}" if not CUDACPU else "sm_35"

    from tinygrad.runtime.graph.cuda import CUDAGraph
    super().__init__(device, CUDAAllocator(self) if not CUDACPU else MallocAllocator, CUDACompiler(self.arch),
                     functools.partial(CUDAProgram, self), graph=CUDAGraph if not CUDACPU else None)
  def synchronize(self):
    if not CUDACPU:
      check(cuda.cuCtxSetCurrent(self.context))
      check(cuda.cuCtxSynchronize())


# tinygrad/runtime/ops_disk.py
      
import os, mmap, _posixshmem, io, functools
from typing import Dict, List, Any
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import prod, OSX
from tinygrad.device import Compiled, Allocator, JITRunner, Buffer
from tinygrad.ops import UnaryOps, LazyOp, BufferOps
from tinygrad.shape.view import strides_for_shape

class UnderlyingDiskBuffer:
  def __init__(self, fd, mem): self.fd, self.mem = fd, mem
  def __del__(self):
    if self.fd is not None: os.close(self.fd)

class DiskBuffer:
  def __init__(self, ud:UnderlyingDiskBuffer, size:int, dtype:DType=dtypes.uint8, offset=0):
    self.ud, self.size, self.dtype, self.offset = ud, size, dtype, offset
  def __repr__(self): return f"<DiskBuffer size={self.size} dtype={self.dtype} offset={self.offset}>"
  def _buf(self) -> memoryview: return memoryview(self.ud.mem)[self.offset:self.offset+self.size*self.dtype.itemsize]

MAP_LOCKED, MAP_POPULATE = 0 if OSX else 0x2000, getattr(mmap, "MAP_POPULATE", 0 if OSX else 0x008000)
class DiskAllocator(Allocator):
  def __init__(self, device:str): self.device = device
  def _alloc(self, size:int):
    if self.device.startswith("shm:"):
      fd = _posixshmem.shm_open("/"+self.device[4:].lstrip("/"), os.O_RDWR, 0o600)
      mem = mmap.mmap(fd, size, mmap.MAP_SHARED | MAP_POPULATE | MAP_LOCKED)
      os.close(fd)
      fd = None
    else:
      try: fd = os.open(self.device, os.O_RDWR|os.O_CREAT|(0 if OSX else os.O_DIRECT))
      except OSError: fd = os.open(self.device, os.O_RDWR|os.O_CREAT)
      if os.fstat(fd).st_size < size: os.ftruncate(fd, size)
      mem = mmap.mmap(fd, size)
    if (hp := getattr(mmap, "MADV_HUGEPAGE", None)) is not None: mem.madvise(hp) # type: ignore
    return DiskBuffer(UnderlyingDiskBuffer(fd, mem), size)
  def as_buffer(self, src:DiskBuffer): return src._buf()
  def copyin(self, dest:DiskBuffer, src:memoryview): dest._buf()[:] = src
  def copyout(self, dest:memoryview, src:DiskBuffer):
    if OSX and src.ud.fd is not None:
      # OSX doesn't seem great at mmap, this is faster
      with io.FileIO(src.ud.fd, "a+b", closefd=False) as fo:
        fo.seek(src.offset)
        fo.readinto(dest)
    else:
      dest[:] = src._buf()

class DiskRunner(JITRunner):
  skip_allocation = True
  def __init__(self, ast:LazyOp):
    # two ASTs are allowed here.
    assert ast.op == BufferOps.STORE, "output of AST must be store"
    assert ast.arg.st.contiguous, "shapetracker must be contiguous"
    # TODO: there shouldn't actually be casts here, bitcasts should fold into the load
    if ast.src[0].op == UnaryOps.CAST:
      top_src = ast.src[0].src[0]
      # TODO: assert that this is bitcast
      self.new_dtype = ast.src[0].arg[0]
    else:
      top_src = ast.src[0]
      self.new_dtype = top_src.arg.dtype
    assert top_src.op == BufferOps.LOAD, "top of AST must be load"
    assert len(top_src.arg.st.views) == 1, "shapetracker must have 1 view"
    view = top_src.arg.st.views[0]
    assert view.mask is None, "view cannot have a mask"
    assert strides_for_shape(view.shape) == view.strides, "disk tensors don't support strides"
    self.new_size = prod(view.shape)
    self.new_offset = view.offset * top_src.arg.dtype.itemsize
  def __call__(self, rawbufs:List[Buffer], var_vals:Dict[Any, int], wait=False, jit=False):
    assert len(rawbufs) == 2
    src = rawbufs[1]._buf
    rawbufs[0]._buf = DiskBuffer(src.ud, self.new_size, self.new_dtype, offset=src.offset+self.new_offset)

class DiskDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, DiskAllocator(device[len("disk:"):]), None, None)
  @functools.lru_cache(None)    # pylint: disable=method-cache-max-size-none
  def get_runner(self, ast:LazyOp): return DiskRunner(ast)


# tinygrad/runtime/ops_ext.py
  
from __future__ import annotations
from typing import Tuple, Optional, List, cast
import ctypes, functools, hashlib
import tinygrad.runtime.autogen.opencl as cl
from tinygrad.helpers import init_c_var, to_char_p_p, from_mv, OSX, DEBUG
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import OpenCLRenderer
from tinygrad.device import Compiled, LRUAllocator, BufferOptions, Compiler

# see test/external/external_osx_profiling.py to determine this ratio. it's in like GPU clocks or something
OSX_TIMING_RATIO = (125/3) if OSX else 1.0

def check(status):
  if status != 0: raise RuntimeError(f"OpenCL Error {status}")
def checked(ret, status): return (check(status.value), ret)[1]

class CLCompiler(Compiler):
  linearizer_opts = LinearizerOptions("GPU")
  def __init__(self, device:CLDevice, compile_key:str):
    self.device = device
    super().__init__(f"compile_cl_{compile_key}")
  def render(self, name:str, uops) -> str: return OpenCLRenderer(name, uops)
  def compile(self, src:str) -> bytes:
    program = checked(cl.clCreateProgramWithSource(self.device.context, 1, to_char_p_p([prg_bytes := src.encode()]),
                                                  ctypes.byref(ctypes.c_size_t(len(prg_bytes))), ctypes.byref(status := ctypes.c_int32())), status)
    build_status: int = cl.clBuildProgram(program, 1, ctypes.byref(self.device.device_id), None, cl.clBuildProgram.argtypes[4](), None)
    if build_status != 0:
      cl.clGetProgramBuildInfo(program, self.device.device_id, cl.CL_PROGRAM_BUILD_LOG, 0, None, ctypes.byref(log_size := ctypes.c_size_t()))
      cl.clGetProgramBuildInfo(program, self.device.device_id, cl.CL_PROGRAM_BUILD_LOG, log_size.value, mstr := ctypes.create_string_buffer(log_size.value), None)  # noqa: E501
      raise RuntimeError(f"OpenCL Compile Error\n\n{ctypes.string_at(mstr, size=log_size.value).decode()}")
    binary_sizes = init_c_var((ctypes.c_size_t * 1)(), lambda x: check(cl.clGetProgramInfo(program, cl.CL_PROGRAM_BINARY_SIZES, ctypes.sizeof(x), ctypes.byref(x), None)))  # noqa: E501
    binary = init_c_var(ctypes.create_string_buffer(binary_sizes[0]), lambda x: check(cl.clGetProgramInfo(program, cl.CL_PROGRAM_BINARIES, ctypes.sizeof(ctypes.c_void_p), ctypes.byref((ctypes.c_void_p * 1)(ctypes.addressof(x))), None)))  # noqa: E501
    check(cl.clReleaseProgram(program))
    return bytes(binary)

class CLProgram:
  def __init__(self, device:CLDevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    self.program = checked(cl.clCreateProgramWithBinary(device.context, 1, ctypes.byref(device.device_id), (ctypes.c_size_t * 1)(len(lib)),
                                                        to_char_p_p([lib], ctypes.c_ubyte), ctypes.byref(binary_status := ctypes.c_int32()),
                                                        ctypes.byref(errcode_ret := ctypes.c_int32())), errcode_ret)
    check(binary_status.value)
    check(cl.clBuildProgram(self.program, 1, ctypes.byref(device.device_id), None, cl.clBuildProgram.argtypes[4](), None)) # NOTE: OSX requires this
    self.kernel = checked(cl.clCreateKernel(self.program, name.encode(), ctypes.byref(status := ctypes.c_int32())), status)

  def __del__(self):
    if hasattr(self, 'kernel'): check(cl.clReleaseKernel(self.kernel))
    if hasattr(self, 'program'): check(cl.clReleaseProgram(self.program))

  def __call__(self, *bufs:ctypes._CData, global_size:Tuple[int,int,int]=(1,1,1), local_size:Optional[Tuple[int,int,int]]=None, vals:Tuple[int, ...]=(), wait=False) -> Optional[float]:  # noqa: E501
    for i,b in enumerate(bufs): cl.clSetKernelArg(self.kernel, i, ctypes.sizeof(b), ctypes.byref(b))
    for i,v in enumerate(vals,start=len(bufs)): cl.clSetKernelArg(self.kernel, i, 4, ctypes.byref(ctypes.c_int32(v)))
    if local_size is not None: global_size = cast(Tuple[int,int,int], tuple(int(g*l) for g,l in zip(global_size, local_size)))
    event = cl.cl_event() if wait else None
    check(cl.clEnqueueNDRangeKernel(self.device.queue, self.kernel, len(global_size), None, (ctypes.c_size_t * len(global_size))(*global_size), (ctypes.c_size_t * len(local_size))(*local_size) if local_size else None, 0, None, event))  # noqa: E501
    if wait:
      assert event is not None
      check(cl.clWaitForEvents(1, ctypes.byref(event)))
      start = init_c_var(ctypes.c_uint64(), lambda x: check(cl.clGetEventProfilingInfo(event, cl.CL_PROFILING_COMMAND_START, ctypes.sizeof(x), ctypes.byref(x), None)))  # noqa: E501
      end = init_c_var(ctypes.c_uint64(), lambda x: check(cl.clGetEventProfilingInfo(event, cl.CL_PROFILING_COMMAND_END, ctypes.sizeof(x), ctypes.byref(x), None)))  # noqa: E501
      return float(end.value-start.value) * OSX_TIMING_RATIO * 1e-9
    return None

class CLAllocator(LRUAllocator):
  def __init__(self, device:CLDevice):
    self.device = device
    super().__init__()
  def _alloc(self, size:int) -> ctypes._CData:
    return checked(cl.clCreateBuffer(self.device.context, cl.CL_MEM_READ_WRITE, size, None, ctypes.byref(status := ctypes.c_int32())), status)
  def _alloc_with_options(self, size:int, options:BufferOptions) -> ctypes._CData:
    if options.image is not None:
      return checked(cl.clCreateImage2D(self.device.context, cl.CL_MEM_READ_WRITE,
                                        cl.cl_image_format(cl.CL_RGBA, {2: cl.CL_HALF_FLOAT, 4: cl.CL_FLOAT}[options.image.itemsize]),
                                        options.image.shape[1], options.image.shape[0], 0, None, ctypes.byref(status := ctypes.c_int32())), status)
    else: return self._alloc(size)
  def _free(self, buf:ctypes._CData): check(cl.clReleaseMemObject(buf))
  def copyin(self, dest:ctypes._CData, src:memoryview):
    check(cl.clEnqueueWriteBuffer(self.device.queue, dest, False, 0, len(src)*src.itemsize, from_mv(src), 0, None, None))
    self.device.pending_copyin.append(src)    # NOTE: these can't be freed until the GPU actually executes this command
  def copyout(self, dest:memoryview, src:ctypes._CData):
    check(cl.clEnqueueReadBuffer(self.device.queue, src, False, 0, len(dest)*dest.itemsize, from_mv(dest), 0, None, None))
    self.device.synchronize()

class CLDevice(Compiled):
  device_ids = None                 # this is global and only initted once
  def __init__(self, device:str=""):
    if CLDevice.device_ids is None:
      num_platforms = init_c_var(ctypes.c_uint32(), lambda x: check(cl.clGetPlatformIDs(0, None, ctypes.byref(x))))
      platform_ids = init_c_var((cl.cl_platform_id * num_platforms.value)(), lambda x: check(cl.clGetPlatformIDs(num_platforms.value, x, None)))
      for device_type in [cl.CL_DEVICE_TYPE_GPU, cl.CL_DEVICE_TYPE_DEFAULT]:
        num_devices = ctypes.c_uint32()
        err = cl.clGetDeviceIDs(platform_ids[0], device_type, 0, None, ctypes.byref(num_devices))
        if err == 0 and num_devices.value != 0: break
      if DEBUG >= 1: print(f"CLDevice: got {num_platforms.value} platforms and {num_devices.value} devices")
      CLDevice.device_ids = init_c_var((cl.cl_device_id * num_devices.value)(), lambda x: check(cl.clGetDeviceIDs(platform_ids[0], device_type, num_devices, x, None)))  # noqa: E501

    self.device_id = CLDevice.device_ids[0 if ":" not in device else int(device.split(":")[1])]
    self.device_name = (cl.clGetDeviceInfo(self.device_id, cl.CL_DEVICE_NAME, 256, ctypes.byref(buf := ctypes.create_string_buffer(256)), ctypes.byref(total := ctypes.c_size_t())), ctypes.string_at(buf, size=total.value).decode())[1]  # noqa: E501
    self.driver_version = (cl.clGetDeviceInfo(self.device_id, cl.CL_DRIVER_VERSION, 256, ctypes.byref(buf := ctypes.create_string_buffer(256)), ctypes.byref(total := ctypes.c_size_t())), ctypes.string_at(buf, size=total.value).decode())[1]  # noqa: E501
    self.context = checked(cl.clCreateContext(None, 1, ctypes.byref(self.device_id), cl.clCreateContext.argtypes[3](), None, ctypes.byref(status := ctypes.c_int32())), status)  # noqa: E501
    self.queue = checked(cl.clCreateCommandQueue(self.context, self.device_id, cl.CL_QUEUE_PROFILING_ENABLE, ctypes.byref(status)), status)
    self.pending_copyin: List[memoryview] = []

    compile_key = hashlib.md5(self.device_name.encode() + self.driver_version.encode()).hexdigest()
    super().__init__(device, CLAllocator(self), CLCompiler(self, f"compile_cl_{compile_key}"), functools.partial(CLProgram, self))
  def synchronize(self):
    check(cl.clFinish(self.queue))
    self.pending_copyin.clear()

GPUDevice = CLDevice # for legacy reasons


# tinygrad/runtime/ops_hip.py

from __future__ import annotations
import ctypes, functools, subprocess, io
from typing import Tuple, TypeVar, List, Any, cast, Set
import tinygrad.runtime.autogen.hip as hip
from tinygrad.helpers import DEBUG, getenv, init_c_var
from tinygrad.helpers import from_mv, round_up, to_mv, colored, init_c_struct_t
from tinygrad.device import Compiled, LRUAllocator, BufferOptions, JITRunner, Device, Buffer, MallocAllocator, update_stats, Compiler
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.runtime.driver.hip_comgr import compile_hip

class HIPCompiler(Compiler):
  linearizer_opts = LinearizerOptions("HIP", has_tensor_cores=True)
  def __init__(self, arch:str):
    self.arch = arch
    super().__init__(f"compile_hip_{self.arch}")
  def render(self, name:str, uops) -> str: return HIPRenderer(name, uops)
  def compile(self, src:str) -> bytes: return compile_hip(src, self.arch)

hip_current_device = None
def hip_set_device(d:int):
  global hip_current_device
  if d == hip_current_device: return
  check(hip.hipSetDevice(d))
  hip_current_device = d

def check(status):
  if status != 0: raise RuntimeError(f"HIP Error {status}, {ctypes.string_at(hip.hipGetErrorString(status)).decode()}")

class HIPProgram:
  def __init__(self, device:int, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib

    if DEBUG >= 6:
      asm = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], input=lib)
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    hip_set_device(self.device)
    self.module = init_c_var(hip.hipModule_t(), lambda x: check(hip.hipModuleLoadData(ctypes.byref(x), lib)))
    self.prg = init_c_var(hip.hipFunction_t(), lambda x: check(hip.hipModuleGetFunction(ctypes.byref(x), self.module, name.encode("utf-8"))))

  def __del__(self):
    if hasattr(self, 'module'): check(hip.hipModuleUnload(self.module))

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    hip_set_device(self.device)
    if not hasattr(self, "vargs"):
      self.c_args = init_c_struct_t(tuple([(f'f{i}', hip.hipDeviceptr_t) for i in range(len(args))] +
                                          [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))(*args, *vals)
      self.vargs = (ctypes.c_void_p * 5)(ctypes.c_void_p(1), ctypes.cast(ctypes.byref(self.c_args), ctypes.c_void_p),
                                         ctypes.c_void_p(2), ctypes.cast(ctypes.byref(ctypes.c_size_t(ctypes.sizeof(self.c_args))), ctypes.c_void_p),
                                         ctypes.c_void_p(3))
    else:
      for i in range(len(args)): self.c_args.__setattr__(f'f{i}', args[i])
      for i in range(len(vals)): self.c_args.__setattr__(f'v{i}', vals[i])
    if wait:
      evs = [init_c_var(hip.hipEvent_t(), lambda x: hip.hipEventCreate(ctypes.byref(x), 0)) for _ in range(2)]
      check(hip.hipEventRecord(evs[0], None))
    check(hip.hipModuleLaunchKernel(self.prg, *global_size, *local_size, 0, None, None, self.vargs))
    if wait:
      check(hip.hipEventRecord(evs[1], None))
      check(hip.hipEventSynchronize(evs[1]))
      check(hip.hipEventElapsedTime(ctypes.byref(ret := ctypes.c_float()), evs[0], evs[1]))
      for ev in evs: check(hip.hipEventDestroy(ev))
      return ret.value * 1e-3
    return None

T = TypeVar("T")
CHUNK_SIZE, PAGE_SIZE = 256*1024*1024, 0x1000
class HIPAllocator(LRUAllocator):
  def __init__(self, device:HIPDevice):
    self.device = device
    self.track_cross_device: Set[HIPDevice] = set()
    super().__init__()
  def full_synchronize(self):
    self.device.synchronize()
    for x in self.track_cross_device: x.synchronize()
    self.track_cross_device.clear()
  def free_cache(self):
    self.full_synchronize()
    return super().free_cache()
  def _alloc(self, size:int):
    hip_set_device(self.device.device)
    return init_c_var(hip.hipDeviceptr_t(), lambda x: check(hip.hipMalloc(ctypes.byref(x), size)))
  def _alloc_with_options(self, size:int, options:BufferOptions):
    hip_set_device(self.device.device)
    if options.uncached:
      return init_c_var(hip.hipDeviceptr_t(), lambda x: check(hip.hipExtMallocWithFlags(ctypes.byref(x), size, 3)))  # hipDeviceMallocUncached = 3
    elif options.host:
      return init_c_var(hip.hipDeviceptr_t(), lambda x: check(hip.hipHostMalloc(ctypes.byref(x), size, 2 if options.signal else 0)))
    else:
      raise Exception("no options")
  def _free(self, opaque:T): check(hip.hipFree(opaque))
  def copy_from_fd(self, dest, fd, offset, size):
    hip_set_device(self.device.device)
    if not hasattr(self, 'hb'):
      self.hb = [self._alloc_with_options(CHUNK_SIZE, BufferOptions(host=True)) for _ in range(2)]
      self.hb_events = [None, None]
      self.hb_polarity = 0
    fo = io.FileIO(fd, "a+b", closefd=False)
    fo.seek(offset - (minor_offset:=offset % PAGE_SIZE))
    copied_in = 0
    for local_offset in range(0, size+minor_offset, CHUNK_SIZE):
      local_size = min(round_up(size+minor_offset, PAGE_SIZE)-local_offset, CHUNK_SIZE)
      if self.hb_events[self.hb_polarity] is not None:
        # NOTE: block doesn't work here because we modify the CPU memory
        check(hip.hipEventSynchronize(self.hb_events[self.hb_polarity]))
        check(hip.hipEventDestroy(self.hb_events[self.hb_polarity]))
        self.hb_events[self.hb_polarity] = None
      fo.readinto(to_mv(self.hb[self.hb_polarity], local_size))
      check(hip.hipMemcpyAsync(ctypes.c_void_p(dest.value + copied_in), ctypes.c_void_p(self.hb[self.hb_polarity].value + minor_offset),
                               copy_size:=min(local_size-minor_offset, size-copied_in), hip.hipMemcpyHostToDevice, None))
      self.hb_events[self.hb_polarity] = init_c_var(hip.hipEvent_t(), lambda x: check(hip.hipEventCreate(ctypes.byref(x))))
      check(hip.hipEventRecord(self.hb_events[self.hb_polarity], None))
      copied_in += copy_size
      self.hb_polarity = (self.hb_polarity+1) % len(self.hb)
      minor_offset = 0 # only on the first
  def copyin(self, dest:T, src: memoryview):
    hip_set_device(self.device.device)
    host_mem = self._alloc_with_options(len(src), BufferOptions(host=True))
    self.device.pending_copyin.append(host_mem)
    ctypes.memmove(host_mem, from_mv(src), len(src))
    check(hip.hipMemcpyAsync(dest, host_mem, len(src), hip.hipMemcpyHostToDevice, None))
  def copyout(self, dest:memoryview, src:T):
    self.full_synchronize()
    hip_set_device(self.device.device)
    check(hip.hipMemcpy(from_mv(dest), src, len(dest), hip.hipMemcpyDeviceToHost))
  def transfer(self, dest:T, src:T, sz:int, **kwargs):
    hip_set_device(self.device.device)
    check(hip.hipMemcpyAsync(dest, src, sz, hip.hipMemcpyDeviceToDevice, None))

class HIPSyncEvent(JITRunner):
  def __init__(self, lb):
    self.lb, self.device, self.dname = lb, cast(HIPDevice, Device[lb.device]), lb.device
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals, wait=False, jit=False):
    to_mv(rawbufs[0]._buf, 4).cast("I")[0] = 0
    hip_set_device(self.device.device)
    check(hip.hipStreamWriteValue32(None, rawbufs[0]._buf, 1, 0))
    update_stats(colored("sync", "red"), 0, 0, {}, None, 1, jit, device=self.dname)

class HIPWaitEvent(JITRunner):
  def __init__(self, device):
    self.device, self.dname = cast(HIPDevice, Device[device]), device
    super().__init__()
  def __call__(self, rawbufs:List[Buffer], var_vals, wait=False, jit=False):
    hip_set_device(self.device.device)
    check(hip.hipStreamWaitValue32(None, rawbufs[0]._buf, 1, 1, 0xFFFFFFFF))
    update_stats(colored("wait", "RED"), 0, 0, {}, None, 1, jit, device=self.dname)

if getenv("HIPCPU"):
  rhip = ctypes.CDLL("/usr/local/lib/libremu.so")
  class RHIPProgram:
    def __init__(self, name:str, lib:bytes):
      self.name, self.lib = name, lib
    def __call__(self, *args, global_size, local_size, vals=(), wait=False):
      args = (*args, *vals)
      rhip.hipModuleLaunchKernel(self.lib, len(self.lib), *global_size, *local_size, 0, None, None,
                                len(args), (ctypes.c_void_p * len(args))(*[ctypes.cast(x, ctypes.c_void_p) for x in args]))

class HIPDevice(Compiled):
  def __init__(self, device:str=""):
    self.device = int(device.split(":")[1]) if ":" in device else 0
    self.pending_copyin: List[ctypes.c_void_p] = []
    self.track_cross_buffer: List[Any] = []
    self.peers: Set[int] = set()

    if getenv("HIPCPU"):
      super().__init__(device, MallocAllocator, HIPCompiler("gfx1100"), RHIPProgram)
    else:
      self.arch = init_c_var(hip.hipDeviceProp_t(), lambda x: check(hip.hipGetDeviceProperties(x, self.device))).gcnArchName.decode()
      from tinygrad.runtime.graph.hip import HIPGraph
      super().__init__(device, HIPAllocator(self), HIPCompiler(self.arch),
                      functools.partial(HIPProgram, self.device), HIPGraph)
  def synchronize(self):
    if getenv("HIPCPU"): return
    hip_set_device(self.device)
    check(hip.hipDeviceSynchronize())
    for opaque in self.pending_copyin: check(hip.hipFree(opaque))
    self.track_cross_buffer.clear()
    self.pending_copyin.clear()
  def enable_peer(self, dnum):
    if self.device == dnum or dnum in self.peers: return
    hip_set_device(self.device)
    check(hip.hipDeviceEnablePeerAccess(dnum, 0))
    self.peers.add(dnum)


# tinygrad/runtime/ops_hsa.py
    
from __future__ import annotations
import ctypes, functools, subprocess, io, atexit
from typing import Tuple, TypeVar, List, Dict
import tinygrad.runtime.autogen.hsa as hsa
from tinygrad.helpers import DEBUG, init_c_var, from_mv, round_up, to_mv, init_c_struct_t
from tinygrad.device import Compiled, LRUAllocator, BufferOptions
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.runtime.ops_hip import HIPCompiler
from tinygrad.runtime.driver.hsa import check, scan_agents, find_memory_pool, AQLQueue

class HSACompiler(HIPCompiler):
  linearizer_opts = LinearizerOptions("HSA", has_tensor_cores=True)

class HSAProgram:
  def __init__(self, device:HSADevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib

    if DEBUG >= 6:
      asm = subprocess.check_output(["/opt/rocm/llvm/bin/llvm-objdump", '-d', '-'], input=lib)
      print('\n'.join([x for x in asm.decode('utf-8').split("\n") if 's_code_end' not in x]))

    self.exec = init_c_var(hsa.hsa_executable_t(), lambda x: check(hsa.hsa_executable_create_alt(hsa.HSA_PROFILE_FULL, hsa.HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, None, ctypes.byref(x)))) # noqa: E501
    self.code_reader = init_c_var(hsa.hsa_code_object_reader_t(),
                                  lambda x: check(hsa.hsa_code_object_reader_create_from_memory(lib, len(lib), ctypes.byref(x))))
    check(hsa.hsa_executable_load_agent_code_object(self.exec, self.device.agent, self.code_reader, None, None))
    check(hsa.hsa_executable_freeze(self.exec, None))

    self.kernel = init_c_var(hsa.hsa_executable_symbol_t(), lambda x: check(hsa.hsa_executable_get_symbol_by_name(self.exec, (name+".kd").encode("utf-8"), ctypes.byref(self.device.agent), ctypes.byref(x)))) # noqa: E501
    self.handle = init_c_var(ctypes.c_uint64(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, ctypes.byref(x)))) # noqa: E501
    self.kernargs_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, ctypes.byref(x)))).value # noqa: E501
    self.group_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, ctypes.byref(x)))).value # noqa: E501
    self.private_segment_size = init_c_var(ctypes.c_uint32(), lambda x: check(hsa.hsa_executable_symbol_get_info(self.kernel, hsa.HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, ctypes.byref(x)))).value # noqa: E501

  def __del__(self):
    self.device.synchronize()
    if hasattr(self, 'code_reader'): check(hsa.hsa_code_object_reader_destroy(self.code_reader))
    if hasattr(self, 'exec'): check(hsa.hsa_executable_destroy(self.exec))

  def __call__(self, *args, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    if not hasattr(self, "args_struct_t"):
      self.args_struct_t = init_c_struct_t(tuple([(f'f{i}', ctypes.c_void_p) for i in range(len(args))] +
                                                 [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))
      assert ctypes.sizeof(self.args_struct_t) == self.kernargs_segment_size, f"{ctypes.sizeof(self.args_struct_t)} != {self.kernargs_segment_size}"

    kernargs = None
    if self.kernargs_segment_size > 0:
      kernargs = self.device.alloc_kernargs(self.kernargs_segment_size)
      args_st = self.args_struct_t.from_address(kernargs)
      for i in range(len(args)): args_st.__setattr__(f'f{i}', args[i])
      for i in range(len(vals)): args_st.__setattr__(f'v{i}', vals[i])
      self.device.flush_hdp()

    signal = self.device.hw_queue.submit_kernel(self, global_size, local_size, kernargs, need_signal=wait)
    if wait:
      hsa.hsa_signal_wait_scacquire(signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
      check(hsa.hsa_amd_profiling_get_dispatch_time(self.device.agent, signal, ctypes.byref(timings := hsa.hsa_amd_profiling_dispatch_time_t())))
      return (timings.end - timings.start) * self.device.clocks_to_time

T = TypeVar("T")
CHUNK_SIZE, PAGE_SIZE = 256*1024*1024, 0x1000
class HSAAllocator(LRUAllocator):
  def __init__(self, device:HSADevice):
    self.device = device
    super().__init__()

  def _alloc(self, size:int):
    c_agents = (hsa.hsa_agent_t * len(HSADevice.agents[hsa.HSA_DEVICE_TYPE_GPU]))(*HSADevice.agents[hsa.HSA_DEVICE_TYPE_GPU])
    check(hsa.hsa_amd_memory_pool_allocate(self.device.gpu_mempool, size, 0, ctypes.byref(buf := ctypes.c_void_p())))
    check(hsa.hsa_amd_agents_allow_access(len(HSADevice.agents[hsa.HSA_DEVICE_TYPE_GPU]), c_agents, None, buf))
    return buf.value

  def _alloc_with_options(self, size:int, options:BufferOptions):
    if options.host:
      check(hsa.hsa_amd_memory_pool_allocate(HSADevice.cpu_mempool, size, 0, ctypes.byref(mem := ctypes.c_void_p())))
      check(hsa.hsa_amd_agents_allow_access(2, (hsa.hsa_agent_t*2)(HSADevice.cpu_agent, self.device.agent), None, mem))
      return mem.value
    else: raise Exception("no options")

  def _free(self, opaque:T):
    HSADevice.synchronize_system()
    check(hsa.hsa_amd_memory_pool_free(opaque))

  def copyin(self, dest:T, src: memoryview):
    # Async copyin sync model uses barriers on the main hw queue, since barriers are guaranteed to execute in order with all other packets.
    copy_signal = self.device.alloc_signal(reusable=True)
    sync_signal = self.device.hw_queue.submit_barrier(need_signal=True)
    mem = self._alloc_with_options(src.nbytes, BufferOptions(host=True))
    ctypes.memmove(mem, from_mv(src), src.nbytes)
    check(hsa.hsa_amd_memory_async_copy_on_engine(dest, self.device.agent, mem, HSADevice.cpu_agent, src.nbytes,
                                                  1, ctypes.byref(sync_signal), copy_signal, hsa.HSA_AMD_SDMA_ENGINE_0, True))
    self.device.hw_queue.submit_barrier(wait_signals=[copy_signal])
    self.device.delayed_free.append(mem)

  def copy_from_fd(self, dest, fd, offset, size):
    sync_signal = self.device.hw_queue.submit_barrier(need_signal=True)

    if not hasattr(self, 'hb'):
      self.hb = [self._alloc_with_options(CHUNK_SIZE, BufferOptions(host=True)) for _ in range(2)]
      self.hb_signals = [self.device.alloc_signal(reusable=False) for _ in range(2)]
      self.hb_polarity = 0
      self.sdma = [hsa.HSA_AMD_SDMA_ENGINE_0, hsa.HSA_AMD_SDMA_ENGINE_1]
      for sig in self.hb_signals: hsa.hsa_signal_store_relaxed(sig, 0)

    fo = io.FileIO(fd, "a+b", closefd=False)
    fo.seek(offset - (minor_offset:=offset % PAGE_SIZE))

    copies_called = 0
    copied_in = 0
    for local_offset in range(0, size+minor_offset, CHUNK_SIZE):
      local_size = min(round_up(size+minor_offset, PAGE_SIZE)-local_offset, CHUNK_SIZE)
      copy_size = min(local_size-minor_offset, size-copied_in)
      if copy_size == 0: break

      hsa.hsa_signal_wait_scacquire(self.hb_signals[self.hb_polarity], hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
      self.device.reusable_signals.append(self.hb_signals[self.hb_polarity]) # it's free now and can be reused
      self.hb_signals[self.hb_polarity] = self.device.alloc_signal(reusable=False)

      fo.readinto(to_mv(self.hb[self.hb_polarity], local_size))
      check(hsa.hsa_amd_memory_async_copy_on_engine(dest+copied_in, self.device.agent, self.hb[self.hb_polarity]+minor_offset, HSADevice.cpu_agent,
                                                    copy_size, 1, ctypes.byref(sync_signal), self.hb_signals[self.hb_polarity],
                                                    self.sdma[self.hb_polarity], True))
      copied_in += copy_size
      self.hb_polarity = (self.hb_polarity + 1) % len(self.hb)
      minor_offset = 0 # only on the first
      copies_called += 1

    wait_signals = [self.hb_signals[self.hb_polarity - 1]]
    if copies_called > 1: wait_signals.append(self.hb_signals[self.hb_polarity])
    self.device.hw_queue.submit_barrier(wait_signals=wait_signals)

  def copyout(self, dest:memoryview, src:T):
    HSADevice.synchronize_system()
    copy_signal = self.device.alloc_signal(reusable=True)
    c_agents = (hsa.hsa_agent_t*2)(self.device.agent, HSADevice.cpu_agent)
    check(hsa.hsa_amd_memory_lock_to_pool(from_mv(dest), dest.nbytes, c_agents, 2, HSADevice.cpu_mempool, 0, ctypes.byref(addr:=ctypes.c_void_p())))
    check(hsa.hsa_amd_memory_async_copy(addr, HSADevice.cpu_agent, src, self.device.agent, dest.nbytes, 0, None, copy_signal))
    hsa.hsa_signal_wait_scacquire(copy_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
    check(hsa.hsa_amd_memory_unlock(from_mv(dest)))

  def transfer(self, dest:T, src:T, sz:int, src_dev=None, dest_dev=None):
    copy_signal = dest_dev.alloc_signal(reusable=False)
    sync_signal_1 = src_dev.hw_queue.submit_barrier(need_signal=True)
    sync_signal_2 = dest_dev.hw_queue.submit_barrier(need_signal=True)
    c_wait_signal = (hsa.hsa_signal_t*2)(sync_signal_1, sync_signal_2)
    check(hsa.hsa_amd_memory_async_copy_on_engine(dest, dest_dev.agent, src, src_dev.agent, sz, 2, c_wait_signal, copy_signal, hsa.HSA_AMD_SDMA_ENGINE_0, True)) # noqa: E501
    src_dev.hw_queue.submit_barrier(wait_signals=[copy_signal])
    dest_dev.hw_queue.submit_barrier(wait_signals=[copy_signal])

class HSADevice(Compiled):
  devices: List[HSADevice] = []
  agents: Dict[int, List[hsa.hsa_agent_t]] = {}
  cpu_agent: hsa.hsa_agent_t
  cpu_mempool: hsa.hsa_amd_memory_pool_t
  def __init__(self, device:str=""):
    if not HSADevice.agents:
      check(hsa.hsa_init())
      atexit.register(lambda: hsa.hsa_shut_down())
      HSADevice.agents = scan_agents()
      HSADevice.cpu_agent = HSADevice.agents[hsa.HSA_DEVICE_TYPE_CPU][0]
      HSADevice.cpu_mempool = find_memory_pool(HSADevice.cpu_agent, segtyp=hsa.HSA_AMD_SEGMENT_GLOBAL, location=hsa.HSA_AMD_MEMORY_POOL_LOCATION_CPU)

    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.agent = HSADevice.agents[hsa.HSA_DEVICE_TYPE_GPU][self.device_id]
    self.gpu_mempool = find_memory_pool(self.agent, segtyp=hsa.HSA_AMD_SEGMENT_GLOBAL, location=hsa.HSA_AMD_MEMORY_POOL_LOCATION_GPU)
    self.hw_queue = AQLQueue(self)
    HSADevice.devices.append(self)

    check(hsa.hsa_agent_get_info(self.agent, hsa.HSA_AGENT_INFO_NAME, ctypes.byref(agent_name_buf := ctypes.create_string_buffer(256))))
    self.arch = ctypes.string_at(agent_name_buf).decode()

    check(hsa.hsa_system_get_info(hsa.HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, ctypes.byref(gpu_freq := ctypes.c_uint64())))
    self.clocks_to_time: float = 1 / gpu_freq.value

    check(hsa.hsa_agent_get_info(self.agent, hsa.HSA_AMD_AGENT_INFO_HDP_FLUSH, ctypes.byref(hdp_flush := hsa.hsa_amd_hdp_flush_t())))
    self.hdp_flush = hdp_flush

    self.delayed_free: List[int] = []
    self.reusable_signals: List[hsa.hsa_signal_t] = []

    from tinygrad.runtime.graph.hsa import HSAGraph
    super().__init__(device, HSAAllocator(self), HSACompiler(self.arch), functools.partial(HSAProgram, self), HSAGraph)

    # Finish init: preallocate some signals + space for kernargs
    self.signal_pool = [init_c_var(hsa.hsa_signal_t(), lambda x: check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(x)))) for _ in range(4096)]
    self._new_kernargs_region(16 << 20) # initial region size is 16mb

  def synchronize(self):
    self.hw_queue.wait()

    for sig in self.reusable_signals: hsa.hsa_signal_silent_store_relaxed(sig, 1)
    self.signal_pool.extend(self.reusable_signals)
    self.reusable_signals.clear()

    for opaque_to_free in self.delayed_free: check(hsa.hsa_amd_memory_pool_free(opaque_to_free))
    self.delayed_free.clear()

    self.kernarg_next_addr = self.kernarg_start_addr

  @staticmethod
  def synchronize_system():
    for d in HSADevice.devices: d.synchronize()

  def alloc_signal(self, reusable=False):
    if len(self.signal_pool): signal = self.signal_pool.pop()
    else: check(hsa.hsa_amd_signal_create(1, 0, None, 0, ctypes.byref(signal := hsa.hsa_signal_t())))

    # reusable means a signal could be reused after synchronize for the device it's allocated from is called.
    if reusable: self.reusable_signals.append(signal)
    return signal

  def alloc_kernargs(self, sz):
    if self.kernarg_next_addr + sz >= self.kernarg_start_addr + self.kernarg_pool_sz: self._new_kernargs_region(int(self.kernarg_pool_sz * 2))
    result = self.kernarg_next_addr
    self.kernarg_next_addr = (self.kernarg_next_addr + sz + 15) & (~15) # align to 16 bytes
    return result

  def _new_kernargs_region(self, sz:int):
    if hasattr(self, 'kernarg_start_addr'): self.delayed_free.append(self.kernarg_start_addr)
    self.kernarg_start_addr: int = self.allocator._alloc(sz)
    self.kernarg_next_addr = self.kernarg_start_addr
    self.kernarg_pool_sz: int = sz

  def flush_hdp(self): self.hdp_flush.HDP_MEM_FLUSH_CNTL[0] = 1


# tinygrad/runtime/ops_llvm.py
  
from __future__ import annotations
import ctypes, functools
from typing import Tuple
from tinygrad.device import Compiled, MallocAllocator, Compiler
from tinygrad.helpers import DEBUG, cpu_time_execution
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.llvmir import uops_to_llvm_ir
import llvmlite.binding as llvm

class LLVMCompiler(Compiler):
  linearizer_opts = LinearizerOptions("LLVM", supports_float4=False, has_local=False, has_shared=False)
  def __init__(self, device:LLVMDevice):
    self.device = device
    super().__init__("compile_llvm")
  def render(self, name:str, uops) -> str: return uops_to_llvm_ir(name, uops)
  def compile(self, src:str) -> bytes:
    mod = llvm.parse_assembly(src)
    mod.verify()
    self.device.optimizer.run(mod)
    if DEBUG >= 5: print(self.device.target_machine.emit_assembly(mod))
    return self.device.target_machine.emit_object(mod)

class LLVMProgram:
  def __init__(self, device:LLVMDevice, name:str, lib:bytes):
    self.name, self.lib = name, lib
    device.engine.add_object_file(llvm.object_file.ObjectFileRef.from_data(lib))
    self.fxn = device.engine.get_function_address(name)

  def __call__(self, *bufs, vals:Tuple[int, ...]=(), wait=False):
    self.cfunc = ctypes.CFUNCTYPE(ctypes.c_int, *([ctypes.c_void_p]*len(bufs)), *([ctypes.c_int32]*len(vals)))(self.fxn)
    return cpu_time_execution(lambda: self.cfunc(*bufs, *vals), enable=wait)

class LLVMDevice(Compiled):
  def __init__(self, device:str):
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()
    self.optimizer: llvm.passmanagers.ModulePassManager = llvm.create_module_pass_manager()
    # this opt actually can change things. ex: opt=3 means no FMA, opt=2 means FMA
    self.target_machine: llvm.targets.TargetMachine = llvm.Target.from_triple(llvm.get_process_triple()).create_target_machine(opt=2)
    self.target_machine.add_analysis_passes(self.optimizer)
    self.target_machine.set_asm_verbosity(True)
    backing_mod = llvm.parse_assembly(str())
    backing_mod.triple = llvm.get_process_triple()
    self.engine: llvm.executionengine.ExecutionEngine = llvm.create_mcjit_compiler(backing_mod, self.target_machine)
    super().__init__(device, MallocAllocator, LLVMCompiler(self), functools.partial(LLVMProgram, self))


# tinygrad/runtime/ops_metal.py
    
from __future__ import annotations
import os, subprocess, pathlib, ctypes, tempfile, functools
import Metal, libdispatch
from typing import List, Any, Tuple, Optional
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.helpers import prod, getenv, DEBUG, unwrap2
from tinygrad.device import Compiled, LRUAllocator, Compiler
from tinygrad.renderer.cstyle import MetalRenderer

def wait_check(cbuf: Any):
  cbuf.waitUntilCompleted()
  if (error := cbuf.error()) is not None:
    raise RuntimeError(error)

class MetalCompiler(Compiler):
  linearizer_opts = LinearizerOptions("METAL", has_tensor_cores=os.uname().machine == "arm64")
  def __init__(self, device:Optional[MetalDevice]):
    self.device = device
    super().__init__("compile_metal")
  def render(self, name:str, uops) -> str: return MetalRenderer(name, uops)
  def compile(self, src:str) -> bytes:
    if self.device is None:
      # NOTE: if you run llvm-dis on "air" you can see the llvm bytecode
      air = subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metal', '-x', 'metal', '-c', '-', '-o', '-'], input=src.encode('utf-8'))
      return subprocess.check_output(['xcrun', '-sdk', 'macosx', 'metallib', '-', '-o', '-'], input=air)
    else:
      options = Metal.MTLCompileOptions.new()
      options.setFastMathEnabled_(getenv("METAL_FAST_MATH"))
      library = unwrap2(self.device.device.newLibraryWithSource_options_error_(src, options, None))
      return library.libraryDataContents().bytes().tobytes()

class MetalProgram:
  def __init__(self, device:MetalDevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    if DEBUG >= 6:
      with tempfile.NamedTemporaryFile(delete=True) as shader:
        shader.write(lib)
        shader.flush()
        os.system(f"cd {pathlib.Path(__file__).parents[2]}/disassemblers/applegpu && python3 compiler_explorer.py {shader.name}")
    data = libdispatch.dispatch_data_create(lib, len(lib), None, None)
    self.library = unwrap2(self.device.device.newLibraryWithData_error_(data, None))
    self.fxn = self.library.newFunctionWithName_(name)
    self.pipeline_state = unwrap2(self.device.device.newComputePipelineStateWithFunction_error_(self.fxn, None))

  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    assert prod(local_size) <= self.pipeline_state.maxTotalThreadsPerThreadgroup(),f"local size {local_size} bigger than {self.pipeline_state.maxTotalThreadsPerThreadgroup()} with exec width {self.pipeline_state.threadExecutionWidth()} memory length {self.pipeline_state.staticThreadgroupMemoryLength()}"  # noqa: E501
    command_buffer = self.device.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(self.pipeline_state)
    for i,a in enumerate(bufs): encoder.setBuffer_offset_atIndex_(a, 0, i)
    for i,a in enumerate(vals,start=len(bufs)): encoder.setBytes_length_atIndex_(ctypes.c_int32(a), 4, i)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
    encoder.endEncoding()
    command_buffer.commit()
    if wait:
      wait_check(command_buffer)
      return command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    self.device.mtl_buffers_in_flight.append(command_buffer)

class MetalAllocator(LRUAllocator):
  def __init__(self, device:MetalDevice):
    self.device:MetalDevice = device
    super().__init__()
  def _alloc(self, size:int) -> Any:
    ret = self.device.device.newBufferWithLength_options_(size, Metal.MTLResourceStorageModeShared)
    if ret is None: raise MemoryError(f"Metal OOM while allocating {size=}")
    return ret
  def transfer(self, dest:Any, src:Any, sz:int, **kwargs):
    command_buffer = self.device.mtl_queue.commandBuffer()
    encoder = command_buffer.blitCommandEncoder()
    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(src, 0, dest, 0, sz)
    encoder.endEncoding()
    command_buffer.commit()
    self.device.mtl_buffers_in_flight.append(command_buffer)
  def from_buffer(self, src:memoryview) -> Optional[Any]:
    ret = self.device.device.newBufferWithBytesNoCopy_length_options_deallocator_(src, len(src), Metal.MTLResourceStorageModeShared, None)
    if ret: self.device.mv_in_metal.append(src)
    return ret
  def _free(self, opaque:Any): opaque.release()
  def as_buffer(self, src:Any) -> memoryview:
    self.device.synchronize()
    return src.contents().as_buffer(src.length())
  def copyin(self, dest:Any, src:memoryview): self.as_buffer(dest)[:] = src
  def copyout(self, dest:memoryview, src:Any): dest[:] = self.as_buffer(src)

class MetalDevice(Compiled):
  def __init__(self, device:str):
    self.device = Metal.MTLCreateSystemDefaultDevice()
    self.mtl_queue = self.device.newCommandQueueWithMaxCommandBufferCount_(1024)
    self.mtl_buffers_in_flight: List[Any] = []
    self.mv_in_metal: List[memoryview] = []
    from tinygrad.runtime.graph.metal import MetalGraph
    super().__init__(device, MetalAllocator(self), MetalCompiler(None if getenv("METAL_XCODE") else self),
                     functools.partial(MetalProgram, self), functools.partial(MetalGraph, self))
  def synchronize(self):
    for cbuf in self.mtl_buffers_in_flight: wait_check(cbuf)
    self.mv_in_metal.clear()
    self.mtl_buffers_in_flight.clear()


# tinygrad/runtime/ops_python.py
    
# a python uops emulator
# works to test the tensor cores, and all the uops in general
# this is the (living) definition of uops
from typing import Tuple, List, Optional, Any, Dict
import pickle, base64, itertools, time, struct
from tinygrad.dtype import DType, dtypes, ImageDType
from tinygrad.helpers import all_same, getenv, flatten
from tinygrad.device import Compiled, Allocator, Compiler
from tinygrad.codegen.uops import UOp, UOps, exec_alu
from tinygrad.ops import BinaryOps, TernaryOps
from tinygrad.codegen.kernel import LinearizerOptions

def _load(m, i):
  if i<0 or i>=len(m): raise IndexError(f"load out of bounds, size is {len(m)} and access is {i}")
  return m[i]
def load(inp, j=0):
  if len(inp) == 4:
    return [_load(m, x+j) if gate else default for m,x,gate,default in zip(*inp)]
  else:
    return [_load(m, x+j) for m,x in zip(inp[0], inp[1])]

def _store(m, i, v):
  if i<0 or i>=len(m): raise IndexError(f"store out of bounds, size is {len(m)}, access is {i}, value is {v}")
  m[i] = v

class PythonProgram:
  def __init__(self, name:str, lib:bytes):
    self.uops: List[Tuple[UOps, Optional[DType], List[int], Any]] = pickle.loads(lib)
  def __call__(self, *bufs, global_size:Tuple[int,int,int]=(1,1,1), local_size:Tuple[int,int,int]=(1,1,1), vals:Tuple[int, ...]=(), wait=False):
    st = time.perf_counter()
    warp = list(itertools.product(*[range(x) for x in local_size[::-1]]))
    warp_size = len(warp)
    for idxs in itertools.product(*[range(x) for x in global_size[::-1]]):
      ul: Dict[int, Any] = {}
      dl: Dict[int, DType] = {}
      pbufs: List[memoryview] = list(bufs)
      pvals: List[int] = list(vals)
      i = 0
      loop_ends: Dict[int, int] = {}
      while i < len(self.uops):
        uop, dtype, idp, arg = self.uops[i]
        void_ops = {UOps.STORE, UOps.ENDLOOP, UOps.BARRIER, UOps.IF, UOps.ENDIF}
        inp = [ul[v] for v in idp if self.uops[v][0] not in void_ops]
        dtp = [dl[v] for v in idp if self.uops[v][0] not in void_ops]
        if getenv("TRACE"): print(i, uop, dtype, arg, inp, dtp)
        if uop is UOps.STORE:
          assert len(inp) <= 3, "gated stores not supported yet"
          if isinstance(dtp[0], ImageDType):
            # image store
            assert dtp[2].count == 4
            for j,val in enumerate(inp[2]):
              for m,ox,oy,v in zip(inp[0], inp[1][0], inp[1][1], val):
                assert ox >= 0 and ox < dtp[0].shape[1] and oy >= 0 and oy < dtp[0].shape[0]
                _store(m, ox*4 + oy*dtp[0].shape[1]*4 + j, v)
          elif dtp[2].count > 1:
            for j,val in enumerate(inp[2]):
              for m,o,v in zip(inp[0], inp[1], val): _store(m, o+j, v)
          else:
            for m,o,v in zip(*inp): _store(m, o, v)
          i += 1
          continue
        elif uop is UOps.ENDLOOP:
          loop_ends[idp[0]] = i
          i = idp[0]
          continue
        elif uop in (UOps.BARRIER, UOps.IF, UOps.ENDIF):
          # in the python emulator, the warp is always in sync
          i += 1
          continue
        assert dtype is not None, f"{uop} is missing a dtype"
        dl[i] = dtype
        if uop is UOps.DEFINE_GLOBAL:
          assert dtype.fmt is not None
          ul[i] = [pbufs[arg[0]].cast(dtype.fmt)] * warp_size
        elif uop is UOps.DEFINE_LOCAL:
          assert dtype.fmt is not None
          lbuf = memoryview(bytearray(arg[1]*dtype.itemsize))
          ul[i] = [lbuf.cast(dtype.fmt)] * warp_size
        elif uop is UOps.DEFINE_VAR:
          ul[i] = [pvals.pop(0)] * warp_size
        elif uop is UOps.SPECIAL:
          if arg[1][0] == 'g':
            ul[i] = [idxs[2-arg[0]]] * warp_size
          elif arg[1][0] == 'l':
            ul[i] = [x[2-arg[0]] for x in warp]
        elif uop is UOps.CONST:
          casted_arg = int(arg) if dtypes.is_int(dtype) else float(arg)
          if dtype.count > 1:
            ul[i] = [[casted_arg] * warp_size for _ in range(dtype.count)]
          else:
            ul[i] = [casted_arg] * warp_size
        elif uop is UOps.DEFINE_ACC:
          if dtype.count > 1:
            ul[i] = [[arg] * warp_size for _ in range(dtype.count)]
          else:
            ul[i] = [arg] * warp_size
        elif uop is UOps.LOOP:
          if i not in ul:
            ul[i] = [inp[0][0]] * warp_size
          else:
            for j in range(len(ul[i])):
              ul[i][j] += 1
            if ul[i][0] == inp[1][0]:
              del ul[i]
              i = loop_ends[i] + 1
              continue
        elif uop is UOps.CAST:
          if dtype.count > 1:
            ul[i] = inp
          else:
            assert dtp[0].fmt and dtype.fmt
            pack_format, unpack_format = str(warp_size) + dtp[0].fmt, str(warp_size) + dtype.fmt
            if arg[1]:
              ul[i] = list(struct.unpack(unpack_format, struct.pack(pack_format, *inp[0])))
            else:
              casted = [float(x) if dtypes.is_float(dtype) else int(x) if dtypes.is_int(dtype) else x for x in inp[0]]
              overflow_adjust = 2**(dtype.itemsize*8 - 1) if (dtypes.is_int(dtype) and not dtypes.is_unsigned(dtype)) else 0
              overflow_fixed = [((x + overflow_adjust) % 2**(dtype.itemsize*8) - overflow_adjust) if dtypes.is_int(dtype) else x for x in casted]
              ul[i] = list(struct.unpack(unpack_format, struct.pack(unpack_format, *overflow_fixed)))
        elif uop is UOps.LOAD:
          if isinstance(dtp[0], ImageDType):
            assert dtype.count == 4
            ul[i] = []
            for j in range(dtype.count):
              ret = []
              for m,ox,oy in zip(inp[0], inp[1][0], inp[1][1]):
                if ox < 0 or ox >= dtp[0].shape[1] or oy < 0 or oy >= dtp[0].shape[0]: ret.append(0)
                else: ret.append(_load(m, ox*4 + oy*dtp[0].shape[1]*4 + j))
              ul[i].append(ret)
          elif dtype.count > 1:
            ul[i] = [load([inp[i][j] if dtp[i].count > 1 else inp[i] for i in range(len(inp))], j) for j in range(dtype.count)]
          else:
            ul[i] = load(inp)
        elif uop is UOps.PHI:
          for j in range(len(inp[0])):
            inp[0][j] = inp[1][j]
          ul[i] = inp[0]
        elif uop is UOps.GEP:
          ul[i] = inp[0][arg]
        elif uop is UOps.WMMA:
          # here are the models for the WMMA instruction on the different hardware
          def wmma_helper(WARP_THREADS, K, NUM_A, NUM_B, NUM_C, a_elem, b_elem, c_map):
            assert len(inp[0]) == NUM_A, f"A must have {NUM_A} elements per thread"
            assert len(inp[1]) == NUM_B, f"B must have {NUM_B} elements per thread"
            assert len(inp[2]) == NUM_C, f"C must have {NUM_C} elements per thread"
            assert len(flatten(inp[0])) == NUM_A * warp_size, f"WMMA must have {NUM_A * warp_size} total elements for A in WMMA"
            assert len(flatten(inp[1])) == NUM_B * warp_size, f"WMMA must have {NUM_B * warp_size} total elements for B in WMMA"
            assert len(flatten(inp[2])) == NUM_C * warp_size, f"WMMA must have {NUM_C * warp_size} total elements for C in WMMA"
            assert warp_size > 0 and warp_size % WARP_THREADS == 0, f"must have multiples of {WARP_THREADS} warp threads"
            out = [inp[2][elem_idx][:] for elem_idx in range(NUM_C)]
            for goff in range(0, warp_size, WARP_THREADS):
              for lane_id in range(WARP_THREADS):
                for elem_idx in range(NUM_C): # calculate new muls and add to acc
                  (c_i, c_j) = c_map(lane_id, elem_idx)
                  out[elem_idx][goff+lane_id] += sum(a_elem(inp[0], _k, c_j, goff) * b_elem(inp[1], c_i, _k, goff) for _k in range(K))
            return out

          if arg.startswith('__metal_wmma'):
            def a_b_elem(x, i, j, goff): # A (2 elements on 32 threads): row major
              return x[(i%2)][goff+(i//2)%2+(j%4)*2+(i//4)*8+(j//4)*16]
            def c_map(lane, elem): # (i, j), C, D (2 elements on 32 threads): row major same as A/B
              return (elem + ((lane%2)*2) + ((lane//8)%2)*4, ((lane//2)%4) + (lane//16)*4)
            ul[i] = wmma_helper(32, 8, 2, 2, 2, a_b_elem, a_b_elem, c_map)
          elif arg == '__builtin_amdgcn_wmma_f32_16x16x16_f16_w32' or arg == '__hip_wmma_f16_f16':
            def a_elem(x, i, j, goff): # A (16 elements on 32 threads): col major, lane 16-32 == lane 0-15
              assert x[i][goff+j] == x[i][goff+j+16], "warp elements not duplicated properly across lanes"
              return x[i][goff+j]
            def b_elem(x, i, j, goff): # B (16 elements on 32 threads): row major, lane 16-32 == lane 0-15
              return a_elem(x, j, i, goff)
            def c_map(lane, elem): return (lane%16, lane//16+elem*2) # (i, j), C, D (8 elements on 32 threads): row major
            ul[i] = wmma_helper(32, 16, 16, 16, 8, a_elem, b_elem, c_map)
          elif arg == '__cuda_mma_m16n8k16_f16_f32':
            def a_elem(x, i, j, goff): return x[(i%2)+(j//8)*2+(i//8)*4][goff+((i//2)%4)+(j%8)*4] # A (8 elements on 32 threads)
            def b_elem(x, i, j, goff): return x[(j%2)+(j//8)*2][goff+(j//2)%4+(i)*4] # B (4 elements on 32 threads)
            def c_map(lane, elem): return ((elem%2)+(lane%4)*2, (lane//4)+(elem//2)*8) # (i, j), C, D (4 elements on 32 threads)
            ul[i] = wmma_helper(32, 16, 8, 4, 4, a_elem, b_elem, c_map)
          else:
            raise Exception(f"unimplemented tensor core {arg}")
        elif uop is UOps.ALU:
          assert all_same([len(x) for x in inp]), f"{[len(x) for x in inp]} doesn't match on {arg}"
          assert all_same([dtype] + dtp) or arg in {BinaryOps.CMPEQ, BinaryOps.CMPLT, TernaryOps.WHERE}, f"dtype mismatch on {arg}"
          ul[i] = [exec_alu(arg, dtype, p) for p in zip(*inp)]
        assert i in ul, (uop, dtype, idp, arg)
        i += 1
    return time.perf_counter() - st

class PythonCompiler(Compiler):
  linearizer_opts = LinearizerOptions("METAL", has_tensor_cores=True) if getenv("EMULATE_METAL") else \
    (LinearizerOptions("HIP", has_tensor_cores=True) if getenv("EMULATE_HIP") else \
    (LinearizerOptions("CUDA", has_tensor_cores=True) if getenv("EMULATE_CUDA") else LinearizerOptions("PYTHON")))
  def render(self, name:str, uops:List[UOp]) -> str:
    lops = [(u.uop, u.dtype, [uops.index(v) for v in u.vin], u.arg) for u in uops]
    return base64.b64encode(pickle.dumps(lops)).decode()
  def compile(self, src:str) -> bytes: return base64.b64decode(src)

class PythonAllocator(Allocator):
  def _alloc(self, size): return memoryview(bytearray(size))
  def copyin(self, dest, src:memoryview): dest[:] = src
  def copyout(self, dest:memoryview, src): dest[:] = src

class PythonDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, PythonAllocator(), PythonCompiler(), PythonProgram)


# tinygrad/runtime/driver/hip_comgr.py

import ctypes
import tinygrad.runtime.autogen.comgr as comgr

def check(status):
  if status != 0:
    comgr.amd_comgr_status_string(status, ctypes.byref(status_str := ctypes.POINTER(ctypes.c_char)()))
    raise RuntimeError(f"comgr fail {status}, {ctypes.string_at(status_str).decode()}")

def _get_comgr_data(data_set, data_type):
  check(comgr.amd_comgr_action_data_get_data(data_set, data_type, 0, ctypes.byref(data_exec := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz := ctypes.c_uint64()), None))
  check(comgr.amd_comgr_get_data(data_exec, ctypes.byref(sz), (dat := ctypes.create_string_buffer(sz.value))))
  check(comgr.amd_comgr_release_data(data_exec))
  return bytes(dat)

# AMD_COMGR_SAVE_TEMPS=1 AMD_COMGR_REDIRECT_LOGS=stdout AMD_COMGR_EMIT_VERBOSE_LOGS=1
def compile_hip(prg:str, arch="gfx1100") -> bytes:
  check(comgr.amd_comgr_create_action_info(ctypes.byref(action_info := comgr.amd_comgr_action_info_t())))
  check(comgr.amd_comgr_action_info_set_language(action_info, comgr.AMD_COMGR_LANGUAGE_HIP))
  check(comgr.amd_comgr_action_info_set_isa_name(action_info, b"amdgcn-amd-amdhsa--" + arch.encode()))
  check(comgr.amd_comgr_action_info_set_logging(action_info, True))

  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_src := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_bc := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_reloc := comgr.amd_comgr_data_set_t())))
  check(comgr.amd_comgr_create_data_set(ctypes.byref(data_set_exec := comgr.amd_comgr_data_set_t())))

  check(comgr.amd_comgr_create_data(comgr.AMD_COMGR_DATA_KIND_SOURCE, ctypes.byref(data_src := comgr.amd_comgr_data_t())))
  check(comgr.amd_comgr_set_data(data_src, len(rprg := prg.encode()), rprg))
  check(comgr.amd_comgr_set_data_name(data_src, b"<null>"))

  check(comgr.amd_comgr_data_set_add(data_set_src, data_src))
  # -include hiprtc_runtime.h was removed
  check(comgr.amd_comgr_action_info_set_options(action_info, f"-O3 -mcumode --hip-version=6.0.32830 -DHIP_VERSION_MAJOR=6 -DHIP_VERSION_MINOR=0 -DHIP_VERSION_PATCH=32830 -D__HIPCC_RTC__ -std=c++14 -nogpuinc -Wno-gnu-line-marker -Wno-missing-prototypes --offload-arch={arch} -I/opt/rocm/include -Xclang -disable-llvm-passes".encode())) # noqa: E501
  status = comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC, action_info, data_set_src, data_set_bc)
  if status != 0:
    print(_get_comgr_data(data_set_bc, comgr.AMD_COMGR_DATA_KIND_LOG).decode())
    raise RuntimeError("compile failed")
  check(comgr.amd_comgr_action_info_set_options(action_info, b"-O3 -mllvm -amdgpu-internalize-symbols"))
  check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, action_info, data_set_bc, data_set_reloc))
  check(comgr.amd_comgr_action_info_set_options(action_info, b""))
  check(comgr.amd_comgr_do_action(comgr.AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, action_info, data_set_reloc, data_set_exec))
  ret = _get_comgr_data(data_set_exec, comgr.AMD_COMGR_DATA_KIND_EXECUTABLE)
  check(comgr.amd_comgr_release_data(data_src))
  for x in [data_set_src, data_set_bc, data_set_reloc, data_set_exec]: check(comgr.amd_comgr_destroy_data_set(x))
  check(comgr.amd_comgr_destroy_action_info(action_info))
  return ret


# tinygrad/runtime/driver/hsa.py

import ctypes, collections
import tinygrad.runtime.autogen.hsa as hsa
from tinygrad.helpers import init_c_var

def check(status):
  if status != 0:
    hsa.hsa_status_string(status, ctypes.byref(status_str := ctypes.POINTER(ctypes.c_char)()))
    raise RuntimeError(f"HSA Error {status}: {ctypes.string_at(status_str).decode()}")

# Precalulated AQL info
AQL_PACKET_SIZE = ctypes.sizeof(hsa.hsa_kernel_dispatch_packet_t)
EMPTY_SIGNAL = hsa.hsa_signal_t()

DISPATCH_KERNEL_SETUP = 3 << hsa.HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS
DISPATCH_KERNEL_HEADER  = 1 << hsa.HSA_PACKET_HEADER_BARRIER
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
DISPATCH_KERNEL_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
DISPATCH_KERNEL_HEADER |= hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE

BARRIER_HEADER  = 1 << hsa.HSA_PACKET_HEADER_BARRIER
BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE
BARRIER_HEADER |= hsa.HSA_FENCE_SCOPE_SYSTEM << hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE
BARRIER_HEADER |= hsa.HSA_PACKET_TYPE_BARRIER_AND << hsa.HSA_PACKET_HEADER_TYPE

class AQLQueue:
  def __init__(self, device, sz=-1):
    self.device = device
    self.wait_signals = []

    check(hsa.hsa_agent_get_info(self.device.agent, hsa.HSA_AGENT_INFO_QUEUE_MAX_SIZE, ctypes.byref(max_queue_size := ctypes.c_uint32())))
    queue_size = min(max_queue_size.value, sz) if sz != -1 else max_queue_size.value

    null_func = ctypes.CFUNCTYPE(None, hsa.hsa_status_t, ctypes.POINTER(hsa.struct_hsa_queue_s), ctypes.c_void_p)()
    self.hw_queue = init_c_var(ctypes.POINTER(hsa.hsa_queue_t)(), lambda x: check(
      hsa.hsa_queue_create(self.device.agent, queue_size, hsa.HSA_QUEUE_TYPE_SINGLE, null_func, None, (1<<32)-1, (1<<32)-1, ctypes.byref(x))))

    self.next_doorbell_index = 0
    self.queue_size = self.hw_queue.contents.size
    self.write_addr = self.hw_queue.contents.base_address
    self.write_addr_end = self.hw_queue.contents.base_address + (AQL_PACKET_SIZE * self.queue_size) - 1
    self.available_packet_slots = self.queue_size

    check(hsa.hsa_amd_queue_set_priority(self.hw_queue, hsa.HSA_AMD_QUEUE_PRIORITY_HIGH))
    check(hsa.hsa_amd_profiling_set_profiler_enabled(self.hw_queue, 1))

  def __del__(self):
    if hasattr(self, 'hw_queue'): check(hsa.hsa_queue_destroy(self.hw_queue))

  def submit_kernel(self, prg, global_size, local_size, kernargs, need_signal=False):
    if self.available_packet_slots == 0: self._wait_queue()
    signal = self._alloc_signal(reusable=True) if need_signal else EMPTY_SIGNAL

    packet = hsa.hsa_kernel_dispatch_packet_t.from_address(self.write_addr)
    packet.workgroup_size_x = local_size[0]
    packet.workgroup_size_y = local_size[1]
    packet.workgroup_size_z = local_size[2]
    packet.reserved0 = 0
    packet.grid_size_x = global_size[0] * local_size[0]
    packet.grid_size_y = global_size[1] * local_size[1]
    packet.grid_size_z = global_size[2] * local_size[2]
    packet.private_segment_size = prg.private_segment_size
    packet.group_segment_size = prg.group_segment_size
    packet.kernel_object = prg.handle
    packet.kernarg_address = kernargs
    packet.reserved2 = 0
    packet.completion_signal = signal
    packet.setup = DISPATCH_KERNEL_SETUP
    packet.header = DISPATCH_KERNEL_HEADER
    self._submit_packet()

    return signal

  def submit_barrier(self, wait_signals=None, need_signal=False, completion_signal=None):
    assert wait_signals is None or len(wait_signals) <= 5
    if self.available_packet_slots == 0: self._wait_queue()
    signal = (completion_signal or self._alloc_signal(reusable=True)) if need_signal else EMPTY_SIGNAL

    packet = hsa.hsa_barrier_and_packet_t.from_address(self.write_addr)
    packet.reserved0 = 0
    packet.reserved1 = 0
    for i in range(5):
      packet.dep_signal[i] = wait_signals[i] if wait_signals and len(wait_signals) > i else EMPTY_SIGNAL
    packet.reserved2 = 0
    packet.completion_signal = signal
    packet.header = BARRIER_HEADER
    self._submit_packet()

    return signal

  def blit_packets(self, packet_addr, packet_cnt):
    if self.available_packet_slots < packet_cnt: self._wait_queue(packet_cnt)

    tail_blit_packets = min(((self.write_addr_end + 1) - self.write_addr) // 64, packet_cnt)
    rem_packet_cnt = packet_cnt - tail_blit_packets
    ctypes.memmove(self.write_addr, packet_addr, AQL_PACKET_SIZE * tail_blit_packets)
    self.write_addr += AQL_PACKET_SIZE * tail_blit_packets
    if self.write_addr > self.write_addr_end: self.write_addr = self.hw_queue.contents.base_address
    if tail_blit_packets > 0:
      ctypes.memmove(self.write_addr, packet_addr + AQL_PACKET_SIZE * tail_blit_packets, AQL_PACKET_SIZE * rem_packet_cnt)
      self.write_addr += AQL_PACKET_SIZE * rem_packet_cnt

    self.next_doorbell_index += packet_cnt
    hsa.hsa_queue_store_write_index_screlease(self.hw_queue, self.next_doorbell_index + 1)
    hsa.hsa_signal_store_screlease(self.hw_queue.contents.doorbell_signal, self.next_doorbell_index)

  def wait(self):
    signal = self.submit_barrier(need_signal=True)
    hsa.hsa_signal_wait_scacquire(signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
    self.available_packet_slots = self.queue_size

  def _wait_queue(self, need_packets=1):
    while self.available_packet_slots < need_packets:
      rindex = hsa.hsa_queue_load_read_index_relaxed(self.hw_queue)
      self.available_packet_slots = self.queue_size - (self.next_doorbell_index - rindex)

  def _submit_packet(self):
    hsa.hsa_queue_store_write_index_relaxed(self.hw_queue, self.next_doorbell_index + 1)
    hsa.hsa_signal_store_screlease(self.hw_queue.contents.doorbell_signal, self.next_doorbell_index)

    self.write_addr += AQL_PACKET_SIZE
    if self.write_addr > self.write_addr_end: self.write_addr = self.hw_queue.contents.base_address
    self.next_doorbell_index += 1
    self.available_packet_slots -= 1

  def _alloc_signal(self, reusable=False): return self.device.alloc_signal(reusable=reusable)

def scan_agents():
  agents = collections.defaultdict(list)

  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_agent_t, ctypes.c_void_p)
  def __scan_agents(agent, data):
    status = hsa.hsa_agent_get_info(agent, hsa.HSA_AGENT_INFO_DEVICE, ctypes.byref(device_type := hsa.hsa_device_type_t()))
    if status == 0: agents[device_type.value].append(agent)
    return hsa.HSA_STATUS_SUCCESS

  hsa.hsa_iterate_agents(__scan_agents, None)
  return agents

def find_memory_pool(agent, segtyp=-1, location=-1):
  @ctypes.CFUNCTYPE(hsa.hsa_status_t, hsa.hsa_amd_memory_pool_t, ctypes.c_void_p)
  def __filter_amd_memory_pools(mem_pool, data):
    check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_SEGMENT, ctypes.byref(segment := hsa.hsa_amd_segment_t())))
    if segtyp >= 0 and segment.value != segtyp: return hsa.HSA_STATUS_SUCCESS

    check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_LOCATION, ctypes.byref(loc:=hsa.hsa_amd_memory_pool_location_t())))
    if location >= 0 and loc.value != location: return hsa.HSA_STATUS_SUCCESS

    check(hsa.hsa_amd_memory_pool_get_info(mem_pool, hsa.HSA_AMD_MEMORY_POOL_INFO_SIZE, ctypes.byref(sz := ctypes.c_size_t())))
    if sz.value == 0: return hsa.HSA_STATUS_SUCCESS

    ret = ctypes.cast(data, ctypes.POINTER(hsa.hsa_amd_memory_pool_t))
    ret[0] = mem_pool
    return hsa.HSA_STATUS_INFO_BREAK

  hsa.hsa_amd_agent_iterate_memory_pools(agent, __filter_amd_memory_pools, ctypes.byref(region := hsa.hsa_amd_memory_pool_t()))
  return region


# tinygrad/runtime/graph/cuda.py

import ctypes
from typing import Any, Optional, Tuple, Dict, List, cast
import tinygrad.runtime.autogen.cuda as cuda
from tinygrad.helpers import init_c_var, encode_args_cuda_style, all_same, GraphException
from tinygrad.device import CompiledASTRunner, update_stats, Buffer
from tinygrad.runtime.ops_cuda import check, cu_time_execution
from tinygrad.shape.symbolic import Variable
from tinygrad.features.jit import JitItem, get_input_replace, get_jit_stats, \
                                  get_jc_idxs_with_updatable_launch_dims, get_jc_idxs_with_updatable_var_vals

class CUDAGraph:
  def __init__(self, jit_cache: List[JitItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    devices = [ji.prg.clprg.device if isinstance(ji.prg, CompiledASTRunner) else None for ji in jit_cache]
    if len(devices) == 0 or not all_same(devices) or devices[0] is None: raise GraphException
    self.device = devices[0]
    self.set_device()

    self.jit_cache = jit_cache
    self.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    self.op_estimate, self.mem_estimate = get_jit_stats(jit_cache)
    self.jc_idxs_with_updatable_launch_dims = get_jc_idxs_with_updatable_launch_dims(jit_cache)
    self.jc_idxs_with_updatable_var_vals = get_jc_idxs_with_updatable_var_vals(jit_cache)
    self.jc_idxs_with_updatable_rawbufs = list(set([x[0] for x in self.input_replace.keys()]))
    self.updatable_nodes: Dict[int, Tuple[Any, Any, Any]] = {} # Dict[jc index] = tuple(graph node, node params, input kernel params)

    self.graph = self.graph_create()
    graph_node: Optional[ctypes._CData] = None

    for (j,i),input_name in self.input_replace.items(): self.jit_cache[j].rawbufs[i] = input_rawbuffers[input_name]
    for j,ji in enumerate(self.jit_cache):
      prg: CompiledASTRunner = cast(CompiledASTRunner, ji.prg)

      c_deps = (type(graph_node)*1)(*(graph_node,)) if graph_node is not None else None
      c_kernel_input_config, c_input_params = encode_args_cuda_style([cast(Buffer, x)._buf for x in ji.rawbufs], [var_vals[x] for x in prg.vars], *self.encode_args_info())  # noqa: E501
      c_node_params = self.build_kernel_node_params(prg, *cast(Tuple[List[int], List[int]], prg.launch_dims(var_vals)), c_kernel_input_config)
      graph_node = self.graph_add_kernel_node(self.graph, c_deps, c_node_params)

      if j in self.jc_idxs_with_updatable_launch_dims or j in self.jc_idxs_with_updatable_var_vals or j in self.jc_idxs_with_updatable_rawbufs:
        self.updatable_nodes[j] = (graph_node, c_node_params, c_input_params)

    self.instance = self.graph_instantiate(self.graph)

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    self.set_device()
    # Update rawbuffers in the c_input_params struct.
    for (j,i),input_idx in self.input_replace.items():
      setattr(self.updatable_nodes[j][2], f'f{i}', input_rawbuffers[input_idx]._buf)

    # Update var_vals in the c_input_params struct.
    for j in self.jc_idxs_with_updatable_var_vals:
      for i,v in enumerate(cast(CompiledASTRunner, self.jit_cache[j].prg).vars):
        setattr(self.updatable_nodes[j][2], f'f{len(self.jit_cache[j].rawbufs) + i}', var_vals[v])

    # Update launch dims in the c_node_params struct.
    for j in self.jc_idxs_with_updatable_launch_dims:
      self.set_kernel_node_launch_dims(self.updatable_nodes[j][1], *cast(CompiledASTRunner, self.jit_cache[j].prg).launch_dims(var_vals))

    # Update graph nodes with the updated structs.
    for node, c_node_params, _ in self.updatable_nodes.values():
      self.graph_exec_kernel_node_set_params(self.instance, node, ctypes.byref(c_node_params))

    et = self.graph_launch(self.instance, None, wait=wait)
    update_stats(f"<batched {len(self.jit_cache)}>", self.op_estimate, self.mem_estimate, var_vals, et, buf_count=len(input_rawbuffers),
                 jit=jit, num_kernels=len(self.jit_cache), device=f"<GPU>:{self.device}")
    return et

  def __del__(self):
    if hasattr(self, 'graph'): check(cuda.cuGraphDestroy(self.graph))
    if hasattr(self, 'instance'): check(cuda.cuGraphExecDestroy(self.instance))

  def set_device(self): pass
  def encode_args_info(self): return (cuda.CUdeviceptr_v2, (1,2,0))
  def graph_create(self): return init_c_var(cuda.CUgraph(), lambda x: check(cuda.cuGraphCreate(ctypes.byref(x), 0)))
  def graph_instantiate(self, graph):
    return init_c_var(cuda.CUgraphExec(), lambda x: check(cuda.cuGraphInstantiate_v2(ctypes.byref(x), graph, None, None, 0)))
  def graph_add_kernel_node(self, graph, c_deps, c_node_params):
    return init_c_var(cuda.CUgraphNode(), lambda x: check(cuda.cuGraphAddKernelNode(ctypes.byref(x), graph, c_deps, ctypes.sizeof(c_deps)//8 if c_deps else 0, ctypes.byref(c_node_params))))  # noqa: E501
  def graph_launch(self, *args, wait=False): return cu_time_execution(lambda: check(cuda.cuGraphLaunch(*args)), enable=wait)
  def graph_exec_kernel_node_set_params(self, *args): return check(cuda.cuGraphExecKernelNodeSetParams(*args))
  def build_kernel_node_params(self, prg, global_size, local_size, c_kernel_config):
    return cuda.CUDA_KERNEL_NODE_PARAMS(prg.clprg.prg, *global_size, *local_size, 0, None, c_kernel_config)
  def set_kernel_node_launch_dims(self, node, global_size: Tuple[int, int, int], local_size: Tuple[int, int, int]):
    node.blockDimX, node.blockDimY, node.blockDimZ, node.gridDimX, node.gridDimY, node.gridDimZ = *local_size, *global_size


# tinygrad/runtime/graph/hip.py
    
import ctypes
from typing import Tuple
import tinygrad.runtime.autogen.hip as hip
from tinygrad.helpers import init_c_var, time_execution_cuda_style
from tinygrad.runtime.ops_hip import check, hip_set_device
from tinygrad.runtime.graph.cuda import CUDAGraph

# TODO: this is only used in graph
def hip_time_execution(cb, enable=False): return time_execution_cuda_style(cb, hip.hipEvent_t, hip.hipEventCreate, hip.hipEventRecord, hip.hipEventSynchronize, hip.hipEventDestroy, hip.hipEventElapsedTime, enable=enable)  # noqa: E501

class HIPGraph(CUDAGraph):
  def __del__(self):
    if hasattr(self, 'graph'): check(hip.hipGraphDestroy(self.graph))
    if hasattr(self, 'instance'): check(hip.hipGraphExecDestroy(self.instance))
  def set_device(self): hip_set_device(self.device)
  def encode_args_info(self): return (hip.hipDeviceptr_t, (1,2,3))
  def graph_create(self): return init_c_var(hip.hipGraph_t(), lambda x: check(hip.hipGraphCreate(ctypes.byref(x), 0)))
  def graph_instantiate(self, graph):
    return init_c_var(hip.hipGraphExec_t(), lambda x: check(hip.hipGraphInstantiate(ctypes.byref(x), graph, None, None, 0)))
  def graph_add_kernel_node(self, graph, c_deps, c_params):
    return init_c_var(hip.hipGraphNode_t(), lambda x: check(hip.hipGraphAddKernelNode(ctypes.byref(x), graph, c_deps, ctypes.sizeof(c_deps)//8 if c_deps else 0, ctypes.byref(c_params))))  # noqa: E501
  def graph_launch(self, *args, wait=False): return hip_time_execution(lambda: check(hip.hipGraphLaunch(*args)), enable=wait)
  def graph_exec_kernel_node_set_params(self, *args): return check(hip.hipGraphExecKernelNodeSetParams(*args))
  def build_kernel_node_params(self, prg, global_size, local_size, c_config):
    return hip.hipKernelNodeParams(hip.dim3(*local_size), c_config, ctypes.cast(prg.clprg.prg, ctypes.c_void_p), hip.dim3(*global_size), None, 0)
  def set_kernel_node_launch_dims(self, node, global_size: Tuple[int, int, int], local_size: Tuple[int, int, int]):
    node.blockDim.x, node.blockDim.y, node.blockDim.z, node.gridDim.x, node.gridDim.y, node.gridDim.z = *local_size, *global_size


# tinygrad/runtime/graph/hsa.py

import ctypes, collections, time, itertools
from typing import List, Any, Dict, cast, Optional, Union
from tinygrad.helpers import GraphException, init_c_var
from tinygrad.device import Compiled, Buffer, CompiledASTRunner, BufferXfer, MultiDeviceJITGraph, update_stats
from tinygrad.shape.symbolic import Variable
from tinygrad.runtime.ops_hsa import HSADevice
from tinygrad.features.jit import JitItem, get_input_replace, get_jit_stats, \
                                  get_jc_idxs_with_updatable_launch_dims, get_jc_idxs_with_updatable_var_vals
import tinygrad.runtime.autogen.hsa as hsa
from tinygrad.runtime.driver.hsa import check, AQLQueue, AQL_PACKET_SIZE, EMPTY_SIGNAL

def dedup_signals(signals): return [hsa.hsa_signal_t(hndl) for hndl in set([x.handle for x in signals if isinstance(x, hsa.hsa_signal_t)])]

class VirtAQLQueue(AQLQueue):
  def __init__(self, device, sz):
    self.device = device
    self.virt_queue = (hsa.hsa_kernel_dispatch_packet_t * sz)()
    self.queue_base = self.write_addr = ctypes.addressof(self.virt_queue)
    self.packets_count = 0
    self.available_packet_slots = sz
  def _wait_queue(self, need_packets=1): assert False, f"VirtQueue is too small to handle {self.packets_count+need_packets} packets!"
  def _submit_packet(self):
    self.write_addr += AQL_PACKET_SIZE
    self.packets_count += 1
    self.available_packet_slots -= 1
  def _alloc_signal(self, reusable=False): return init_c_var(hsa.hsa_signal_t(), lambda x: check(hsa.hsa_signal_create(1, 0, None, ctypes.byref(x))))

class HSAGraph(MultiDeviceJITGraph):
  def __init__(self, jit_cache: List[JitItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    self.jit_cache = jit_cache
    self.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    self.op_estimate, self.mem_estimate = get_jit_stats(jit_cache) #type:ignore
    self.jc_idxs_with_updatable_launch_dims = get_jc_idxs_with_updatable_launch_dims(jit_cache)
    self.jc_idxs_with_updatable_var_vals = get_jc_idxs_with_updatable_var_vals(jit_cache)

    # Check all jit items are compatible.
    compiled_devices = set()
    for ji in self.jit_cache:
      if isinstance(ji.prg, CompiledASTRunner): compiled_devices.add(ji.prg.device)
      elif isinstance(ji.prg, BufferXfer):
        for x in ji.rawbufs[0:2]: compiled_devices.add(cast(Buffer, x).d)
      else: raise GraphException
    if any(not isinstance(d, HSADevice) for d in compiled_devices): raise GraphException

    self.devices: List[HSADevice] = list(compiled_devices) #type:ignore

    # Allocate kernel args.
    kernargs_size: Dict[HSADevice, int] = collections.defaultdict(int)
    for ji in self.jit_cache:
      if isinstance(ji.prg, CompiledASTRunner): kernargs_size[cast(HSADevice, ji.prg.device)] += (ctypes.sizeof(ji.prg.clprg.args_struct_t)+15) & ~15
    kernargs_ptrs: Dict[Compiled, int] = {dev:dev.allocator._alloc(sz) for dev,sz in kernargs_size.items()}

    # Fill initial arguments.
    self.ji_kargs_structs: Dict[int, ctypes.Structure] = {}
    for j,ji in enumerate(self.jit_cache):
      if not isinstance(ji.prg, CompiledASTRunner): continue
      self.ji_kargs_structs[j] = ji.prg.clprg.args_struct_t.from_address(kernargs_ptrs[ji.prg.device])
      kernargs_ptrs[ji.prg.device] += (ctypes.sizeof(ji.prg.clprg.args_struct_t) + 15) & ~15
      for i in range(len(ji.rawbufs)): self.ji_kargs_structs[j].__setattr__(f'f{i}', cast(Buffer, ji.rawbufs[i])._buf)
      for i in range(len(ji.prg.vars)): self.ji_kargs_structs[j].__setattr__(f'v{i}', var_vals[ji.prg.vars[i]])

    # Build queues.
    self.virt_aql_queues: Dict[Compiled, VirtAQLQueue] = {dev:VirtAQLQueue(dev, 2*len(self.jit_cache)+16) for dev in self.devices}
    self.packets = {}
    self.transfers = []
    self.signals_to_reset: List[hsa.hsa_signal_t] = []
    self.w_dependency_map: Dict[Any, Union[hsa.hsa_signal_t, int]] = {}
    self.r_dependency_map: Dict[Any, List[Union[hsa.hsa_signal_t, int]]] = collections.defaultdict(list)
    signals_to_devices: Dict[ctypes.c_uint64, List[HSADevice]] = {}

    # Special packet to wait for the world.
    self.kickoff_signals: Dict[HSADevice, hsa.hsa_signal_t] = {}
    for dev in self.devices: self.kickoff_signals[dev] = self.virt_aql_queues[dev].submit_barrier(need_signal=True)
    self.signals_to_reset += list(self.kickoff_signals.values())

    for j,ji in enumerate(self.jit_cache):
      if isinstance(ji.prg, CompiledASTRunner):
        wait_signals = self.access_resources(read=ji.rawbufs[1:], write=ji.rawbufs[0:1], new_dependency=j, sync_with_aql_packets=False)
        for i in range(0, len(wait_signals), 5):
          self.virt_aql_queues[ji.prg.device].submit_barrier(wait_signals=wait_signals[i:i+5])
        self.packets[j] = hsa.hsa_kernel_dispatch_packet_t.from_address(self.virt_aql_queues[ji.prg.device].write_addr)
        self.virt_aql_queues[ji.prg.device].submit_kernel(ji.prg.clprg, *ji.prg.launch_dims(var_vals), ctypes.addressof(self.ji_kargs_structs[j])) #type:ignore
      elif isinstance(ji.prg, BufferXfer):
        dest, src = [cast(Buffer, x) for x in ji.rawbufs[0:2]]
        dest_dev, src_dev = cast(HSADevice, dest.d), cast(HSADevice, src.d)
        sync_signal = init_c_var(hsa.hsa_signal_t(), lambda x: check(hsa.hsa_amd_signal_create(1, 0, None, 0, ctypes.byref(x))))
        self.signals_to_reset.append(sync_signal)
        signals_to_devices[sync_signal.handle] = [dest_dev, src_dev]

        wait_signals = self.access_resources(read=[src], write=[dest], new_dependency=sync_signal, sync_with_aql_packets=True)
        self.transfers.append((dest._buf, dest_dev.agent, src._buf, src_dev.agent, dest.nbytes, len(wait_signals),
                              (hsa.hsa_signal_t*len(wait_signals))(*wait_signals), sync_signal, hsa.HSA_AMD_SDMA_ENGINE_0, True))

    # Wait for all active signals to finish the graph
    wait_signals_to_finish: Dict[HSADevice, List[hsa.hsa_signal_t]] = collections.defaultdict(list)
    for v in dedup_signals(list(self.w_dependency_map.values()) + list(itertools.chain.from_iterable(self.r_dependency_map.values()))):
      for dev in signals_to_devices[v.handle]:
        wait_signals_to_finish[dev].append(v)

    self.finish_signal = init_c_var(hsa.hsa_signal_t(), lambda x: check(hsa.hsa_amd_signal_create(1, 0, None, 0, ctypes.byref(x))))
    for dev in self.devices:
      wait_signals = wait_signals_to_finish[dev]
      for i in range(0, max(1, len(wait_signals)), 5):
        self.virt_aql_queues[dev].submit_barrier(wait_signals[i:i+5], need_signal=(i+5>=len(wait_signals)), completion_signal=self.finish_signal)

    # Zero signals to allow graph to start and execute.
    for sig in self.signals_to_reset: hsa.hsa_signal_silent_store_relaxed(sig, 0)
    hsa.hsa_signal_silent_store_relaxed(self.finish_signal, 0)

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    # Wait and restore signals
    hsa.hsa_signal_wait_scacquire(self.finish_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
    for sig in self.signals_to_reset: hsa.hsa_signal_silent_store_relaxed(sig, 1)
    hsa.hsa_signal_silent_store_relaxed(self.finish_signal, len(self.devices))

    # Update rawbuffers
    for (j,i),input_idx in self.input_replace.items():
      self.ji_kargs_structs[j].__setattr__(f'f{i}', input_rawbuffers[input_idx]._buf)

    # Update var_vals
    for j in self.jc_idxs_with_updatable_var_vals:
      for i,v in enumerate(cast(CompiledASTRunner, self.jit_cache[j].prg).vars):
        self.ji_kargs_structs[j].__setattr__(f'v{i}', var_vals[v])

    # Update launch dims
    for j in self.jc_idxs_with_updatable_launch_dims:
      gl, lc = cast(CompiledASTRunner, self.jit_cache[j].prg).launch_dims(var_vals)
      self.packets[j].workgroup_size_x = lc[0]
      self.packets[j].workgroup_size_y = lc[1]
      self.packets[j].workgroup_size_z = lc[2]
      self.packets[j].grid_size_x = gl[0] * lc[0]
      self.packets[j].grid_size_y = gl[1] * lc[1]
      self.packets[j].grid_size_z = gl[2] * lc[2]

    for dev in self.devices:
      dev.flush_hdp()
      dev.hw_queue.blit_packets(self.virt_aql_queues[dev].queue_base, self.virt_aql_queues[dev].packets_count)

    for transfer_data in self.transfers:
      check(hsa.hsa_amd_memory_async_copy_on_engine(*transfer_data))

    et = None
    if wait:
      st = time.perf_counter()
      hsa.hsa_signal_wait_scacquire(self.finish_signal, hsa.HSA_SIGNAL_CONDITION_LT, 1, (1 << 64) - 1, hsa.HSA_WAIT_STATE_ACTIVE)
      et = time.perf_counter() - st

    update_stats(f"<batched {len(self.jit_cache)}>", self.op_estimate, self.mem_estimate, var_vals, et, buf_count=len(input_rawbuffers),
                 jit=jit, num_kernels=len(self.jit_cache), device="HSA")
    return et

  def dependency_as_signal(self, dep, sync_with_aql_packets) -> Optional[hsa.hsa_signal_t]:
    if isinstance(dep, hsa.hsa_signal_t): return dep
    elif sync_with_aql_packets and isinstance(packet := self.packets.get(dep), hsa.hsa_kernel_dispatch_packet_t):
      if packet.completion_signal.handle == EMPTY_SIGNAL.handle:
        packet.completion_signal = init_c_var(hsa.hsa_signal_t(), lambda x: check(hsa.hsa_amd_signal_create(1, 0, None, 0, ctypes.byref(x))))
        self.signals_to_reset.append(packet.completion_signal)
      return packet.completion_signal
    return None

  def access_resources(self, read, write, new_dependency=None, sync_with_aql_packets=False):
    # To synchronize access to resources, we monitor the necessary prerequisites for accessing each resource,
    # whether for write or read operations. A resource can be accessed by either a single writer or multiple readers.
    # The tracked dependencies are either hsa signals or ints that reference a specific aql packet.
    wait_signals: List[Optional[hsa.hsa_signal_t]] = []

    if sync_with_aql_packets: wait_signals += [self.kickoff_signals[rawbuf.d] for rawbuf in read+write]
    for rawbuf in read:
      wait_signals.append(self.dependency_as_signal(self.w_dependency_map.get(rawbuf._buf), sync_with_aql_packets=sync_with_aql_packets))
    for rawbuf in write:
      wait_signals.append(self.dependency_as_signal(self.w_dependency_map.get(rawbuf._buf), sync_with_aql_packets=sync_with_aql_packets))
      if rawbuf._buf in self.r_dependency_map:
        rdeps = self.r_dependency_map.pop(rawbuf._buf)

        # When synchronizing to aql packets, we only need to sync to the latest one, as they are executed in order.
        signal_deps, aql_deps = [x for x in rdeps if isinstance(x, hsa.hsa_signal_t)], [x for x in rdeps if isinstance(x, int)]
        deps = signal_deps + [max(aql_deps)] if aql_deps else []
        for dep in deps: wait_signals.append(self.dependency_as_signal(dep, sync_with_aql_packets=sync_with_aql_packets))

    if new_dependency is not None:
      for rawbuf in read: self.r_dependency_map[rawbuf._buf].append(new_dependency)
      for rawbuf in write: self.w_dependency_map[rawbuf._buf] = new_dependency

    return dedup_signals(wait_signals)


# tinygrad/runtime/graph/metal.py
  
from typing import List, Any, Dict, cast, Optional
import Metal
from tinygrad.dtype import dtypes
from tinygrad.helpers import dedup, unwrap2, GraphException
from tinygrad.device import Buffer, CompiledASTRunner, update_stats
from tinygrad.features.jit import JitItem, get_input_replace, get_jit_stats, get_jc_idxs_with_updatable_launch_dims
from tinygrad.shape.symbolic import Variable
from tinygrad.runtime.ops_metal import MetalDevice, wait_check

class MetalGraph:
  def __init__(self, device:MetalDevice, jit_cache: List[JitItem], input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int]):
    if not all(isinstance(ji.prg, CompiledASTRunner) for ji in jit_cache): raise GraphException

    self.jit_cache = jit_cache
    self.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    self.op_estimate, self.mem_estimate = get_jit_stats(jit_cache)
    self.jc_idx_with_updatable_launch_dims = get_jc_idxs_with_updatable_launch_dims(jit_cache)
    self.device: MetalDevice = device

    # create metal batch exec
    icb_descriptor = Metal.MTLIndirectCommandBufferDescriptor.new()
    icb_descriptor.setCommandTypes_(Metal.MTLIndirectCommandType(Metal.MTLIndirectCommandTypeConcurrentDispatch))
    icb_descriptor.setInheritBuffers_(False)
    icb_descriptor.setInheritPipelineState_(False)
    icb_descriptor.setMaxKernelBufferBindCount_(31)
    self.icb = self.device.device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options_(icb_descriptor, len(self.jit_cache), Metal.MTLResourceOptions(0))  # noqa: E501
    if self.icb is None: raise GraphException("create indirect command buffer failed, does your system support this?")

    if len(var_vals): self.int_buf = self.device.allocator.alloc(len(var_vals)*dtypes.int32.itemsize)
    all_resources = [self.int_buf] if len(var_vals) else []
    for j,ji in enumerate(self.jit_cache):
      prg: CompiledASTRunner = cast(CompiledASTRunner, ji.prg)
      descriptor = Metal.MTLComputePipelineDescriptor.new()
      descriptor.setComputeFunction_(prg.clprg.fxn)
      descriptor.setSupportIndirectCommandBuffers_(True)
      pipeline_state = unwrap2(self.device.device.newComputePipelineStateWithDescriptor_options_reflection_error_(descriptor, Metal.MTLPipelineOption(0), None, None))  # noqa: E501
      icb_command = self.icb.indirectComputeCommandAtIndex_(j)
      icb_command.setComputePipelineState_(pipeline_state)
      for i,b in enumerate(ji.rawbufs):
        if b is not None:
          icb_command.setKernelBuffer_offset_atIndex_(b._buf, 0, i)
          all_resources.append(b._buf)
      var_vals_keys = list(var_vals.keys())
      for i,v in enumerate(prg.vars):
        icb_command.setKernelBuffer_offset_atIndex_(self.int_buf, var_vals_keys.index(v)*4, len(ji.rawbufs)+i)
      if j not in self.jc_idx_with_updatable_launch_dims:
        global_size, local_size = prg.launch_dims(var_vals)
        icb_command.concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))
      icb_command.setBarrier()
    self.all_resources = dedup(all_resources)
    self.command_buffer: Any = None
    if len(var_vals): self.int_buf_view = self.int_buf.contents().as_buffer(self.int_buf.length()).cast('i')

  def __call__(self, input_rawbuffers: List[Buffer], var_vals: Dict[Variable, int], wait=False, jit=False) -> Optional[float]:
    # NOTE: you at least can't update the ints if this is running
    if self.command_buffer is not None and self.command_buffer in self.device.mtl_buffers_in_flight: wait_check(self.command_buffer)
    all_resources = self.all_resources + [x._buf for x in input_rawbuffers]
    for (j,i),input_idx in self.input_replace.items():
      self.icb.indirectComputeCommandAtIndex_(j).setKernelBuffer_offset_atIndex_(input_rawbuffers[input_idx]._buf, 0, i)
    for j in self.jc_idx_with_updatable_launch_dims:
      global_size, local_size = cast(CompiledASTRunner, self.jit_cache[j].prg).launch_dims(var_vals)
      self.icb.indirectComputeCommandAtIndex_(j).concurrentDispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*global_size), Metal.MTLSize(*local_size))  # noqa: E501
    for j, value in enumerate(var_vals.values()): self.int_buf_view[j] = value
    command_buffer = self.device.mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.useResources_count_usage_(all_resources, len(all_resources), Metal.MTLResourceUsageRead | Metal.MTLResourceUsageWrite)
    encoder.executeCommandsInBuffer_withRange_(self.icb, Metal.MTLIndirectCommandBufferExecutionRangeMake(0,len(self.jit_cache)))
    encoder.endEncoding()
    command_buffer.commit()
    self.command_buffer = command_buffer
    if wait:
      wait_check(command_buffer)
      et = command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    else:
      self.device.mtl_buffers_in_flight.append(command_buffer)
      et = None
    update_stats(f"<batched {len(self.jit_cache)}>", self.op_estimate, self.mem_estimate, var_vals, et, buf_count=len(input_rawbuffers), jit=jit, num_kernels=len(self.jit_cache))  # noqa: E501
    return et
