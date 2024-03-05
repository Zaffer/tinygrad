# this is a single file representing each of the modules in the
# /shape directory of the tinygrad project


# tinygrad/shape/shapetracker.py

# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from __future__ import annotations
import functools, math
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Set, Iterable, cast
from tinygrad.helpers import merge_dicts, getenv
from tinygrad.shape.symbolic import Variable, MulNode, Node, SumNode, NumNode, create_lt_node, create_ge_node, sint
from tinygrad.shape.view import View, strides_for_shape

def un1d(shape:Tuple[sint, ...], offs:sint) -> List[sint]:
  strides = strides_for_shape(shape)
  result = []
  for stride in strides:
    here = offs // stride if stride else 0
    result.append(here)
    offs -= here * stride
  return result

@functools.lru_cache(maxsize=None)
def merge_views(vm2:View, vm1:View) -> Optional[View]:
  if vm2.contiguous: return vm1
  if vm1.contiguous and vm1.shape == vm2.shape: return vm2
  if vm1.contiguous and vm1.size() == vm2.size() and (ret := vm2.reshape(vm1.shape)) is not None: return ret
  if not vm2.mask and vm1.offset == 0 and None not in (rstrides := ShapeTracker((vm2, vm1)).real_strides()):
    return View.create(vm1.shape, cast(Tuple[sint, ...], rstrides), vm2.offset, vm1.mask)
  if vm1.mask:
    for b,e in vm1.mask:
      if not (b < e): return View.create(vm1.shape, (0,) * len(vm1.shape), 0, ((0,0),) * len(vm1.shape))
    return (merged := merge_views(vm2, vm1.shrink(vm1.mask))) and merged.pad(tuple((b,s-e) for (b,e),s in zip(vm1.mask, vm1.shape)))

  # Project vm1's offset and strides on to vm2.
  origin = un1d(vm2.shape, vm1.offset)
  terms: List[List[Tuple[int, sint]]] = [[] for _ in origin]
  strides: List[sint] = [0] * len(vm1.shape)
  for d1, st in enumerate(vm1.strides):
    if st == 0: continue
    for d2, (o, s1) in enumerate(zip(origin, un1d(vm2.shape, vm1.offset + st))):
      if (s1 := s1 - o) == 0: continue
      terms[d2].append((d1, s1))
      strides[d1] += s1 * vm2.strides[d2]

  # Merge dimensions in vm2 if required.
  # NB: Merging too many dimensions can make it difficult to project vm2's mask, hence only combining when required.
  idxs: List[Node] = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(vm1.shape)]
  merged_size, merged_term = 1, NumNode(0)
  extents: List[Tuple[sint, Node]] = []
  for term, s, o in zip(reversed(terms), reversed(vm2.shape), reversed(origin)):
    merged_term += Variable.sum([idxs[d1] * (s1 * merged_size) for d1, s1 in term]) + o * merged_size
    merged_size *= s
    if not (merged_term >= merged_size) and not (merged_term < 0):
      extents.append((merged_size, merged_term))
      merged_size, merged_term = 1, NumNode(0)
  if merged_term: return None
  if (vm2_shape := tuple(s for s,_ in reversed(extents))) != vm2.shape:
    return (reshaped_vm2 := vm2.reshape(vm2_shape)) and merge_views(reshaped_vm2, vm1)

  if vm2.mask:
    # Try to project vm2's mask on to vm1.
    newb, newe, bad = [0] * len(vm1.shape), list(vm1.shape), False
    for d2, ((b, e), o, (_, t)) in enumerate(zip(vm2.mask, origin, reversed(extents))):
      if not (t.min < b or t.max >= e): continue
      if not isinstance(o, int) or not isinstance(b, int) or not isinstance(e, int):
        bad = True
        continue
      term = terms[d2]
      if len(term) != 1:
        if not term and newe: newe[0] = 0
        else: bad = True
        continue
      d1, s1 = term[0]
      if not isinstance(s1, int) or not isinstance(newe[d1], int):
        bad = True
        continue
      newb[d1] = max(newb[d1], math.ceil((b - o if s1 > 0 else e - o - 1) / s1))
      newe[d1] = min(newe[d1], (b - o if s1 < 0 else e - o - 1) // s1 + 1)

    # If any of vm1 was masked off, try again with that mask in place.
    for b, e, s in zip(newb, newe, vm1.shape):
      if b != 0 or e != s:
        return merge_views(vm2, View.create(vm1.shape, vm1.strides, vm1.offset, tuple(zip(newb, newe))))
    # Otherwise if vm2's mask was violated, then cannot merge.
    if bad: return None

  return View.create(vm1.shape, tuple(strides), sum(o * s for o, s in zip(origin, vm2.strides)) + vm2.offset)

def _expr_view(view:View, idxs:List[Node], valid:Optional[Node]=None) -> Tuple[Node, Node]:
  assert len(idxs) == len(view.shape), f"need an idx for all dimensions {idxs} vs {view.shape}"
  iexpr: List[Node] = [NumNode(view.offset) if isinstance(view.offset, int) else view.offset]
  vexpr: List[Node] = [valid] if valid is not None else []
  for idx,sh,st,m in zip(idxs, view.shape, view.strides, view.mask if view.mask is not None else [None]*len(view.shape)):
    if sh != 1 and st != 0: iexpr.append(idx*st)
    if m is not None: vexpr += [create_ge_node(idx, m[0]), create_lt_node(idx, m[1])]  # idx >= m[0], idx < m[1]
  return Node.sum(iexpr), Node.ands(vexpr)

@dataclass(frozen=True)
class ShapeTracker:
  views: Tuple[View, ...]

  def __add__(self, st:ShapeTracker) -> ShapeTracker:
    ret = self
    for v in st.views: ret = ShapeTracker(ret.views + (v,)).simplify() # one view at a time = better simplification
    return ret

  def invert(self, out_shape:Tuple[sint, ...]) -> Optional[ShapeTracker]:
    ret = tuple(v.invert(s) for v,s in zip(self.views[::-1], [x.shape for x in self.views[::-1][1:]]+[out_shape]))
    return ShapeTracker(cast(Tuple[View, ...], ret)).reshape(out_shape) if all(x is not None for x in ret) else None

  @staticmethod
  def from_shape(shape:Tuple[sint, ...]): return ShapeTracker((View.create(shape),))

  @property
  def contiguous(self) -> bool: return len(self.views) == 1 and self.views[0].contiguous

  @property
  def shape(self) -> Tuple[sint, ...]: return self.views[-1].shape

  @property
  def size(self) -> int: return self.views[-1].size()

  def real_size(self) -> int:
    if 0 in self.shape: return 0
    ret = self.expr_idxs()[0].max
    if not isinstance(ret, int): ret = ret.max  # might be represent by symbolic shape, take one more max for int max
    assert isinstance(ret, int), f"ret must be integer, {ret=} isn't"
    return ret+1

  def vars(self) -> Set[Variable]: return set.union(*[v.vars() for v in self.views], set())

  @property
  def var_vals(self) -> Dict[Variable, int]: return merge_dicts([dict([v.unbind()]) for v in self.vars()])

  def unbind(self) -> Tuple[ShapeTracker, Dict[Variable, int]]:
    unbound_views, var_vals = zip(*[v.unbind() for v in self.views])
    return ShapeTracker(tuple(unbound_views)), merge_dicts(var_vals)

  # NOTE: if a stride is not always valid, it will be None
  def real_strides(self, ignore_valid=False) -> Tuple[Optional[sint], ...]:
    if len(self.views) == 1 and self.views[-1].mask is None: return self.views[-1].strides
    idxs: List[Node] = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)]
    idx, valid = self.expr_idxs(idxs)
    ret: List[Optional[sint]] = [None] * len(self.views[-1].shape)
    bad_idx_vars: Set[Variable] = set()
    for this_dim in (idx.nodes if isinstance(idx, SumNode) else [idx]):
      idx_maybe, stride_maybe = (this_dim.a, this_dim.b) if isinstance(this_dim, MulNode) else (this_dim, 1)
      try: ret[idxs.index(idx_maybe)] = cast(sint, stride_maybe)
      except ValueError: bad_idx_vars = bad_idx_vars.union(idx_maybe.vars())
    idx_vars, valid_vars = idx.vars(), valid.vars()
    for i,tidx in enumerate(idxs):
      if tidx in bad_idx_vars or (tidx in valid_vars and not ignore_valid): ret[i] = None
      elif tidx not in idx_vars: ret[i] = 0
    return tuple(ret)

  def unit_stride_axes(self, ignore_valid=False) -> List[int]: return [i for i,st in enumerate(self.real_strides(ignore_valid)) if st == 1]

  def expr_idxs(self, idxs:Optional[Iterable[Node]]=None) -> Tuple[Node, Node]:
    idxs = [Variable(f"idx{i}", 0, s-1) for i,s in enumerate(self.shape)] if idxs is None else list(idxs)
    idx, valid = _expr_view(self.views[-1], idxs)
    for view in reversed(self.views[0:-1]):
      if valid.max == 0: return NumNode(-1), valid
      view = view.minify()
      acc, idxs = 1, []
      for d in reversed(view.shape):
        idxs.append((idx//acc)%d)
        acc *= d
      idx, valid = _expr_view(view, idxs[::-1], valid)
    return idx, valid

  def axis_is_masked(self, axis:int) -> bool:
    _, valid = self.expr_idxs()
    return f'idx{axis}' in [v.expr for v in valid.vars()]

  def simplify(self) -> ShapeTracker:
    if len(self.views) >= 2 and (new_view := merge_views(self.views[-2], self.views[-1])) is not None:
      return ShapeTracker(self.views[:-2] + (new_view,)).simplify()
    return self

  # *** under this line are the movement ops ***

  def pad(self, arg: Tuple[Tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].pad(arg), ))
  def shrink(self, arg: Tuple[Tuple[sint, sint], ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].shrink(arg), ))
  def expand(self, new_shape: Tuple[sint, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].expand(new_shape), ))
  def permute(self, axis: Tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].permute(axis), ))
  def stride(self, mul: Tuple[int, ...]) -> ShapeTracker: return ShapeTracker(self.views[0:-1] + (self.views[-1].stride(mul), ))

  def reshape(self, new_shape: Tuple[sint, ...]) -> ShapeTracker:
    if getenv("MERGE_VIEW", 1) and (new_view := self.views[-1].reshape(new_shape)) is not None: return ShapeTracker(self.views[0:-1] + (new_view,))
    return ShapeTracker(self.views + (View.create(new_shape), ))


# tinygrad/shape/symbolic.py
  
from __future__ import annotations
import functools
from math import gcd
from tinygrad.helpers import partition
from typing import List, Dict, Callable, Tuple, Type, Union, Optional, Any, Set, Mapping

# NOTE: Python has different behavior for negative mod and floor div than c
# symbolic matches the Python behavior, but the code output is agnostic, and will never have negative numbers in div or mod

class Node:
  b: Union[Node, int]
  min: int
  max: sint
  def render(self, ops=None, ctx=None) -> Any:
    if ops is None: ops = render_python
    assert self.__class__ in (Variable, NumNode) or self.min != self.max
    return ops[type(self)](self, ops, ctx)
  def vars(self) -> Set[Variable]: return set()
  # substitute Variables with the values in var_vals
  def substitute(self, var_vals: Mapping[Variable, Union[NumNode, Variable]]) -> Node: raise RuntimeError(self.__class__.__name__)
  def unbind(self) -> Tuple[Node, Optional[int]]: return self.substitute({v: v.unbind()[0] for v in self.vars() if v.val is not None}), None

  @functools.cached_property
  def key(self) -> str: return self.render(ctx="DEBUG")
  @functools.cached_property
  def hash(self) -> int: return hash(self.key)
  def __repr__(self): return self.render(ctx="REPR")
  def __str__(self): return "<"+self.key+">"
  def __hash__(self): return self.hash
  def __bool__(self): return not (self.max == self.min == 0)
  def __eq__(self, other:object) -> bool:
    if not isinstance(other, Node): return NotImplemented
    return self.key == other.key
  def __neg__(self): return self*-1
  def __add__(self, b:Union[Node,int]): return Node.sum([self, NumNode(b) if isinstance(b, int) else b])
  def __radd__(self, b:int): return self+b
  def __sub__(self, b:Union[Node,int]): return self+-b
  def __rsub__(self, b:int): return -self+b
  def __le__(self, b:Union[Node,int]): return self < (b+1)
  def __gt__(self, b:Union[Node,int]): return (-self) < (-b)
  def __ge__(self, b:Union[Node,int]): return (-self) < (-b+1)
  def __lt__(self, b:Union[Node,int]): return create_node(LtNode(self, b))
  def __mul__(self, b:Union[Node, int]):
    if b == 0: return NumNode(0)
    if b == 1: return self
    return create_node(MulNode(self, b.b)) if isinstance(b, NumNode) else create_node(MulNode(self, b))
  def __rmul__(self, b:int): return self*b

  # *** complex ops ***

  def __rfloordiv__(self, b:int): return NumNode(b) // self
  def __floordiv__(self, b:Union[Node,int], factoring_allowed=True):
    if isinstance(b, Node):
      if b.__class__ is NumNode: return self // b.b
      if self == b: return NumNode(1)
      if (b - self).min > 0 and self.min >= 0: return NumNode(0) # b - self simplifies the node
      raise RuntimeError(f"not supported: {self} // {b}")
    assert b != 0
    if b < 0: return (self*-1)//-b
    if b == 1: return self

    # the numerator of div is not allowed to be negative
    if self.min < 0:
      offset = self.min//b
      # factor out an "offset" to make the numerator positive. don't allowing factoring again
      return (self + -offset*b).__floordiv__(b, factoring_allowed=False) + offset
    return create_node(DivNode(self, b))

  def __rmod__(self, b:int): return NumNode(b) % self
  def __mod__(self, b:Union[Node,int]):
    if isinstance(b, Node):
      if b.__class__ is NumNode: return self % b.b
      if self == b: return NumNode(0)
      if (b - self).min > 0 and self.min >= 0: return self # b - self simplifies the node
      raise RuntimeError(f"not supported: {self} % {b}")
    assert b > 0
    if b == 1: return NumNode(0)
    if isinstance(self.max, int) and isinstance(self.min, int):
      if self.min >= 0 and self.max < b: return self
      if (self.min//b) == (self.max//b): return self - (b*(self.min//b))
      if self.min < 0: return (self - ((self.min//b)*b)) % b
    return create_node(ModNode(self, b))

  @staticmethod
  def sum(nodes:List[Node]) -> Node:
    nodes = [x for x in nodes if x.max or x.min]
    if not nodes: return NumNode(0)
    if len(nodes) == 1: return nodes[0]

    mul_groups: Dict[Node, int] = {}
    num_node_sum = 0
    for node in SumNode(nodes).flat_components:
      if node.__class__ is NumNode: num_node_sum += node.b
      elif node.__class__ is MulNode: mul_groups[node.a] = mul_groups.get(node.a, 0) + node.b
      else: mul_groups[node] = mul_groups.get(node, 0) + 1
    new_nodes = [MulNode(a, b_sum) if b_sum != 1 else a for a, b_sum in mul_groups.items() if b_sum != 0]
    if num_node_sum: new_nodes.append(NumNode(num_node_sum))
    return create_node(SumNode(new_nodes)) if len(new_nodes) > 1 else new_nodes[0] if len(new_nodes) == 1 else NumNode(0)

  @staticmethod
  def ands(nodes:List[Node]) -> Node:
    if not nodes: return NumNode(1)
    if len(nodes) == 1: return nodes[0]
    if any(not x for x in nodes): return NumNode(0)

    # filter 1s
    nodes = [x for x in nodes if x.min != x.max]
    return create_node(AndNode(nodes)) if len(nodes) > 1 else (nodes[0] if len(nodes) == 1 else NumNode(1))

# 4 basic node types

class Variable(Node):
  def __new__(cls, *args):
    expr, nmin, nmax = args
    assert nmin >= 0 and nmin <= nmax, f"invalid Variable {expr=} {nmin=} {nmax=}"
    if nmin == nmax: return NumNode(nmin)
    return super().__new__(cls)

  def __getnewargs__(self): return (self.expr, self.min, self.max)  # args passed to __new__ when unpickling

  def __init__(self, expr:str, nmin:int, nmax:sint):
    self.expr, self.min, self.max = expr, nmin, nmax
    self._val: Optional[int] = None
  @property
  def val(self):
    assert self._val is not None, f"Variable isn't bound, can't access val of {self}"
    return self._val
  def bind(self, val):
    assert self._val is None and self.min<=val<=self.max, f"cannot bind {val} to {self}"
    self._val = val
    return self
  def unbind(self) -> Tuple[Variable, int]:
    assert self.val is not None, f"cannot unbind {self}"
    return Variable(self.expr, self.min, self.max), self.val
  def vars(self): return {self}
  def substitute(self, var_vals: Mapping[Variable, Union[NumNode, Variable]]) -> Node: return var_vals.get(self, self)

class NumNode(Node):
  def __init__(self, num:int):
    assert isinstance(num, int), f"{num} is not an int"
    self.b:int = num
    self.min, self.max = num, num
  def bind(self, val):
    assert self.b == val, f"cannot bind {val} to {self}"
    return self
  def __mul__(self, b:Union[Node,int]): return NumNode(self.b*b) if isinstance(b, int) else b*self.b
  def __eq__(self, other): return self.b == other
  def __hash__(self): return hash(self.b)  # needed with __eq__ override
  def substitute(self, var_vals: Mapping[Variable, Union[NumNode, Variable]]) -> Node: return self

def create_node(ret:Node):
  assert ret.min <= ret.max, f"min greater than max! {ret.min} {ret.max} when creating {type(ret)} {ret}"
  if ret.min == ret.max: return NumNode(ret.min)
  return ret

def create_lt_node(lhs:Node, b:Union[Node, int]):
  if isinstance(lhs, SumNode):
    if isinstance(b, int):
      new_sum = []
      for x in lhs.nodes:
        # TODO: should we just force the last one to always be the number
        if isinstance(x, NumNode): b -= x.b
        else: new_sum.append(x)
      lhs = Node.sum(new_sum)
      nodes = lhs.nodes if isinstance(lhs, SumNode) else [lhs]
      assert all(not isinstance(node, MulNode) or isinstance(node.b, int) for node in nodes), "not supported"
      muls, others = partition(nodes, lambda x: isinstance(x, MulNode) and x.b > 0 and x.max >= b)
      if muls:
        # NOTE: gcd in python 3.8 takes exactly 2 args
        mul_gcd = b
        for x in muls: mul_gcd = gcd(mul_gcd, x.b)  # type: ignore  # mypy cannot tell that x.b is int here due to assert above
        all_others = Node.sum(others)
        if all_others.min >= 0 and all_others.max < mul_gcd:
          lhs, b = Node.sum([mul//mul_gcd for mul in muls]), b//mul_gcd
    return create_node(LtNode(lhs, b)) if isinstance(lhs, SumNode) else create_lt_node(lhs, b)
  if isinstance(lhs, MulNode):
    if isinstance(b, Node) or isinstance(lhs.b, Node) or lhs.b == -1: return create_node(LtNode(lhs, b))
    sgn = 1 if lhs.b > 0 else -1
    return create_node(LtNode(lhs.a*sgn, (b + abs(lhs.b) - 1)//abs(lhs.b)))
  return create_node(LtNode(lhs, b))

def create_ge_node(lhs:Node, b:Union[Node, int]): return create_lt_node(-lhs, -b+1)

class OpNode(Node):
  def __init__(self, a:Node, b:Union[Node, int]):
    self.a, self.b = a, b
    self.min, self.max = self.get_bounds()
  def vars(self): return self.a.vars() | (self.b.vars() if isinstance(self.b, Node) else set())
  def get_bounds(self) -> Tuple[int, sint]: raise NotImplementedError("must be implemented")

class LtNode(OpNode):
  def get_bounds(self) -> Tuple[int, int]:
    if self.a == self.b: return (0, 0)
    if isinstance(self.b, int): return (1, 1) if self.a.max < self.b else (0, 0) if self.a.min >= self.b else (0, 1)
    return (1, 1) if self.a.max < self.b.min else (0, 0) if self.a.min >= self.b.max else (0, 1)
  def substitute(self, var_vals: Mapping[Variable, Union[NumNode, Variable]]) -> Node:
    return create_lt_node(self.a.substitute(var_vals), (self.b if isinstance(self.b, int) else self.b.substitute(var_vals)))

class MulNode(OpNode):
  def __mul__(self, b: Union[Node, int]): return self.a*(self.b*b) # two muls in one mul
  def __floordiv__(self, b: Union[Node, int], factoring_allowed=False): # NOTE: mod negative isn't handled right
    if self.b % b == 0: return self.a*(self.b//b)
    if b % self.b == 0 and self.b > 0: return self.a//(b//self.b)
    return Node.__floordiv__(self, b, factoring_allowed)
  def __mod__(self, b: Union[Node, int]): return Node.__mod__(self.a * (self.b%b), b)
  def get_bounds(self) -> Tuple[int, sint]:
    assert self.a.min >= 0
    if isinstance(self.b, int): return (self.a.min*self.b, self.a.max*self.b) if self.b >= 0 else (self.a.max*self.b, self.a.min*self.b)
    return (self.a.min*self.b.min, self.a.max*self.b.max) if self.b.min >= 0 else (self.a.max*self.b.min, self.a.min*self.b.max)
  def substitute(self, var_vals: Mapping[Variable, Union[NumNode, Variable]]) -> Node:
    return self.a.substitute(var_vals) * (self.b if isinstance(self.b, int) else self.b.substitute(var_vals))

class DivNode(OpNode):
  def __floordiv__(self, b: Union[Node, int], _=False): return self.a//(self.b*b) # two divs is one div
  def get_bounds(self) -> Tuple[int, sint]:
    assert self.a.min >= 0 and isinstance(self.b, int)
    return self.a.min//self.b, self.a.max//self.b
  def substitute(self, var_vals: Mapping[Variable, Union[NumNode, Variable]]) -> Node: return self.a.substitute(var_vals) // self.b

class ModNode(OpNode):
  def __mod__(self, b: Union[Node, int]):
    if isinstance(b, int) and isinstance(self.b, int) and self.b % b == 0: return self.a % b
    return Node.__mod__(self, b)
  def __floordiv__(self, b: Union[Node, int], factoring_allowed=True):
    return (self.a//b) % (self.b//b) if self.b % b == 0 else Node.__floordiv__(self, b, factoring_allowed)
  def get_bounds(self) -> Tuple[int, sint]:
    assert self.a.min >= 0 and isinstance(self.b, int)
    if self.a.max - self.a.min >= self.b or (self.a.min != self.a.max and self.a.min%self.b >= self.a.max%self.b): return (0, self.b-1)
    return (self.a.min%self.b, self.a.max%self.b)
  def substitute(self, var_vals: Mapping[Variable, Union[NumNode, Variable]]) -> Node: return self.a.substitute(var_vals) % self.b

class RedNode(Node):
  def __init__(self, nodes:List[Node]):
    self.nodes = nodes
    self.min, self.max = self.get_bounds()
  def vars(self) -> Set[Variable]: return set.union(*[x.vars() for x in self.nodes], set())
  def get_bounds(self) -> Tuple[int, sint]: raise NotImplementedError("must be implemented")

class SumNode(RedNode):
  def get_bounds(self) -> Tuple[int, sint]: return sum([x.min for x in self.nodes]), sum([x.max for x in self.nodes])
  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def __mul__(self, b: Union[Node, int]): return Node.sum([x*b for x in self.nodes]) # distribute mul into sum
  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def __floordiv__(self, b: Union[Node, sint], factoring_allowed=True):
    if self == b: return NumNode(1)
    fully_divided: List[Node] = []
    rest: List[Node] = []
    if isinstance(b, Node):
      for x in self.flat_components:
        if x % b == 0: fully_divided.append(x // b)
        else: rest.append(x)
      if (sum_fully_divided:=create_node(SumNode(fully_divided))) != 0: return sum_fully_divided + create_node(SumNode(rest)) // b
      return Node.__floordiv__(self, b, False)
    if b == 1: return self
    if not factoring_allowed: return Node.__floordiv__(self, b, factoring_allowed)
    _gcd = b
    divisor = 1
    for x in self.flat_components:
      if x.__class__ in (NumNode, MulNode):
        if x.b%b == 0: fully_divided.append(x//b)
        else:
          rest.append(x)
          if isinstance(x.b, int):
            _gcd = gcd(_gcd, x.b)
            if x.__class__ == MulNode and divisor == 1 and b%x.b == 0: divisor = x.b
          else:
            _gcd = 1
      else:
        rest.append(x)
        _gcd = 1
    if _gcd > 1: return Node.sum(fully_divided) + Node.sum(rest).__floordiv__(_gcd) // (b//_gcd)
    if divisor > 1: return Node.sum(fully_divided) + Node.sum(rest).__floordiv__(divisor) // (b//divisor)
    return Node.sum(fully_divided) + Node.__floordiv__(Node.sum(rest), b)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def __mod__(self, b: Union[Node, int]):
    if self == b: return NumNode(0)
    if isinstance(b, Node) and (b - self).min > 0: return self # b - self simplifies the node
    new_sum = Node.sum([node%b if node.__class__ in (NumNode, MulNode) else node for node in self.nodes])
    return Node.__mod__(new_sum, b)

  def substitute(self, var_vals: Mapping[Variable, Union[NumNode, Variable]]) -> Node:
    return Node.sum([node.substitute(var_vals) for node in self.nodes])

  # recursively expand sumnode components
  # TODO: can remove this if there's no SumNode inside SumNode
  @property
  def flat_components(self): return [y for x in self.nodes for y in (x.flat_components if isinstance(x, SumNode) else [x])]

class AndNode(RedNode):
  def get_bounds(self) -> Tuple[int, sint]: return min([x.min for x in self.nodes]), max([x.max for x in self.nodes])
  def substitute(self, var_vals: Mapping[Variable, Union[NumNode, Variable]]) -> Node:
    subed = []
    for node in self.nodes:
      if not (sub:=node.substitute(var_vals)): return NumNode(0)
      subed.append(sub)
    return Node.ands(subed)

def sym_render(a: Union[Node, int], ops=None, ctx=None) -> str: return str(a) if isinstance(a, int) else a.render(ops, ctx)
def sym_infer(a: Union[Node, int], var_vals: Dict[Variable, int]) -> int:
  if isinstance(a, (int, float)): return a
  ret = a.substitute({k:NumNode(v) for k, v in var_vals.items()})
  assert isinstance(ret, NumNode), f"sym_infer didn't produce NumNode from {a} with {var_vals}"
  return ret.b

# symbolic int, these are allowed in a Tensor shape
sint = Union[int, Variable, MulNode, SumNode]

def render_mulnode(node:MulNode, ops, ctx):
  # TODO: add ProdNode and remove this case
  if isinstance(node.a,Variable) and isinstance(node.b,Variable) and node.a.expr and node.b.expr and node.b.expr < node.a.expr:
    return f"({sym_render(node.b,ops,ctx)}*{node.a.render(ops,ctx)})"
  return f"({node.a.render(ops,ctx)}*{sym_render(node.b,ops,ctx)})"

render_python: Dict[Type, Callable[..., str]] = {
  Variable: lambda self,ops,ctx: f"{self.expr}[{self.min}-{self.max}{'='+str(self.val) if self._val is not None else ''}]" if ctx == "DEBUG" \
    else (f"Variable('{self.expr}', {self.min}, {self.max})"+(f".bind({self.val})" if self._val is not None else '') if ctx == "REPR" \
    else f"{self.expr}"),
  NumNode: lambda self,ops,ctx: f"NumNode({self.b})" if ctx == "REPR" else f"{self.b}",
  MulNode: render_mulnode,
  DivNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}//{self.b})",
  ModNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}%{self.b})",
  LtNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}<{sym_render(self.b,ops,ctx)})",
  SumNode: lambda self,ops,ctx: f"({'+'.join(sorted([x.render(ops,ctx) for x in self.nodes]))})",
  AndNode: lambda self,ops,ctx: f"({' and '.join(sorted([x.render(ops,ctx) for x in self.nodes]))})",
}


# tinygrad/shape/view.py

from __future__ import annotations
import functools, operator, itertools
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Set, cast
from tinygrad.helpers import prod, all_int, argsort
from tinygrad.shape.symbolic import Node, NumNode, Variable, sint

@functools.lru_cache(maxsize=None)
def canonicalize_strides(shape:Tuple[sint, ...], strides:Tuple[sint, ...]) -> Tuple[sint, ...]:
  return tuple(0 if s == 1 else st for s, st in zip(shape, strides))

@functools.lru_cache(maxsize=None)
def strides_for_shape(shape:Tuple[sint, ...]) -> Tuple[sint, ...]:
  if not shape: return ()
  strides = tuple(itertools.accumulate(reversed(shape[1:]), operator.mul, initial=1))
  return canonicalize_strides(shape, strides[::-1])

@functools.lru_cache(maxsize=None)
def _merge_dims(shape:Tuple[int, ...], strides:Tuple[int, ...], mask:Optional[Tuple[Tuple[int, int], ...]]=None) -> Tuple[Tuple[int, int, int], ...]:
  # merge contiguous subparts or zero strided dims. ret = List[(merged_dims, stride, merged dims w/o zero stride), ...]
  if not shape: return tuple()
  assert len(shape) == len(strides)
  ret = [(shape[0], strides[0], shape[0] if strides[0] else 0)]
  # wrt merging zero strided dimensions
  merging = strides[0] == 0 and (mask[0][1] - mask[0][0] == 1 if mask else shape[0] == 1)
  for i, (sh, st) in enumerate(zip(shape[1:], strides[1:]), start=1):
    if sh == 1: continue
    if merging or ret[-1][1] == sh * st: # mergeable
      ret[-1] = (ret[-1][0] * sh, st, (sh if merging else ret[-1][2] * sh) if st else 0)
    else: ret.append((sh, st, sh if st else 0)) # begin new
    # merging ends with either non-zero strided dim or zero strided dim with mask range > 1
    merging = st == 0 and (mask[i][1] - mask[i][0] == 1 if mask else sh == 1)
  return tuple(ret)

@functools.lru_cache(maxsize=None)
def _reshape_mask(view: View, new_shape:Tuple[sint, ...]) -> Tuple[Optional[Tuple[Tuple[sint, sint], ...]], bool]:
  if view.mask is None: return view.mask, False
  if any(not isinstance(m[0], int) or not isinstance(m[1], int) for m in view.mask): return view.mask, True
  new_mask: List[Tuple[int, int]] = []

  r_masks, r_shape, r_new_shape = reversed(view.mask), reversed(view.shape), reversed(new_shape)
  curr_stride, old_dim, new_dim, mask = 1, next(r_shape, 1), next(r_new_shape, 1), next(r_masks, (0,1))
  if mask[1] - mask[0] < 1: return ((0, 0),) * len(new_shape), False # invalid mask

  while len(new_mask) < len(new_shape):
    (l, r), next_stride = mask, new_dim * curr_stride

    if old_dim >= next_stride: # need to split mask.
      if old_dim == next_stride: # simply copy the mask and get next batch for merging
        new_mask.append((l // curr_stride, (r - 1) // curr_stride + 1))
        curr_stride, old_dim, new_dim, mask = 1, next(r_shape, 1), next(r_new_shape, 1), next(r_masks, (0,1))
        if mask[1] - mask[0] < 1: return ((0, 0),) * len(new_shape), False # invalid mask

      else: # mask can only be splitted if reshape doesn't cut across the mask.
        if ((l % next_stride != 0 or r % next_stride != 0) and l // next_stride != (r - 1) // next_stride): return view.mask, True
        new_mask.append((l % next_stride // curr_stride, (r - 1) % next_stride // curr_stride + 1))
        curr_stride, new_dim = next_stride,  next(r_new_shape, 1) # need to get mask for next dimension

    else:
      next_mask = next(r_masks, (0, 1))
      # combine if the mask can unfold continuously
      if mask != (0, old_dim) and next_mask[1] - next_mask[0] != 1: return view.mask, True
      mask, old_dim = (next_mask[0] * old_dim + l, (next_mask[1] - 1) * old_dim + r), old_dim * next(r_shape, 1)

  for mask in r_masks: # if the old shape has leading 1s, need to make sure their mask is (0,1)
    if mask != (0, 1): return ((0, 0),) * len(new_shape), False # invalid mask

  return tuple(reversed(new_mask)), False

@dataclass(frozen=True)
class View:
  shape:Tuple[sint, ...]
  strides:Tuple[sint, ...]
  offset:sint
  mask:Optional[Tuple[Tuple[sint, sint], ...]]
  contiguous:bool

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def size(self) -> int:
    # NOTE: Variable and the Node derived from it in symbolic shapes can only have int as max.
    ret = prod([x.max if isinstance(x, Node) else x for x in self.shape])
    assert isinstance(ret, int), f"{ret=} is not int"
    return ret

  @staticmethod
  @functools.lru_cache(maxsize=None)
  def create(shape:Tuple[sint, ...], strides:Optional[Tuple[sint, ...]]=None, offset:sint=0, mask:Optional[Tuple[Tuple[sint, sint], ...]]=None):
    strides = canonicalize_strides(shape, strides) if strides else strides_for_shape(shape)
    # canonicalize empty mask
    if mask is not None and all(m == (0,s) for m,s in zip(mask, shape)): mask = None
    contiguous = offset == 0 and mask is None and strides == strides_for_shape(shape)
    # if any dimension has size >1, but is masked such that only one index in the dimension is unmasked
    # then its stride can also be set to 0, albeit with a corresponding adjustment required to the offset
    # TODO: assert comparison with LtNode to avoid mis-using symbolic
    if mask and any(elim := [not (b+1 < e) for b,e in mask]):
      if any(not (b < e) for b,e in mask):
        strides, offset, mask = (0,) * len(shape), 0, ((0,0),) * len(shape)
      offset += sum((strides[i] * mask[i][0]) if e else 0 for i, e in enumerate(elim))
      strides = tuple(0 if e else st for st,e in zip(strides, elim))
    return View(shape, strides, offset, mask, contiguous)

  @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  def vars(self) -> Set[Variable]:
    flatten_mask = tuple(x for m in self.mask for x in m) if self.mask is not None else tuple()
    return functools.reduce(operator.or_, [x.vars() for x in self.shape+self.strides+(self.offset,)+flatten_mask if isinstance(x, Node)], set())

  @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  def unbind(self) -> Tuple[View, Dict[Variable, int]]:
    var_unboundvar_val = [(v, v.unbind()) for v in self.vars() if v.val is not None]
    unbound_vars = {v:uv for v,(uv,_) in var_unboundvar_val}
    new_shape = tuple([s if isinstance(s, int) else s.substitute(unbound_vars) for s in self.shape])
    new_strides = tuple([s if isinstance(s, int) else s.substitute(unbound_vars) for s in self.strides])
    new_offset = self.offset if isinstance(self.offset, int) else self.offset.substitute(unbound_vars)
    new_mask = tuple((a if isinstance(a, int) else a.substitute(unbound_vars),
                      b if isinstance(b, int) else b.substitute(unbound_vars)) for (a, b) in self.mask) if self.mask is not None else None
    return View.create(new_shape, new_strides, new_offset, new_mask), dict(x[1] for x in var_unboundvar_val)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def invert(self, out_shape:Tuple[sint, ...]) -> Optional[View]:
    ret = View.create(self.shape)
    if self.mask: ret = ret.shrink(self.mask)
    ret = ret.stride(tuple(-1 if x < 0 else 1 for x in self.strides)).permute(argsort(tuple(-x if x > 0 else x for x in self.strides)))
    return ret if prod(ret.shape) == prod(out_shape) else None   # don't support shrink, expand, or stride != (-1, 1)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def minify(self):
    min_shape = tuple(x[0] for x in _merge_dims(self.shape, self.strides, self.mask))
    return nv if (nv := self.reshape(min_shape)) else self

  def __unsafe_resize(self, arg: Tuple[Tuple[sint, sint], ...], mask=None) -> View:
    offset = sum([s * x[0] for s, x in zip(self.strides,arg)])
    if self.mask:
      # move the old mask
      nmask = tuple([(max(0, min(mx-ax,ay-ax)), max(0, min(my-ax,ay-ax))) for (mx,my),(ax,ay) in zip(self.mask, arg)])
      # merge the masks if we have two
      mask = tuple([(max(mx1, mx2), min(my1, my2)) for (mx1, my1), (mx2, my2) in zip(nmask, mask)]) if mask is not None else nmask
    shape = [y-x for x,y in arg]
    if mask is not None and all(m[0] == 0 and m[1] == s for m,s in zip(mask, shape)): mask = None
    return View.create(tuple(s.b if isinstance(s, NumNode) else s for s in shape), self.strides, self.offset+offset, mask)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def pad(self, arg: Tuple[Tuple[int, int], ...]) -> View:
    assert all((b>=0 and e>=0) for b,e in arg) and len(arg) == len(self.shape), f"{self.shape=}, {arg=}"
    if any(b or e for b, e in arg):
      zvarg = tuple([(-b,s+e) for s,(b,e) in zip(self.shape, arg)])
      mask = tuple([(b,s+b) for s,(b,_) in zip(self.shape, arg)])
      return self.__unsafe_resize(zvarg, mask=mask)
    return self

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def shrink(self, arg: Tuple[Tuple[sint, sint], ...]) -> View:
    assert all((0<=b<=e<=s) for s,(b,e) in zip(self.shape,arg)) and len(arg) == len(self.shape), f"invalid shrink {arg} for {self.shape}"
    return self.__unsafe_resize(arg)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def expand(self, new_shape: Tuple[sint, ...]) -> View:
    if len(new_shape) != len(self.shape): raise ValueError(f"expand arg {new_shape=} must have same number of dimensions as shape {self.shape=}")
    if 0 in self.shape:
      assert all((s == x == 0) or (s > 0 and (x % s) == 0) for s,x in zip(self.shape, new_shape)), f"can't expand {self.shape} into {new_shape}"
      return View.create(new_shape)
    assert all((s == x or (s == 1 and st == 0)) for s,x,st in zip(self.shape, new_shape, self.strides)), f"can't expand {self.shape} into {new_shape}"
    # NOTE: can the mask ever be (0,0)?
    mask = tuple([(((0,0) if m != (0,1) else (0,ns)) if s != ns else m) for m,s,ns in zip(self.mask, self.shape, new_shape)]) if self.mask else None
    return View.create(new_shape, self.strides, self.offset, mask)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def permute(self, axis: Tuple[int, ...]) -> View:
    assert all(isinstance(x, int) and x >= 0 and x < len(self.shape) for x in axis), f"invalid permute {axis} for {self.shape}"
    assert len(set(axis)) == len(axis) and len(axis) == len(self.shape), f"can't permute {self.shape} with {axis}"
    return View.create(tuple(self.shape[a] for a in axis), tuple(self.strides[a] for a in axis), self.offset,
                       tuple(self.mask[a] for a in axis) if self.mask is not None else None)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def stride(self, mul: Tuple[int, ...]) -> View:
    # except for the negative case, you can build this from the others. invertible in the negative case
    assert all(isinstance(x, int) and x != 0 for x in mul), f"invalid stride {mul} for {self.shape}"
    strides = tuple([z*m for z,m in zip(self.strides, mul)])
    new_shape = tuple([(s+(abs(m)-1))//abs(m) for s,m in zip(self.shape, mul)])
    offset = sum([(s-1)*z for s,z,m in zip(self.shape, self.strides, mul) if m < 0])
    mask = tuple([(((mx if m > 0 else s-my)+(abs(m)-1))//abs(m), ((my if m > 0 else s-mx)+(abs(m)-1))//abs(m)) \
                  for (mx,my),s,m in zip(self.mask, self.shape, mul)]) if self.mask is not None else None
    return View.create(new_shape, strides, self.offset + offset, mask)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def reshape(self, new_shape: Tuple[sint, ...]) -> Optional[View]:
    if self.shape == new_shape: return self

    assert all(x >= 0 for x in new_shape), f"shape can't contain negative numbers {new_shape}"
    if 0 in self.shape:
      assert 0 in new_shape, f"cannot reshape 0 size to {new_shape}"
      return View.create(new_shape)
    # check for the same size
    if all_int(self.shape):
      assert all(isinstance(s, (int, Variable)) for s in new_shape), f"{self.shape=} -> {new_shape=} contains non (int, Variable) dim"
      if prod(self.shape) != prod([s if isinstance(s, int) else cast(Variable,s).val for s in new_shape]):
        raise ValueError(f"size mismatched, can't reshape {self.shape=} -> {new_shape=}")

    if new_shape == () and self.mask and any(mx==my for (mx,my) in self.mask): return None

    # after the asserts, it's okay to check contiguous
    if self.contiguous: return View.create(new_shape)

    strides, r_new_shape = [], reversed(new_shape)
    for merged_dim, new_stride, real_dim in reversed(_merge_dims(self.shape, self.strides, self.mask)):
      acc = 1
      # TODO: this <= and != is for symbolic!?
      while acc <= merged_dim and acc != merged_dim and (new_dim := next(r_new_shape, None)):
        strides.append(new_stride)
        if new_dim != 1: new_stride *= (new_dim if (acc :=  acc * new_dim) < real_dim else 0)
      if acc != merged_dim: break
    else:
      strides += [0,] * (len(new_shape) - len(strides))
      new_mask, extra = _reshape_mask(self, new_shape)
      if not extra:
        new_strides = canonicalize_strides(tuple(e-b for b,e in new_mask) if new_mask else new_shape, tuple(reversed(strides)))
        extra_offset = (sum(m[0] * s for m,s in zip(self.mask, self.strides)) if self.mask else 0) - \
                       (sum(m[0] * s for m,s in zip(new_mask, new_strides)) if new_mask else 0)
        return View.create(new_shape, new_strides, self.offset + extra_offset, new_mask)

    return None
