from allo._mlir.ir import MemRefType
from allo._mlir.dialects import affine as affine_d, func as func_d

from ...utils import get_bitwidth_from_type, get_dtype_and_shape_from_type
from .utils import allo_dtype_to_dslx_type

# ================================================================
# RAM template
# ================================================================
RAM_TEMPLATE = """\
// Simple dual-port RAM model with independent read and write ports.
// Parameterized on address width, data width, and depth.
// Reads observe the state before a concurrent write (read-before-write).
// Writes always update a full word.

pub struct SimpleReadReq<ADDR_WIDTH: u32> {
  addr: uN[ADDR_WIDTH],
}

pub struct SimpleReadResp<DATA_WIDTH: u32> {
  data: uN[DATA_WIDTH],
}

pub struct SimpleWriteReq<ADDR_WIDTH: u32, DATA_WIDTH: u32> {
  addr: uN[ADDR_WIDTH],
  data: uN[DATA_WIDTH],
}

pub struct SimpleWriteResp {}

pub proc Simple1R1WRam<ADDR_WIDTH: u32, DATA_WIDTH: u32, SIZE: u32> {
  read_req: chan<SimpleReadReq<ADDR_WIDTH>> in;
  read_resp: chan<SimpleReadResp<DATA_WIDTH>> out;
  write_req: chan<SimpleWriteReq<ADDR_WIDTH, DATA_WIDTH>> in;
  write_resp: chan<SimpleWriteResp> out;

  config(read_req: chan<SimpleReadReq<ADDR_WIDTH>> in,
         read_resp: chan<SimpleReadResp<DATA_WIDTH>> out,
         write_req: chan<SimpleWriteReq<ADDR_WIDTH, DATA_WIDTH>> in,
         write_resp: chan<SimpleWriteResp> out) {
    (read_req, read_resp, write_req, write_resp)
  }

  init { uN[DATA_WIDTH][SIZE]:[uN[DATA_WIDTH]:0, ...] }

  next(state: uN[DATA_WIDTH][SIZE]) {
    let (tok_r, r_req, r_valid) =
        recv_non_blocking(join(), read_req, zero!<SimpleReadReq<ADDR_WIDTH>>());
    let (tok_w, w_req, w_valid) =
        recv_non_blocking(join(), write_req, zero!<SimpleWriteReq<ADDR_WIDTH, DATA_WIDTH>>());

    let addr_r = r_req.addr as u32;
    let addr_w = w_req.addr as u32;

    let state_before_write = state;
    let state =
        if w_valid { update(state, addr_w, w_req.data) } else { state };

    let read_data = state_before_write[addr_r];
    send_if(
        tok_r, read_resp, r_valid,
        SimpleReadResp<DATA_WIDTH> { data: read_data });

    send_if(
        tok_w, write_resp, w_valid,
        SimpleWriteResp {});

    state
  }
}
"""


# ================================================================
# memory binding metadata
# ================================================================
class MemoryBinding:
  def __init__(self, value, dtype, shape, index, arg_index=None):
    self.value = value
    self.key = str(value)
    self.dtype = dtype
    self.dslx_type = allo_dtype_to_dslx_type(dtype)
    self.shape = tuple(shape)
    self.data_width = get_bitwidth_from_type(dtype)
    self.size = 1
    for dim in self.shape:
      self.size *= dim
    self.addr_width = max(1, (self.size - 1).bit_length()) if self.size else 1
    self.base = f"mem{index}"
    self.read_req_chan = f"{self.base}__read_req"
    self.read_resp_chan = f"{self.base}__read_resp"
    self.write_req_chan = f"{self.base}__write_req"
    self.write_resp_chan = f"{self.base}__write_resp"
    self.arg_index = arg_index
    self.needs_read = False
    self.needs_write = False
    self.strides = self._compute_strides()

  def addr_param(self):
    return f"u32:{self.addr_width}"

  def data_param(self):
    return f"u32:{self.data_width}"

  def _compute_strides(self):
    if not self.shape:
      return []
    reversed_strides = []
    running = 1
    for dim in reversed(self.shape):
      reversed_strides.append(running)
      running *= dim
    return list(reversed(reversed_strides))

  def linearize(self, index_exprs):
    if not index_exprs:
      return "s32:0"
    if len(index_exprs) != len(self.shape):
      raise NotImplementedError("partial indexing of memories is not supported")
    terms = []
    for expr, stride in zip(index_exprs, self.strides):
      if stride == 1:
        terms.append(expr)
      else:
        terms.append(f"({expr} * s32:{stride})")
    if not terms:
      return "s32:0"
    if len(terms) == 1:
      return terms[0]
    return "(" + " + ".join(terms) + ")"


# ================================================================
# memory discovery helpers
# ================================================================
def discover_memory_bindings(func, channel_arg_keys={}):
  func_args = [
      arg for arg in func.arguments
      if "!allo.stream" not in str(arg.type)
  ]
  bindings = []
  memory_map = {}
  idx = 0
  for arg_index, arg in enumerate(func_args):
    if not MemRefType.isinstance(arg.type):
      continue
    if channel_arg_keys and str(arg) in channel_arg_keys:
      continue
    mt = MemRefType(arg.type)
    shape = tuple(mt.shape)
    if not shape:
      continue
    if any(dim == -1 for dim in shape):
      raise NotImplementedError("dynamic memrefs are not supported yet")
    elem_dtype, _ = get_dtype_and_shape_from_type(mt.element_type)
    binding = MemoryBinding(arg, elem_dtype, shape, idx, arg_index)
    bindings.append(binding)
    memory_map[binding.key] = binding
    idx += 1
  idx = _discover_return_memrefs(func, bindings, memory_map, idx)
  if bindings:
    _mark_memory_usage(func, memory_map)
  return bindings, memory_map


def _mark_memory_usage(func, memory_map):
  def visit_block(block):
    for op in block.operations:
      if isinstance(op, affine_d.AffineLoadOp):
        binding = memory_map.get(str(op.operands[0]))
        if binding:
          binding.needs_read = True
      elif isinstance(op, affine_d.AffineStoreOp):
        binding = memory_map.get(str(op.operands[1]))
        if binding:
          binding.needs_write = True
      for region in op.regions:
        for blk in region.blocks:
          visit_block(blk)

  for block in func.body.blocks:
    visit_block(block)


def _discover_return_memrefs(func, bindings, memory_map, start_idx):
  idx = start_idx
  for block in func.body.blocks:
    for op in block.operations:
      if isinstance(op, func_d.ReturnOp):
        for operand in op.operands:
          if not MemRefType.isinstance(operand.type):
            continue
          key = str(operand)
          if key in memory_map:
            continue
          mt = MemRefType(operand.type)
          shape = tuple(mt.shape)
          if not shape:
            continue
          if any(dim == -1 for dim in shape):
            raise NotImplementedError("dynamic memrefs are not supported yet")
          elem_dtype, _ = get_dtype_and_shape_from_type(mt.element_type)
          binding = MemoryBinding(operand, elem_dtype, shape, idx)
          binding.needs_write = True
          bindings.append(binding)
          memory_map[key] = binding
          idx += 1
        return idx
  return idx


def discover_memory_bindings_pe(func):
  # collect all memref access summaries up front
  reads, writes, index_patterns = _collect_memref_accesses(func)

  # iterate args (skip stream args) and locate first memref first-dim
  func_args = [arg for arg in func.arguments if "!allo.stream" not in str(arg.type)]
  channel_size = None
  memref_args = []
  for arg in func_args:
    if not MemRefType.isinstance(arg.type):
      continue
    mt = MemRefType(arg.type)
    shape = tuple(mt.shape)
    if not shape:
      continue
    if any(dim == -1 for dim in shape):
      continue
    memref_args.append((arg, shape))
    if channel_size is None and len(shape) > 0:
      channel_size = shape[0]

  if channel_size is None:
    raise ValueError("no channels found for PE element")

  channel_infos = []
  channel_arg_keys = []
  for arg, shape in memref_args:
    if shape[0] != channel_size:  # not a channel
      continue
    key = str(arg)
    seen_reads = key in reads
    seen_writes = key in writes
    # channels must be read-only or write-only
    if seen_reads and seen_writes:
      continue
    
    chan_dir = 'in' if seen_reads and not seen_writes else ('out' if seen_writes and not seen_reads else None)
    if chan_dir is None:  # memory not used
      continue

    pats = index_patterns.get(key, [])
    # require at least one indexed access and conservative prefix check
    if not pats:
      continue
    pat_lens = {len(p) for p in pats}
    if len(pat_lens) != 1:
      continue
    plen = pat_lens.pop()
    if plen == 0:
      continue
    prefixes = {p[:-1] for p in pats}
    if len(prefixes) != 1:
      continue

    ci = {
      'key': key,
      'channel_name': f"mem{len(channel_arg_keys)}__chan",
      'chan_dir': chan_dir,
      'chan_dslx_type': allo_dtype_to_dslx_type(get_dtype_and_shape_from_type(MemRefType(arg.type).element_type)[0]),
      'index_patterns': pats,
    }
    channel_infos.append(ci)
    channel_arg_keys.append(key)

  return channel_infos, channel_arg_keys


def _collect_memref_accesses(func):
  # Track membership (read/write seen) with sets for cheap checks.
  reads = set()
  writes = set()
  index_patterns = {}

  def visit_block(block):
    for op in block.operations:
      if isinstance(op, affine_d.AffineLoadOp):
        key = str(op.operands[0])
        reads.add(key)
        idxs = tuple(str(o) for o in op.operands[1:])
        index_patterns.setdefault(key, []).append(idxs)
      elif isinstance(op, affine_d.AffineStoreOp):
        key = str(op.operands[1])
        writes.add(key)
        idxs = tuple(str(o) for o in op.operands[2:])
        index_patterns.setdefault(key, []).append(idxs)
      for region in op.regions:
        for blk in region.blocks:
          visit_block(blk)

  for block in func.body.blocks:
    visit_block(block)

  return reads, writes, index_patterns


# ================================================================
# memory emitter
# ================================================================
class MemoryEmitter:
  def __init__(self, parent):
    self.p = parent

  def emit_read(self, binding, op):
    lines = []
    index_exprs = self._index_exprs(op.operands, start=1)
    addr_expr = binding.linearize(index_exprs)
    addr_bits = self.p.new_tmp()
    lines.append(f"    let {addr_bits} = ({addr_expr} as uN[{binding.addr_width}]);")
    req_tmp = self.p.new_tmp()
    lines.append(f"    let {req_tmp} = "
                 f"SimpleReadReq<{binding.addr_param()}> {{ addr: {addr_bits} }};")
    tok = self.p.new_tok()
    lines.append(f"    let {tok} = send(join(), {binding.read_req_chan}, {req_tmp});")
    resp_tok = self.p.new_tok()
    resp_val = self.p.new_tmp()
    lines.append(f"    let ({resp_tok}, {resp_val}) = recv({tok}, {binding.read_resp_chan});")
    data_tmp = self.p.new_tmp()
    lines.append(f"    let {data_tmp} = ({resp_val}.data as {binding.dslx_type});")
    self.p.register(op.result, data_tmp)
    self.p.track_token(resp_tok)
    return lines

  def emit_write(self, binding, op, predicate=None):
    lines = []
    value_expr = self.p.lookup(op.operands[0])
    index_exprs = self._index_exprs(op.operands, start=2)
    addr_expr = binding.linearize(index_exprs)
    addr_bits = self.p.new_tmp()
    lines.append(f"    let {addr_bits} = ({addr_expr} as uN[{binding.addr_width}]);")
    data_bits = self.p.new_tmp()
    lines.append(f"    let {data_bits} = ({value_expr} as uN[{binding.data_width}]);")
    req_tmp = self.p.new_tmp()
    lines.append(
        "    let "
        f"{req_tmp} = SimpleWriteReq<{binding.addr_param()}, {binding.data_param()}> "
        f"{{ addr: {addr_bits}, data: {data_bits} }};")
    token_expr = self.p._consume_tokens("join()", lines)
    tok = self.p.new_tok()
    if predicate:
      # Use send_if/recv_if when predicate is provided
      lines.append(f"    let {tok} = send_if({token_expr}, {binding.write_req_chan}, {predicate}, {req_tmp});")
      ack_tok = self.p.new_tok()
      lines.append(f"    let ({ack_tok}, _) = recv_if({tok}, {binding.write_resp_chan}, {predicate}, zero!<SimpleWriteResp>());")
    else:
      # Unconditional write
      lines.append(f"    let {tok} = send({token_expr}, {binding.write_req_chan}, {req_tmp});")
      ack_tok = self.p.new_tok()
      lines.append(f"    let ({ack_tok}, _) = recv({tok}, {binding.write_resp_chan});")
    self.p.track_token(ack_tok)
    return lines

  def _index_exprs(self, operands, start):
    if len(operands) <= start:
      return []
    exprs = []
    for idx in operands[start:]:
      exprs.append(f"({self.p.lookup(idx)} as s32)")
    return exprs