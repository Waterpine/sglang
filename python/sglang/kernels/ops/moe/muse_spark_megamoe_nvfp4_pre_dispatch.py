"""Stage an NVFP4 sequence shard into DeepGEMM's MegaMoE symmetric buffer.

Byte-identical replacement for ``deep_gemm.mega_moe_pre_dispatch`` with
``mma_type="nvfp4xnvfp4"``, group size 16 and no expert scales. DeepGEMM runs
one 512-thread CTA per token, one 16-byte load per thread, two block barriers.
Here four warps share a token and each lane owns whole 16-element groups, so
the group amax and UE4M3 scale stay in registers; only the row amax crosses
lanes (warp shuffles, then one shared-memory step).

Arithmetic follows DeepGEMM's sm100 pre-dispatch kernel instruction for
instruction (inline PTX): row amax clamped at 1e-4, ``pts = amax / 2688``,
``sf = e4m3_rn_satfinite(group_amax / (6 * pts))``, ``inv = 1 / (sf * pts)``
(zero when ``sf <= 0`` or NaN), codes ``e2m1(v * inv)`` with DeepGEMM's
midpoint rules. The E2M1 encode uses ``cvt.rn.satfinite.e2m1x2``, which matches
DeepGEMM's bucketize on every FP32 input except NaN (forced to code 0 first)
and negative zero-magnitude results (sign cleared afterwards).
``fast_math=True`` mirrors a DeepGEMM build with ``--use_fast_math``
(approximate division and reciprocal, FTZ); the default matches the installed
DeepGEMM JIT build.
"""

from functools import cache

import torch

_WARPS = 4  # minimum warps per CTA
_PAD_UNROLL = 4
_WARPS_PER_TOKEN = 4
_FAST_MATH = False  # DeepGEMM's JIT build of the pre-dispatch kernel
_GROUP = 16  # NVFP4 elements per UE4M3 scale
_LANE_SPAN = 32 * _GROUP  # hidden elements one warp covers per group step


def can_pre_dispatch_nvfp4(
    x, ids, scores, buf_x, buf_x_sf, buf_x_scales, buf_ids, buf_scores
):
    """Whether this launch reproduces DeepGEMM's pre-dispatch for these tensors."""
    if x.ndim != 2 or ids.ndim != 2:
        return False
    n, hidden = x.shape
    capacity = buf_ids.shape[0] if buf_ids.ndim == 2 else -1
    return (
        x.is_cuda
        and torch.cuda.get_device_capability(x.device)[0] == 10
        and x.dtype == torch.bfloat16
        and ids.dtype == torch.int32
        and scores.dtype == torch.float32
        and buf_x.dtype == torch.uint8
        and buf_x_sf.dtype == torch.int32
        and buf_x_scales is not None
        and buf_x_scales.dtype == torch.float32
        and buf_ids.dtype == torch.int64
        and buf_scores.dtype == torch.float32
        and all(
            t.device == x.device and t.is_contiguous()
            for t in (
                x,
                ids,
                scores,
                buf_x,
                buf_x_sf,
                buf_x_scales,
                buf_ids,
                buf_scores,
            )
        )
        and 0 < n <= capacity
        and hidden in (512, 2048, 4096)
        and ids.shape == scores.shape == (n, buf_ids.shape[1])
        and 0 < ids.shape[1] <= 32
        and buf_scores.shape == buf_ids.shape
        and buf_x.shape == (capacity, hidden // 2)
        and buf_x_sf.shape == (capacity, hidden // (4 * _GROUP))
        and buf_x_scales.numel() >= capacity
        and buf_x.numel() < 2**31
        and buf_ids.numel() < 2**31
        and x.data_ptr() % 16 == 0
        and buf_x.data_ptr() % 8 == 0
    )


def pre_dispatch_nvfp4(
    x,
    ids,
    scores,
    buf_x,
    buf_x_sf,
    buf_x_scales,
    buf_ids,
    buf_scores,
    fast_math=_FAST_MATH,
    warps_per_token=_WARPS_PER_TOKEN,
):
    """Quantize rows [0, n) into the buffer and reset routing rows [n, capacity)."""
    groups_per_lane = x.shape[1] // _LANE_SPAN
    warps_per_token = min(warps_per_token, groups_per_lane)
    compiled = _compile(
        x.device, groups_per_lane, ids.shape[1], bool(fast_math), warps_per_token
    )
    # Flatten before reinterpreting: a contiguous single row may keep any row stride.
    compiled(
        x.view(-1).view(torch.int32).view(-1, 4),
        ids.view(-1),
        scores.view(-1),
        buf_x.view(-1).view(torch.int32).view(-1, 2),
        buf_x_sf.view(-1).view(torch.uint8),
        buf_x_scales.view(-1),
        buf_ids.view(-1),
        buf_scores.view(-1),
    )


@cache
def _compile(device, groups_per_lane, top_k, fast_math, warps_per_token):
    import cutlass
    import cutlass.cute as cute

    def fake(dtype, shape, align):
        order = tuple(range(len(shape) - 1, -1, -1))
        return cute.runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=order, assumed_align=align
        )

    return cute.compile(
        _kernel_type(groups_per_lane, top_k, fast_math, warps_per_token)(),
        fake(cutlass.Uint32, (cute.sym_int(), 4), 16),  # BF16 rows, 16 B vectors
        fake(cutlass.Int32, (cute.sym_int(),), 4),
        fake(cutlass.Float32, (cute.sym_int(),), 4),
        fake(cutlass.Uint32, (cute.sym_int(), 2), 8),  # FP4 rows, one group per row
        fake(cutlass.Uint8, (cute.sym_int(),), 1),
        fake(cutlass.Float32, (cute.sym_int(),), 4),
        fake(cutlass.Int64, (cute.sym_int(),), 8),
        fake(cutlass.Float32, (cute.sym_int(),), 4),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _ptx(fast_math):
    """PTX for DeepGEMM's FP32 steps, with or without --use_fast_math."""
    f = ".ftz" if fast_math else ""
    div = "div.approx.ftz.f32" if fast_math else "div.rn.f32"
    rcp = "rcp.approx.ftz.f32" if fast_math else "rcp.rn.f32"
    unpack = (
        "shl.b32 r, {w}, 16; mov.b32 a, r; and.b32 r, {w}, 0xFFFF0000; mov.b32 b, r; "
    )
    amax = "".join(
        unpack.format(w=f"${i + 1}")
        + f"abs{f}.f32 a, a; max{f}.f32 m, m, a; abs{f}.f32 b, b; max{f}.f32 m, m, b; "
        for i in range(8)
    )
    encode = "".join(
        unpack.format(w=f"${i + 1}") + f"mul{f}.f32 a, a, $5; mul{f}.f32 b, b, $5; "
        "setp.nan.f32 p, a, a; selp.f32 a, 0f00000000, a, p; "
        "setp.nan.f32 p, b, b; selp.f32 b, 0f00000000, b, p; "
        f"cvt.rn.satfinite.e2m1x2.f32 c{i}, b, a; "
        for i in range(4)
    )
    return dict(
        # Group amax of 16 BF16 values (8 words), starting from +0 like fmaxf.
        amax="{ .reg .b32 r; .reg .f32 a, b, m; mov.f32 m, 0f00000000; "
        + amax
        + "mov.f32 $0, m; }",
        # pts = max(row_amax, 1e-4) / (448 * 6)
        pts="{ .reg .f32 m, c; "
        f"max{f}.f32 m, $1, 0f38D1B717; mov.f32 c, 0f45280000; {div} $0, m, c; }}",
        # UE4M3 byte of group_amax / (6 * pts)
        sf="{ .reg .f32 s, q, z; .reg .b16 h; "
        f"mul{f}.f32 s, $2, 0f40C00000; {div} q, $1, s; mov.f32 z, 0f00000000; "
        "cvt.rn.satfinite.e4m3x2.f32 h, z, q; and.b16 h, h, 255; cvt.u32.u16 $0, h; }",
        # inv = sf > 0 ? 1 / (sf * pts) : 0, sf decoded through FP16 as cuda_fp8.h does
        inv="{ .reg .b16 e, h; .reg .b32 hh; .reg .f32 d, s; .reg .pred p; "
        "cvt.u16.u32 e, $1; cvt.rn.f16x2.e4m3x2 hh, e; cvt.u16.u32 h, hh; cvt.f32.f16 d, h; "
        f"mov.f32 $0, 0f00000000; setp.leu{f}.f32 p, d, 0f00000000; "
        f"@!p mul{f}.f32 s, $2, d; @!p {rcp} $0, s; }}",
        # Eight E2M1 codes of v * inv, low nibble = even column
        encode="{ .reg .b32 r, m; .reg .f32 a, b; .reg .pred p; .reg .b8 c0, c1, c2, c3; "
        + encode
        + "mov.b32 r, {c0, c1, c2, c3}; "
        "and.b32 m, r, 0x77777777; add.u32 m, m, 0x77777777; and.b32 m, m, 0x88888888; "
        "or.b32 m, m, 0x77777777; and.b32 $0, r, m; }",
    )


@cache
def _kernel_type(groups_per_lane, top_k, fast_math, warps_per_token):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64, Uint8, Uint32
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    ptx = _ptx(fast_math)

    def asm(result_type, text, constraints, *values, loc=None, ip=None):
        return llvm.inline_asm(
            result_type,
            [v.ir_value(loc=loc, ip=ip) for v in values],
            text,
            constraints,
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @dsl_user_op
    def group_amax(w, *, loc=None, ip=None):
        return Float32(asm(T.f32(), ptx["amax"], "=f" + ",r" * 8, *w, loc=loc, ip=ip))

    @dsl_user_op
    def fmax(a, b, *, loc=None, ip=None):
        op = "max.ftz.f32" if fast_math else "max.f32"
        return Float32(
            asm(T.f32(), op + " $0, $1, $2;", "=f,f,f", a, b, loc=loc, ip=ip)
        )

    @dsl_user_op
    def token_scale(row_amax, *, loc=None, ip=None):
        return Float32(asm(T.f32(), ptx["pts"], "=f,f", row_amax, loc=loc, ip=ip))

    @dsl_user_op
    def group_sf(amax, pts, *, loc=None, ip=None):
        return Uint32(asm(T.i32(), ptx["sf"], "=r,f,f", amax, pts, loc=loc, ip=ip))

    @dsl_user_op
    def group_inv(sf, pts, *, loc=None, ip=None):
        return Float32(asm(T.f32(), ptx["inv"], "=f,r,f", sf, pts, loc=loc, ip=ip))

    @dsl_user_op
    def encode8(w, inv, *, loc=None, ip=None):
        return Uint32(
            asm(T.i32(), ptx["encode"], "=r,r,r,r,r,f", *w, inv, loc=loc, ip=ip)
        )

    W = warps_per_token
    G = groups_per_lane // W  # groups per lane
    K = top_k
    warps = max(_WARPS, W)
    threads = 32 * warps
    tokens_per_block = warps // W
    pad_per_block = threads * _PAD_UNROLL

    class MegaMoENVFP4PreDispatch:
        @cute.jit
        def __call__(
            self,
            x: cute.Tensor,
            ids: cute.Tensor,
            scores: cute.Tensor,
            buf_x: cute.Tensor,
            buf_sf: cute.Tensor,
            buf_scales: cute.Tensor,
            buf_ids: cute.Tensor,
            buf_scores: cute.Tensor,
            stream,
        ):
            num_tokens = ids.shape[0] // K
            token_blocks = cute.ceil_div(num_tokens, tokens_per_block)
            pad_blocks = cute.ceil_div(buf_ids.shape[0] - ids.shape[0], pad_per_block)
            self.kernel(
                x,
                ids,
                scores,
                buf_x,
                buf_sf,
                buf_scales,
                buf_ids,
                buf_scores,
                num_tokens,
                token_blocks,
            ).launch(
                grid=[token_blocks + pad_blocks, 1, 1],
                block=[threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            x: cute.Tensor,
            ids: cute.Tensor,
            scores: cute.Tensor,
            buf_x: cute.Tensor,
            buf_sf: cute.Tensor,
            buf_scales: cute.Tensor,
            buf_ids: cute.Tensor,
            buf_scores: cute.Tensor,
            num_tokens: Int32,
            token_blocks: Int32,
        ):
            tid, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            if block < token_blocks:
                lane = tid % 32
                warp = tid // 32
                token = block * tokens_per_block + warp // W
                # Lane-major groups: this lane owns groups lane + 32 * (warp % W + W * k).
                first = token * (groups_per_lane * 32) + lane + (warp % W) * 32
                valid = token < num_tokens
                load = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), Uint32, num_bits_per_copy=128
                )
                store = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), Uint32, num_bits_per_copy=64
                )
                words = cute.make_rmem_tensor(cute.make_layout((4, 2, G)), Uint32)
                amax = cute.make_rmem_tensor(cute.make_layout((G,)), Float32)
                row = Float32(0.0)
                if valid:
                    # Issue every load before any arithmetic for memory-level parallelism.
                    for k in cutlass.range_constexpr(G):
                        vector = (first + k * 32 * W) * 2
                        for half in cutlass.range_constexpr(2):
                            cute.copy(
                                load, x[vector + half, None], words[None, half, k]
                            )
                    for k in cutlass.range_constexpr(G):
                        amax[k] = group_amax(
                            [words[i % 4, i // 4, k] for i in range(8)]
                        )
                        row = fmax(row, amax[k])
                for i in cutlass.range_constexpr(5):
                    row = fmax(row, cute.arch.shuffle_sync_bfly(row, offset=16 >> i))
                if cutlass.const_expr(W > 1):
                    smem = cutlass.memory.SmemAllocator()
                    partial = smem.allocate_tensor(
                        Float32, cute.make_layout((warps,)), byte_alignment=16
                    )
                    if lane == 0:
                        partial[warp] = row
                    cute.arch.barrier()
                    base = warp - warp % W
                    for i in cutlass.range_constexpr(W):
                        row = fmax(row, partial[base + i])
                if valid:
                    pts = token_scale(row)
                    codes = cute.make_rmem_tensor(cute.make_layout((2,)), Uint32)
                    for k in cutlass.range_constexpr(G):
                        group = first + k * 32 * W
                        sf = group_sf(amax[k], pts)
                        inv = group_inv(sf, pts)
                        for half in cutlass.range_constexpr(2):
                            codes[half] = encode8(
                                [words[i, half, k] for i in range(4)], inv
                            )
                        cute.copy(store, codes, buf_x[group, None])
                        buf_sf[group] = Uint8(sf)
                    if warp % W == 0:
                        if lane == 0:
                            buf_scales[token] = pts
                        if lane < K:
                            slot = token * K + lane
                            buf_ids[slot] = Int64(ids[slot])
                            buf_scores[slot] = scores[slot]
            else:
                # Invalidate routing rows past the shard, as DeepGEMM's pad CTAs do.
                base = ids.shape[0] + (block - token_blocks) * pad_per_block + tid
                total = buf_ids.shape[0]
                for u in cutlass.range_constexpr(_PAD_UNROLL):
                    slot = base + u * threads
                    if slot < total:
                        buf_ids[slot] = Int64(-1)
                        buf_scores[slot] = Float32(0.0)

    return MegaMoENVFP4PreDispatch
