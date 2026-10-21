import itertools
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.moe.muse_spark_megamoe_nvfp4_pre_dispatch import (
    can_pre_dispatch_nvfp4,
    pre_dispatch_nvfp4,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_GUARD = 128
_SLOTS = ("x", "x_sf", "x_scales", "topk_idx", "topk_weights")

try:
    import deep_gemm

    _HAS_REFERENCE = hasattr(deep_gemm, "mega_moe_pre_dispatch")
except ImportError:
    _HAS_REFERENCE = False


def _guarded(shape, dtype, fill):
    """A contiguous view with guard bytes on both sides."""
    numel = 1
    for s in shape:
        numel *= s
    size = torch.empty((), dtype=dtype).element_size()
    raw = torch.full(
        (2 * _GUARD + numel * size,), 0xA5, device="cuda", dtype=torch.uint8
    )
    view = raw[_GUARD:-_GUARD].view(dtype).view(shape)
    view.view(torch.uint8).fill_(fill)
    return raw, view


def _buffer(capacity, hidden, topk, fill=0x33):
    shapes = (
        ((capacity, hidden // 2), torch.uint8),
        ((capacity, hidden // 64), torch.int32),
        ((capacity,), torch.float32),
        ((capacity, topk), torch.int64),
        ((capacity, topk), torch.float32),
    )
    raws, views = zip(*(_guarded(s, d, fill) for s, d in shapes))
    # DeepGEMM buffers carry their MMA type; the adapter may branch on it.
    return list(raws), SimpleNamespace(
        mma_type="nvfp4xnvfp4", **dict(zip(_SLOTS, views))
    )


def _views(buf):
    return [getattr(buf, k) for k in _SLOTS]


def _inputs(n, hidden, topk, dist, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n, hidden, device="cuda", generator=generator)
    if dist == "groupscale":
        scale = torch.rand(n, hidden // 16, 1, device="cuda", generator=generator)
        x = (x.view(n, -1, 16) * torch.exp2(scale * 50 - 40)).view(n, hidden)
        x[:, :16] = 0
    elif dist == "huge":
        x = x * 2.0**100
    elif dist == "special" and n:
        bits = x.bfloat16().view(torch.int16)
        bits[:, ::7] = 0x0001  # BF16 subnormal
        bits[:, 1::11] = -0x8000  # -0
        bits[0, 2::13] = 0x7F80  # +inf
        bits[0, 3::17] = 0x7FC0  # NaN
        if n > 1:
            bits[1] = 0x7FC0
        x = bits.view(torch.bfloat16)
    x = x.bfloat16()
    ids = torch.randint(
        -1, 256, (n, topk), device="cuda", dtype=torch.int32, generator=generator
    )
    scores = torch.randn(n, topk, device="cuda", generator=generator)
    return x, ids, scores


def _reference(x, ids, scores, buf):
    deep_gemm.mega_moe_pre_dispatch(
        x,
        ids,
        scores,
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
        x.shape[0],
        group_size=16,
        mma_type="nvfp4xnvfp4",
        buf_x_scales=buf.x_scales,
    )


def _assert_same(test, raws, expected):
    for slot, got, ref in zip(_SLOTS, raws, expected):
        test.assertTrue(torch.equal(got, ref), f"{slot} differs from DeepGEMM")


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 10
    and _HAS_REFERENCE,
    "requires SM100 and DeepGEMM's pre-dispatch",
)
class TestMuseSparkMegaMoENVFP4PreDispatch(CustomTestCase):
    def test_matches_deep_gemm_bytes(self):
        cases = itertools.product(
            (1, 7, 31, 256, 1312, 2048),
            ((4096, 8, 9600), (512, 1, 256), (2048, 4, 2048)),
            ("normal", "groupscale", "huge", "special"),
        )
        for seed, (n, (hidden, topk, capacity), dist) in enumerate(cases):
            if n > capacity:
                continue
            with self.subTest(n=n, hidden=hidden, topk=topk, dist=dist):
                x, ids, scores = _inputs(n, hidden, topk, dist, seed)
                fill = (0x33, 0xC5)[seed % 2]
                ref_raws, ref = _buffer(capacity, hidden, topk, fill)
                raws, buf = _buffer(capacity, hidden, topk, fill)
                self.assertTrue(can_pre_dispatch_nvfp4(x, ids, scores, *_views(buf)))
                saved = [t.clone() for t in (x, ids, scores)]
                _reference(x, ids, scores, ref)
                pre_dispatch_nvfp4(x, ids, scores, *_views(buf))
                torch.cuda.synchronize()
                _assert_same(self, raws, ref_raws)
                for before, t in zip(saved, (x, ids, scores)):
                    # Bytes, not values: NaN inputs never compare equal.
                    self.assertTrue(
                        torch.equal(before.view(torch.uint8), t.view(torch.uint8))
                    )

    def test_single_row_with_row_gap(self):
        # PyTorch calls a one-row view contiguous whatever its row stride.
        for hidden, gap in ((4096, 17), (512, 1)):
            with self.subTest(hidden=hidden, gap=gap):
                base, ids, scores = _inputs(1, hidden + gap, 8, "normal", 11)
                x = base[:, :hidden]
                self.assertTrue(x.is_contiguous() and x.stride(0) == hidden + gap)
                ref_raws, ref = _buffer(64, hidden, 8)
                raws, buf = _buffer(64, hidden, 8)
                self.assertTrue(can_pre_dispatch_nvfp4(x, ids, scores, *_views(buf)))
                _reference(x, ids, scores, ref)
                pre_dispatch_nvfp4(x, ids, scores, *_views(buf))
                torch.cuda.synchronize()
                _assert_same(self, raws, ref_raws)

    def test_rejects_uncovered_layouts(self):
        x, ids, scores = _inputs(8, 512, 8, "normal", 0)
        _, buf = _buffer(64, 512, 8)
        views = _views(buf)
        rejected = (
            (x[:, :256], ids, scores),
            (x.t().contiguous().t(), ids, scores),
            (x, ids.long(), scores),
            (x, ids, scores.bfloat16()),
            (x[:0], ids[:0], scores[:0]),
            (torch.cat((x.view(-1)[:1], x.view(-1)))[1:].view(8, 512), ids, scores),
        )
        for inputs in rejected:
            self.assertFalse(can_pre_dispatch_nvfp4(*inputs, *views))
        _, small = _buffer(4, 512, 8)
        self.assertFalse(can_pre_dispatch_nvfp4(x, ids, scores, *_views(small)))
        x520, _, _ = _inputs(8, 520, 8, "normal", 0)
        _, buf520 = _buffer(64, 520, 8)
        self.assertFalse(can_pre_dispatch_nvfp4(x520, ids, scores, *_views(buf520)))

    def test_rejects_unvalidated_widths(self):
        # Four-warp grouping must cover every group. In particular widths
        # 1536/2560/3072/3584/7168 cannot use the frozen implementation.
        for hidden in (0, 1024, 1536, 2560, 3072, 3584, 7168, 8192):
            with self.subTest(hidden=hidden):
                x, ids, scores = _inputs(2, hidden, 8, "normal", 12)
                _, buf = _buffer(4, hidden, 8)
                self.assertFalse(can_pre_dispatch_nvfp4(x, ids, scores, *_views(buf)))

    def test_shrinking_batches_graph_and_side_stream(self):
        hidden, topk, capacity = 4096, 8, 2048
        x, ids, scores = _inputs(2048, hidden, topk, "normal", 7)
        raws, buf = _buffer(capacity, hidden, topk)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs = {}
        for n in (2048, 31, 1):
            with torch.cuda.stream(stream):
                pre_dispatch_nvfp4(x[:n], ids[:n], scores[:n], *_views(buf))
            stream.synchronize()
            graphs[n] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[n], stream=stream):
                pre_dispatch_nvfp4(x[:n], ids[:n], scores[:n], *_views(buf))
        for rep in range(24):
            n = (2048, 31, 1)[rep % 3]
            # Same starting bytes on both sides, as after a larger previous batch.
            ref_raws, ref = _buffer(capacity, hidden, topk)
            for got, want in zip(ref_raws, raws):
                got.copy_(want)
            _reference(x[:n], ids[:n], scores[:n], ref)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                if rep % 2:
                    graphs[n].replay()
                else:
                    pre_dispatch_nvfp4(x[:n], ids[:n], scores[:n], *_views(buf))
            stream.synchronize()
            _assert_same(self, raws, ref_raws)

    def test_forward_megamoe_dispatch(self):
        from sglang.srt.layers.muse_spark import muse_spark_megamoe as adapter

        hidden, topk, capacity, n = 4096, 8, 256, 31
        consumer = mock.Mock()
        layer = SimpleNamespace(
            _megamoe_weights_built=True,
            _megamoe_mma_type="nvfp4xnvfp4",
            _megamoe_norm_cache=(None, None, 1e-8, False),
            num_experts=256,
            expert_hidden_size=hidden,
            intermediate_size=4096,
            top_k=topk,
            megamoe_l1_weights=None,
            megamoe_l2_weights=None,
            megamoe_l1_alphas=None,
            megamoe_l2_alphas=None,
            megamoe_l2_act_scales=None,
        )
        for contiguous, ids_dtype in ((True, torch.int32), (False, torch.int64)):
            with self.subTest(contiguous=contiguous, ids=ids_dtype):
                raws, buf = _buffer(capacity, hidden, topk)
                ref_raws, ref = _buffer(capacity, hidden, topk)
                x, ids, scores = _inputs(n, hidden, topk, "normal", 3)
                _reference(x, ids, scores, ref)
                if not contiguous:
                    x = x.t().contiguous().t()
                with (
                    mock.patch.object(
                        deep_gemm, "nvfp4_mega_moe", consumer, create=True
                    ),
                    mock.patch.object(adapter, "_get_ep_group", lambda: None),
                    mock.patch.object(adapter, "_get_symm_buffer", lambda *a: buf),
                    mock.patch.object(adapter, "_token_capacity", lambda: capacity),
                    mock.patch(
                        "sglang.kernels.ops.moe.muse_spark_megamoe_nvfp4_pre_dispatch.pre_dispatch_nvfp4",
                        wraps=pre_dispatch_nvfp4,
                    ) as fused,
                ):
                    y = adapter.forward_megamoe(
                        layer, x, scores, ids.to(ids_dtype), None
                    )
                torch.cuda.synchronize()
                self.assertEqual(fused.call_count, 1)
                self.assertEqual(tuple(y.shape), (n, hidden))
                self.assertIs(consumer.call_args.args[3], buf)
                _assert_same(self, raws, ref_raws)


if __name__ == "__main__":
    unittest.main()
