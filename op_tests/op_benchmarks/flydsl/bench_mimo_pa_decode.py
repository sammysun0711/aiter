#!/usr/bin/env python3
"""MiMo Flash DFlash8 paged-attention microbenchmark for MI300X/gfx942.

Run with the same AITER/FlyDSL/Torch environment as the server. Example:
  python3 op_tests/op_benchmarks/flydsl/bench_mimo_pa_decode.py \
      --families full swa draft --output /tmp/pa-run

Measures one layer on one GPU, with local attention BS (not DP-gathered MoE M).
Defaults reproduce the ten ISL/BS pairs, BF16 packed page64 caches, physical
pool extents and table bounds used by the PR36 TP8/DP2 decode campaign.
Synthetic inputs exclude QKV projection/cache writes, table construction,
communication and scheduling. Two timings separate PA+reduction from
plan-refresh+PA+reduction. Neither is whole-model decode latency.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time


# ISL, local BS, full/draft pool token capacity, target SWA/full ratio.
CASES = [
    (4096, 256, 10513664, 0.03),
    (8192, 256, 10513664, 0.03),
    (16384, 240, 10513664, 0.03),
    (32768, 208, 10513664, 0.03),
    (65536, 152, 10513664, 0.03),
    (131072, 96, 12863872, 0.012),
    (262144, 40, 12863872, 0.012),
    (524288, 24, 13211264, 0.003),
    (786432, 12, 10513664, 0.03),
    (1040000, 12, 13211264, 0.003),
]
PAGE, QLEN, HQ, DV, CONTEXT_BOUND = 64, 8, 16, 128, 1048576


def geometry(case, family, args):
    isl, bs, full_pool, ratio = case
    if family == "full":
        hkv, dk, window, length = 1, 192, 0, isl + args.decode_offset + QLEN
        pool, width = full_pool, CONTEXT_BOUND // PAGE
    elif family == "swa":
        # The grouped adapter retains the final four pages. At aligned ISL,
        # three complete pages plus eight new tokens give compact length200.
        hkv, dk, window = 2, 192, 128
        length = 3 * PAGE + (isl + args.decode_offset + QLEN - 1) % PAGE + 1
        pool, width = int(full_pool * ratio) // PAGE * PAGE, 4
    else:
        # Aligned draft entry: retained prefix1024 followed by eight queries.
        hkv, dk, window = 2, 128, 1023
        length = 1024 + args.decode_offset % PAGE + QLEN
        pool, width = full_pool, CONTEXT_BOUND // PAGE
    pages_per_seq = math.ceil(length / PAGE)
    if args.compact_pool:
        pool = (bs * pages_per_seq + 2) * PAGE
    if args.compact_table:
        width = pages_per_seq
    if pages_per_seq > width or bs * pages_per_seq + 2 > pool // PAGE:
        raise ValueError("Case exceeds the selected table/pool; reduce decode-offset")
    override = args.case_overrides.get(family, {}).get(str(isl), {})
    return dict(family=family, isl=isl, bs_per_dp=bs, global_c=2 * bs,
                local_query_tokens=bs * QLEN, qlen=QLEN, hq=HQ, hkv=hkv,
                dk=dk, dv=DV, page=PAGE, length=length, window=window,
                pool_tokens=pool, table_width=width,
                partitions=override.get("partitions", getattr(args, family + "_partitions")),
                query_splits=override.get("query_splits", args.query_splits))


def make_inputs(torch, spec, args):
    b, h, d, length = (spec[k] for k in ("bs_per_dp", "hkv", "dk", "length"))
    pages = spec["pool_tokens"] // PAGE
    used = math.ceil(length / PAGE)
    q = torch.randn(b * QLEN, HQ, d, device="cuda", dtype=torch.bfloat16)
    unused = float("nan") if args.unused_kv == "nan" else 0.0
    k = torch.full((pages, h, d // 8, PAGE, 8), unused, device="cuda", dtype=torch.bfloat16)
    v = torch.full((pages, h, PAGE // 8, DV, 8), unused, device="cuda", dtype=torch.bfloat16)
    # Spread live pages across the real allocation, exercising high addresses.
    ids = torch.arange(b * used, device="cuda")
    ids = 2 + (ids * (pages - 2) // (b * used) if args.page_layout == "spread" else ids)
    for begin in range(0, ids.numel(), 4096):
        selected = ids[begin:begin + 4096]
        if args.kv_pattern == "zero":
            k[selected] = 0
            v[selected] = 0
        else:
            k[selected] = torch.randn(selected.numel(), h, d // 8, PAGE, 8, device="cuda", dtype=k.dtype)
            v[selected] = torch.randn(selected.numel(), h, PAGE // 8, DV, 8, device="cuda", dtype=v.dtype)
    k[1].zero_()
    v[1].fill_(0.75)  # Reserved finite page for the page-map replay check.
    tables = torch.zeros((b, spec["table_width"]), device="cuda", dtype=torch.int32)
    tables[:, :used] = ids.reshape(b, used).int()
    if length % PAGE:
        last = ids.reshape(b, used)[:, -1]
        tail = torch.arange(length % PAGE, PAGE, device="cuda")
        k[last[:, None], :, :, tail[None, :], :] = unused
        v[last[:, None], :, tail[None, :] // 8, :, tail[None, :] % 8] = unused
    lengths = torch.full((b,), length, device="cuda", dtype=torch.int32)
    sinks = torch.randn(HQ, device="cuda", dtype=torch.float32) if spec["window"] else None
    out = torch.empty(b * QLEN, HQ, DV, device="cuda", dtype=torch.bfloat16)
    return dict(q=q, k=k, v=v, lengths=lengths, tables=tables, sinks=sinks, out=out)


def check_reference(torch, spec, data):
    """Streaming FP32 oracle for first/last query and head of first/last batch."""
    errors, refs = [], []
    hkv, dk = spec["hkv"], spec["dk"]
    selected = [(0, 0), (0, HQ - 1), (QLEN - 1, 0), (QLEN - 1, HQ - 1)]
    for seq in sorted({0, spec["bs_per_dp"] - 1}):
        length = int(data["lengths"][seq].item())
        qpos = torch.tensor([p for p, _ in selected], device="cuda")
        heads = torch.tensor([h for _, h in selected], device="cuda")
        kvheads = heads // (HQ // hkv)
        q = data["q"][seq * QLEN + qpos, heads].float()
        ends = length - QLEN + qpos
        starts = (ends - spec["window"] + 1).clamp_min(0) if spec["window"] else torch.zeros_like(ends)
        if data["sinks"] is None:
            maximum = torch.full((4,), -float("inf"), device="cuda")
            denominator = torch.zeros(4, device="cuda")
        else:
            maximum = data["sinks"][heads].clone()
            denominator = torch.ones(4, device="cuda")
        numerator = torch.zeros(4, DV, device="cuda")
        first_page = int(starts.min().item()) // PAGE
        for first in range(first_page, math.ceil(length / PAGE), 128):
            last = min(first + 128, math.ceil(length / PAGE))
            ids = data["tables"][seq, first:last].long()
            k = data["k"][ids].permute(0, 3, 1, 2, 4).reshape(-1, hkv, dk)
            v = data["v"][ids].permute(0, 2, 4, 1, 3).reshape(-1, hkv, DV)
            valid = min(k.shape[0], length - first * PAGE)
            positions = torch.arange(first * PAGE, first * PAGE + valid, device="cuda")
            keys = k[:valid, kvheads].permute(1, 0, 2).float()
            values = v[:valid, kvheads].permute(1, 0, 2).float()
            logits = torch.einsum("rd,rtd->rt", q, keys) / math.sqrt(dk)
            logits.masked_fill_((positions[None] > ends[:, None]) | (positions[None] < starts[:, None]), -float("inf"))
            new_max = torch.maximum(maximum, logits.amax(dim=1))
            old_scale = (maximum - new_max).exp()
            weights = (logits - new_max[:, None]).exp()
            numerator = numerator * old_scale[:, None] + torch.einsum("rt,rtd->rd", weights, values)
            denominator = denominator * old_scale + weights.sum(dim=1)
            maximum = new_max
        ref = numerator / denominator[:, None]
        actual = data["out"][seq * QLEN + qpos, heads].float()
        refs.append(ref.flatten())
        errors.append((actual - ref).flatten())
    assert bool(torch.isfinite(data["out"]).all()), "Non-finite PA output"
    error, ref = torch.cat(errors), torch.cat(refs)
    absolute = error.abs().max().item()
    relative = (error.norm() / ref.norm().clamp_min(1e-6)).item()
    assert absolute < 0.01 and relative < 0.02, (absolute, relative)
    return dict(sampled_rows=len(selected) * len(sorted({0, spec["bs_per_dp"] - 1})), max_abs=absolute, relative_l2=relative)


def measure(torch, pa, spec, args):
    data = make_inputs(torch, spec, args)
    digest = hashlib.sha256()
    sample_ids = data["tables"][:, [0, math.ceil(spec["length"] / PAGE) - 1]].long().flatten()
    for tensor in (data["q"], data["tables"], data["lengths"], data["k"][sample_ids], data["v"][sample_ids]):
        digest.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    input_signature = digest.hexdigest()
    compile_info = {}
    original_compile = pa.compile_pa_decode_tile

    def record_compile(**kwargs):
        compiled = original_compile(**kwargs)
        compile_info["arguments"] = kwargs
        module = sys.modules[original_compile.__module__]
        if hasattr(module, "PaDecodeSchedule"):
            fields = inspect.signature(module.PaDecodeSchedule.select).parameters
            schedule = module.PaDecodeSchedule.select(**{k: v for k, v in kwargs.items() if k in fields})
            traits = module.PaDecodeTraits.create(schedule)
            compile_info.update(
                resolved_query_splits=traits.query_splits,
                lds_bytes=traits.total_bytes,
                batch_first_plan_grid=traits.batch_first_plan_grid,
                query_first_plan_grid=getattr(traits, "query_first_plan_grid", False),
            )
        return compiled

    pa.compile_pa_decode_tile = record_compile
    planned_api = hasattr(pa, "pa_decode")
    if planned_api:
        plan = pa.plan_pa_decode(data["lengths"], spec["hkv"], max_partitions=spec["partitions"],
                                max_batch_size=spec["bs_per_dp"],
                                workgroup_budget=spec["bs_per_dp"] * spec["hkv"] * spec["partitions"],
                                sliding_window=spec["window"], query_length=QLEN)
        shape = (spec["hkv"], plan.capacity, QLEN * HQ // spec["hkv"])
    else:
        plan = None
        shape = (spec["bs_per_dp"], spec["hkv"], spec["partitions"], QLEN * HQ // spec["hkv"])
    pmax = torch.empty(shape, device="cuda", dtype=torch.float32)
    psum = torch.empty_like(pmax)
    pout = torch.empty(*shape, DV, device="cuda", dtype=torch.bfloat16)

    def decode():
        if not planned_api:
            pa.pa_decode_tile(data["out"], data["q"], data["k"], data["v"], data["tables"], data["lengths"],
                              None, None, softmax_scale=1 / math.sqrt(spec["dk"]), num_partitions=spec["partitions"],
                              pmax=pmax, psum=psum, pout=pout, sliding_window=spec["window"], sinks=data["sinks"],
                              query_splits=spec["query_splits"])
            return
        pa.pa_decode(data["out"], data["q"], data["k"], data["v"], data["lengths"], data["tables"],
                     1 / math.sqrt(spec["dk"]), QLEN, work_plan=plan,
                     sliding_window=spec["window"], sinks=data["sinks"],
                     max_context_length=spec["table_width"] * PAGE,
                     exp_sums=psum, max_logits=pmax, temporary_output=pout,
                     query_splits=spec["query_splits"], mask_unowned_kv=args.mask_unowned_kv)

    def refresh_decode():
        if planned_api:
            pa.plan_pa_decode(data["lengths"], spec["hkv"], plan=plan, sliding_window=spec["window"], query_length=QLEN)
        decode()

    for _ in range(10):
        refresh_decode()
    torch.cuda.synchronize()
    validation = {"eager": check_reference(torch, spec, data)}
    original = data["out"].clone()
    original_q = data["q"].clone()
    graphs = {}
    # Capture on a side stream after eager compilation/warmup.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name, fn in (("decode_reduce", decode), ("plan_decode_reduce", refresh_decode)):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graph.replay()
            stream.synchronize()
            torch.testing.assert_close(data["out"], original, atol=0, rtol=0)
            graphs[name] = graph
    torch.cuda.current_stream().wait_stream(stream)
    data["q"].mul_(-0.75)
    data["lengths"][::2].sub_(3)
    graphs["plan_decode_reduce"].replay()
    torch.cuda.synchronize()
    validation["changed_query_lengths"] = check_reference(torch, spec, data)
    saved_table = data["tables"][0].clone()
    data["tables"][0].fill_(1)
    graphs["plan_decode_reduce"].replay()
    torch.cuda.synchronize()
    validation["changed_page_map"] = check_reference(torch, spec, data)
    data["tables"][0].copy_(saved_table)
    data["lengths"].fill_(spec["length"])
    data["q"].copy_(original_q)
    refresh_decode()
    validation["restored"] = check_reference(torch, spec, data)
    warmup = {}
    for name, graph in graphs.items():
        start, count = time.perf_counter(), 0
        while count < 200 or time.perf_counter() - start < args.warmup_seconds:
            for _ in range(50):
                graph.replay()
            torch.cuda.synchronize()
            count += 50
        warmup[name] = dict(replays=count, seconds=time.perf_counter() - start)
    rounds = []
    for i in range(args.rounds):
        order = list(graphs) if i % 2 == 0 else list(reversed(graphs))
        row = {}
        for name in order:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(args.repeat):
                graphs[name].replay()
            end.record()
            end.synchronize()
            row[name] = 1000 * begin.elapsed_time(end) / args.repeat
        rounds.append(row)
    values = {name: [r[name] for r in rounds] for name in graphs}
    pa.compile_pa_decode_tile = original_compile
    return dict(**spec, validation=validation, graph_eager_bitwise=True,
                warmup=warmup, rounds_us=rounds, median_us={n: statistics.median(v) for n, v in values.items()},
                cv_percent={n: 100 * statistics.pstdev(v) / statistics.mean(v) for n, v in values.items()},
                k_extent_bytes=data["k"].numel() * 2, v_extent_bytes=data["v"].numel() * 2,
                plan_capacity=plan.capacity if planned_api else None, requested_query_splits=spec["query_splits"],
                api="planned" if planned_api else "legacy", page_layout=args.page_layout, kv_pattern=args.kv_pattern,
                unused_kv=args.unused_kv,
                input_signature=input_signature, compile_info=compile_info)


def write_report(report, directory):
    (directory / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ["family", "isl", "bs_per_dp", "global_c", "local_query_tokens", "length", "pool_tokens", "table_width", "partitions", "decode_reduce_us", "plan_decode_reduce_us"]
    with (directory / "results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in report["results"]:
            writer.writerow({**{k: r[k] for k in fields[:-2]}, **{k + "_us": v for k, v in r["median_us"].items()}})
    lines = ["# MiMo Flash DFlash8 PA decode microbenchmark", "", f"Status: **{report['status']}**.", "",
             "One synthetic layer on one MI300X. BF16 Q/K/V, page64, Q8. Full Hq16/Hkv1/QK192/V128; grouped SWA Hq16/Hkv2/QK192/V128/window128; draft Hq16/Hkv2/D128/window1023. Physical pool extents and table bounds are in JSON/CSV. No projection, cache writes, table construction, MoE, communication or scheduling is timed.", "",
             "Each mode receives at least200 graph replays and the configured warmup duration. Table values are medians of alternating-mode timed intervals, including reduction. Plan-refresh timing includes its GPU kernel only. Sampled FP32 reference, poisoned unused page/tails, exact eager/replay parity, changed query/lengths and changed page-map replay checks must pass before timing.", "",
             "| Family | ISL | BS/DP | BS×8 | PA + reduce µs | Plan + PA + reduce µs |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in report["results"]:
        lines.append(f"| {r['family']} | {r['isl']:,} | {r['bs_per_dp']} | {r['local_query_tokens']:,} | {r['median_us']['decode_reduce']:.2f} | {r['median_us']['plan_decode_reduce']:.2f} |")
    lines += ["", "SWA and draft use bounded retained context, so their nominal ISL rows mainly differ in BS and allocation geometry. Equal-shape cases are still measured separately. Warmup/timing counts, every interval, error metrics, runtime and source hashes are in [results.json](results.json). Compact-pool/table overrides are distinct experiments and must not be presented as the serving recipe."]
    (directory / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", nargs="+", choices=("full", "swa", "draft"), default=["full"])
    parser.add_argument("--isls", nargs="+", type=int, choices=[c[0] for c in CASES])
    parser.add_argument("--batch-sizes", nargs="+", type=int, help="Override local BS for a fixed-ISL scaling sweep")
    parser.add_argument("--case-config", type=Path, help="JSON family/ISL mapping to partitions and query_splits")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pa-source", type=Path, help="Isolated aiter/ops/flydsl source snapshot; other AITER modules retain installed versions")
    parser.add_argument("--page-layout", choices=("spread", "contiguous"), default="spread")
    parser.add_argument("--kv-pattern", choices=("random", "zero"), default="random")
    parser.add_argument("--unused-kv", choices=("nan", "zero"), default="nan", help="NaN validates tail masking; zero permits comparisons to legacy finite-tail APIs")
    parser.add_argument("--mask-unowned-kv", action=argparse.BooleanOptionalAction, default=True,
                        help="Disable only for finite-tail diagnostic measurements")
    parser.add_argument("--full-partitions", type=int, default=16)
    parser.add_argument("--swa-partitions", type=int, default=1)
    parser.add_argument("--draft-partitions", type=int, default=8)
    parser.add_argument("--query-splits", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--decode-offset", type=int, default=0)
    parser.add_argument("--compact-pool", action="store_true")
    parser.add_argument("--compact-table", action="store_true")
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--list", action="store_true", help="Print geometry without importing Torch or using GPUs")
    parser.add_argument("--output", type=Path, help="New result directory; existing evidence is not overwritten")
    args = parser.parse_args()
    args.case_overrides = json.loads(args.case_config.read_text()) if args.case_config else {}
    for family, cases in args.case_overrides.items():
        if family not in ("full", "swa", "draft"):
            parser.error(f"Unknown case-config family: {family}")
        for isl, config in cases.items():
            if int(isl) not in {c[0] for c in CASES} or set(config) - {"partitions", "query_splits"}:
                parser.error(f"Invalid case-config entry: {family}/{isl}")
            if "partitions" in config and (type(config["partitions"]) is not int or config["partitions"] < 1):
                parser.error("case-config partitions must be positive integers")
            if "query_splits" in config:
                value = config["query_splits"]
                if value is not None and (type(value) is not int or value not in (1, 2, 4, 8)):
                    parser.error("case-config query_splits must be1,2,4,8 or null")
    if min(args.full_partitions, args.swa_partitions, args.draft_partitions, args.rounds, args.repeat) < 1 or args.warmup_seconds < 0 or args.decode_offset < 0:
        parser.error("Partitions/rounds/repeat must be positive; warmup and decode-offset nonnegative")
    if args.batch_sizes and min(args.batch_sizes) < 1:
        parser.error("Batch sizes must be positive")
    if not args.mask_unowned_kv and args.unused_kv != "zero":
        parser.error("--no-mask-unowned-kv requires --unused-kv zero")
    cases = [c for c in CASES if args.isls is None or c[0] in args.isls]
    if args.batch_sizes:
        cases = [(c[0], bs, c[2], c[3]) for c in cases for bs in args.batch_sizes]
    specs = [geometry(c, f, args) for c in cases for f in args.families]
    if args.list:
        print(json.dumps(specs, indent=2))
        return
    if args.output is None or args.output.exists():
        parser.error("--output must be a new directory")
    import torch
    import flydsl
    if args.pa_source:
        import aiter.ops
        if not (args.pa_source / "pa_decode.py").is_file():
            parser.error("--pa-source must be an aiter/ops/flydsl source directory")
        # AITER's top-level exports may preload this package. No PA has run yet;
        # discard that entire subtree before loading the isolated snapshot so
        # its absolute and relative imports both resolve to the same version.
        for name in list(sys.modules):
            if name == "aiter.ops.flydsl" or name.startswith("aiter.ops.flydsl."):
                del sys.modules[name]
        if hasattr(aiter.ops, "flydsl"):
            delattr(aiter.ops, "flydsl")
        aiter.ops.__path__.insert(0, str(args.pa_source.resolve().parent))
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    properties = torch.cuda.get_device_properties(args.gpu)
    if properties.gcnArchName.split(":")[0] != "gfx942":
        raise RuntimeError("This benchmark requires MI300X/gfx942")
    files = [Path(__file__).resolve(), Path(pa.__file__),
             *sorted((Path(pa.__file__).parent / "kernels").glob("pa_decode*.py")),
             *sorted((Path(pa.__file__).parent / "kernels/pa_decode").glob("*.py"))]
    if args.case_config:
        files.append(args.case_config.resolve())
    args.output.mkdir(parents=True)
    # ROCm SMI can report the host PID while /proc exposes the container PID.
    # Record the process cohort from inside the sole GPU process after an idle
    # startup, so the external monitor does not mistake this job for a rival.
    probe = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    smi = json.loads(subprocess.check_output(["rocm-smi", "--showpids", "--json"], text=True))
    (args.output / "gpu_identity.json").write_text(json.dumps(dict(container_pid=__import__("os").getpid(), rocm=smi), indent=2) + "\n")
    report = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
                  torch=torch.__version__, flydsl=getattr(flydsl, "__version__", "unknown"), gpu=properties.name,
                  aiter_module=pa.__file__, torch_module=torch.__file__, arch=properties.gcnArchName,
                  libraries=sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                                    if any(n in line for n in ("libamdhip64", "libhsa-runtime64", "libroctracer64"))}),
                  config={**vars(args), "output": str(args.output), "pa_source": str(args.pa_source) if args.pa_source else None,
                          "case_config": str(args.case_config) if args.case_config else None},
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, results=[])
    write_report(report, args.output)
    try:
        for spec in specs:
            print(f"START {spec['family']} ISL={spec['isl']} BS={spec['bs_per_dp']}", flush=True)
            result = measure(torch, pa, spec, args)
            report["results"].append(result)
            write_report(report, args.output)
            print(json.dumps({k: result[k] for k in ("family", "isl", "median_us", "cv_percent")}), flush=True)
            torch.cuda.empty_cache()
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_report(report, args.output)


if __name__ == "__main__":
    main()
