# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sustained MI300X graph replay with matched AMD SMI telemetry."""

import argparse
import json
import math
import threading
from pathlib import Path
from statistics import median

import amdsmi
import torch


def identify_gpu(props):
    for handle in amdsmi.amdsmi_get_processor_handles():
        bdf = amdsmi.amdsmi_get_gpu_device_bdf(handle)
        domain, bus, devfn = bdf.split(":")
        device, function = devfn.split(".")
        if (int(domain, 16), int(bus, 16), int(device, 16), int(function, 16)) == (
            props.pci_domain_id,
            props.pci_bus_id,
            props.pci_device_id,
            0,
        ):
            return handle, bdf
    raise RuntimeError("No AMD SMI handle for the HIP device")


def sample(handle):
    data = amdsmi.amdsmi_get_gpu_metrics_info(handle)
    fields = (
        "firmware_timestamp",
        "accumulation_counter",
        "energy_accumulator",
        "ppt_residency_acc",
        "socket_thm_residency_acc",
        "vr_thm_residency_acc",
        "hbm_thm_residency_acc",
        "prochot_residency_acc",
        "current_socket_power",
        "current_gfxclks",
        "temperature_hotspot",
        "average_gfx_activity",
    )
    return {field: data[field] for field in fields}


def capture(call, stream, count):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(count):
            call()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def duration(graph, replays):
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / 1000


def sustained(graph, repetitions, handle, resolution, count, shape):
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    # HIP may block the submitting thread when a graph contains many short
    # dispatches. Sample concurrently from the first enqueue through completion.
    start_sampling = threading.Event()
    stop_sampling = threading.Event()
    samples, failures = [], []

    def poll():
        start_sampling.wait()
        try:
            while not stop_sampling.is_set():
                snap = sample(handle)
                if not stop_sampling.is_set():
                    samples.append(snap)
                stop_sampling.wait(0.1)
        # Relay sampler errors to the thread running the benchmark.
        except Exception as exc:  # noqa: BLE001
            failures.append(exc)

    sampler = threading.Thread(target=poll)
    sampler.start()
    try:
        begin.record()
        start_sampling.set()
        for _ in range(repetitions):
            graph.replay()
        end.record()
        end.synchronize()
    finally:
        stop_sampling.set()
        start_sampling.set()
        sampler.join()
    if failures:
        raise failures[0]
    seconds = begin.elapsed_time(end) / 1000
    if len(samples) < 2:
        raise RuntimeError("Too few in-dispatch telemetry samples")
    lo, hi = samples[0], samples[-1]
    dt = (hi["firmware_timestamp"] - lo["firmware_timestamp"]) * 1e-9
    ticks = (hi["accumulation_counter"] - lo["accumulation_counter"]) % 2**32
    if dt < 0.8 * seconds or ticks == 0:
        raise RuntimeError(
            f"Insufficient telemetry coverage: {dt:.3f} s of {seconds:.3f} s, {ticks} ticks"
        )
    watts = (
        (hi["energy_accumulator"] - lo["energy_accumulator"]) * resolution * 1e-6 / dt
    )
    us = seconds * 1e6 / (count * repetitions)
    result = {
        "event_s": seconds,
        "samples": len(samples),
        "coverage_s": dt,
        "us": us,
        "mean_energy_w": watts,
        "median_socket_w": median(x["current_socket_power"] for x in samples),
        "max_socket_w": max(x["current_socket_power"] for x in samples),
        "median_gfx_mhz": median(
            sum(x["current_gfxclks"]) / len(x["current_gfxclks"]) for x in samples
        ),
        "median_hotspot_c": median(x["temperature_hotspot"] for x in samples),
    }
    result.update(
        tflops=2 * math.prod(shape) / us / 1e6, joules_per_gemm=watts * us / 1e6
    )
    for key in ("ppt", "socket_thm", "hbm_thm", "vr_thm", "prochot"):
        field = key + "_residency_acc"
        pct = 100 * ((hi[field] - lo[field]) % 2**32) / ticks
        if pct > 100.01:
            raise RuntimeError(f"Invalid firmware residency delta for {key}: {pct}")
        result[key + "_pct"] = pct
    return {"result": result, "samples": samples}


def run(args):
    from bench_fp8_scaling_comparison import find_row, methods
    from common import Inputs, error_metric

    torch.cuda.set_device(0)
    d = Inputs(tuple(args.shape))
    plan = find_row(args.config_dir / "replay-plans.csv", *args.shape)
    limits = {int(v) for key, v in plan.items() if key not in ("M", "N", "K") and v}
    ri, ci, _, ref = d.reference(sorted(step for step in limits if step < d.m))
    stream = torch.cuda.Stream()
    graphs, errors, keepalive = {}, {}, {}
    for method, call in methods(d, plan, args.config_dir, args.variants):
        if method.scaling != "block128":
            raise ValueError("The power comparison uses K128 block scaling")
        errors[method.name] = error_metric(call, d.out, ri, ci, ref)
        graphs[method.name] = capture(call, stream, args.graph_ops)
        keepalive[method.name] = call
    if set(graphs) != set(args.variants):
        raise ValueError(
            f"No qualified call for {set(args.variants)-set(graphs)} on {args.shape}"
        )
    measure_graphs(args, graphs, {"relative_rms_and_max_over_rms": errors})


def measure_graphs(args, graphs, metadata):
    props = torch.cuda.get_device_properties(0)
    assert props.multi_processor_count == 304 and props.gcnArchName.startswith("gfx942")
    amdsmi.amdsmi_init()
    handle, bdf = identify_gpu(props)
    resolution = amdsmi.amdsmi_get_energy_count(handle)["counter_resolution"]
    cap = amdsmi.amdsmi_get_power_cap_info(handle)["power_cap"] / 1e6
    calibration = {name: duration(graph, 4) / 4 for name, graph in graphs.items()}
    record = dict(
        shape=args.shape,
        physical_bdf=bdf,
        physical_uuid=str(props.uuid),
        power_cap_w=cap,
        energy_resolution_uj=resolution,
        graph_ops=args.graph_ops,
        rounds=args.rounds,
        telemetry_poll="concurrent, 100 ms",
        requested_warm_s=args.warm_s,
        requested_measure_s=args.measure_s,
        order=args.variants,
        **metadata,
        trials=[],
    )
    # Start the measured rotation with the board already under load.
    first = args.variants[0]
    duration(graphs[first], math.ceil(10 / calibration[first]))
    for trial in range(args.rounds):
        # Pair each order with its reverse, rotating the first variant across pairs.
        shift = (trial // 2) % len(graphs)
        order = args.variants[shift:] + args.variants[:shift]
        if trial % 2:
            order.reverse()
        for name in order:
            warm_count = max(1, math.ceil(args.warm_s / calibration[name]))
            sec_per_replay = duration(graphs[name], warm_count) / warm_count
            reps = max(1, math.ceil(args.measure_s / sec_per_replay))
            data = sustained(
                graphs[name], reps, handle, resolution, args.graph_ops, args.shape
            )
            row = dict(trial=trial, variant=name, replays=reps, **data)
            record["trials"].append(row)
            print(
                trial,
                name,
                {
                    key: round(row["result"][key], 2)
                    for key in (
                        "us",
                        "mean_energy_w",
                        "median_gfx_mhz",
                        "ppt_pct",
                        "median_socket_w",
                    )
                },
                flush=True,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.output.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=2) + "\n")
            tmp.replace(args.output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variants", nargs="+", default=["block_register", "block_lds_n", "block_asm"]
    )
    p.add_argument("--shape", type=int, nargs=3, default=[8192, 4096, 32768])
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--warm-s", type=float, default=3)
    p.add_argument("--measure-s", type=float, default=6)
    p.add_argument("--graph-ops", type=int, default=8)
    p.add_argument(
        "--config-dir", type=Path, default=Path(__file__).resolve().parent / "configs"
    )
    p.add_argument("--output", type=Path, required=True)
    with torch.inference_mode():
        run(p.parse_args())
