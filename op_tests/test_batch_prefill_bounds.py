# SPDX-License-Identifier: MIT
"""GLOBAL_LOAD_LDS must zero padded head lanes without touching guard pages."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip,
    reason="requires a ROCm GPU with HIP virtual memory",
)


@pytest.fixture(scope="module")
def guard_library(tmp_path_factory):
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith(
        ("gfx942", "gfx950")
    ):
        pytest.skip("GLOBAL_LOAD_LDS regression targets gfx942/gfx950")
    library = tmp_path_factory.mktemp("ck_guard") / "guard.so"
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    subprocess.run(
        [
            "g++",
            "-shared",
            "-fPIC",
            "-O2",
            "-D__HIP_PLATFORM_AMD__",
            f"-I{rocm}/include",
            str(Path(__file__).with_name("ck_prefill_guard.cpp")),
            f"-L{rocm}/lib",
            f"-Wl,-rpath,{rocm}/lib",
            "-lamdhip64",
            "-o",
            str(library),
        ],
        check=True,
    )
    return library


@pytest.mark.parametrize(
    "q_lens,kv_lens,qk_dim,v_dim,permute",
    [
        ([3968], [16384], 192, 128, False),
        ([129], [257], 192, 128, False),
        ([64], [129], 160, 96, False),
        ([33], [97], 96, 64, False),
        ([7, 63, 129], [73, 192, 257], 192, 128, True),
    ],
)
@pytest.mark.parametrize("cache_dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("load_mode", ["global", "buffer"])
def test_global_load_padded_heads(
    guard_library, q_lens, kv_lens, qk_dim, v_dim, permute, cache_dtype, load_mode
):
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("ck_prefill_guard.py")),
        "--library",
        str(guard_library),
        "--q-lens",
        *map(str, q_lens),
        "--kv-lens",
        *map(str, kv_lens),
        "--qk-dim",
        str(qk_dim),
        "--v-dim",
        str(v_dim),
        "--cache-dtype",
        cache_dtype,
        "--load-mode",
        load_mode,
    ]
    if permute:
        cmd.append("--permute")
    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "HIP_LAUNCH_BLOCKING": "1"},
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert '"passed": true' in result.stdout, result.stdout
