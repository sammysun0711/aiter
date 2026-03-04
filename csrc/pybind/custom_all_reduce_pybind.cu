// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "custom_all_reduce.h"
#include "communication_asm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      CUSTOM_ALL_REDUCE_PYBIND;
      // Standard RMSNorm (10 args, backward compatible)
      m.def("fused_allreduce_rmsnorm",
            &aiter::fused_allreduce_rmsnorm,
            py::arg("_fa"),
            py::arg("inp"),
            py::arg("res_inp"),
            py::arg("res_out"),
            py::arg("out"),
            py::arg("w"),
            py::arg("eps"),
            py::arg("reg_buffer") = std::nullopt,
            py::arg("use_1stage") = false);
      m.def("fused_allreduce_rmsnorm_quant",
            &aiter::fused_allreduce_rmsnorm_quant,
            py::arg("_fa"),
            py::arg("inp"),
            py::arg("res_inp"),
            py::arg("res_out"),
            py::arg("out"),
            py::arg("scale_out"),
            py::arg("w"),
            py::arg("eps"),
            py::arg("reg_buffer") = std::nullopt,
            py::arg("use_1stage") = false);
      // Gemma RMSNorm (separate API, same 10 args)
      m.def("fused_allreduce_gemma_rmsnorm",
            &aiter::fused_allreduce_gemma_rmsnorm,
            py::arg("_fa"),
            py::arg("inp"),
            py::arg("res_inp"),
            py::arg("res_out"),
            py::arg("out"),
            py::arg("w"),
            py::arg("eps"),
            py::arg("reg_buffer") = std::nullopt,
            py::arg("use_1stage") = false);
      m.def("fused_allreduce_gemma_rmsnorm_quant",
            &aiter::fused_allreduce_gemma_rmsnorm_quant,
            py::arg("_fa"),
            py::arg("inp"),
            py::arg("res_inp"),
            py::arg("res_out"),
            py::arg("out"),
            py::arg("scale_out"),
            py::arg("w"),
            py::arg("eps"),
            py::arg("reg_buffer") = std::nullopt,
            py::arg("use_1stage") = false);
}