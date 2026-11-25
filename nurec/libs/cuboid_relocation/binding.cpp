// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cuboid_relocation.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("constrain_perturbation", &constrain_perturbation, "Reject outward cuboid noise (CUDA)",
          py::arg("positions"), py::arg("proposal"), py::arg("ids"), py::arg("tracks"),
          py::arg("centers"), py::arg("rotations"), py::arg("dimensions"), py::arg("expansion"));
    m.def("prepare_tracks", &prepare_tracks, "Prepare track relocation metadata (CUDA)",
          py::arg("order"), py::arg("offsets"), py::arg("weights"), py::arg("dead"),
          py::arg("sorted_dead"), py::arg("metadata"), py::arg("group"));
    m.def("track_cdf", &track_cdf, "Build independent double-precision track CDFs (CUDA)",
          py::arg("weights"), py::arg("offsets"), py::arg("cdf"));
    m.def("sample_tracks", &sample_tracks, "Sample track-local donors from supplied uniforms (CUDA)",
          py::arg("rows"), py::arg("cdf"), py::arg("offsets"), py::arg("draw_offsets"),
          py::arg("uniforms"), py::arg("sources"));
    m.def("correct_relocation", &correct_relocation, "Count donors and correct relocation (CUDA)",
          py::arg("order"), py::arg("sorted_dead"), py::arg("group"), py::arg("metadata"),
          py::arg("sources"), py::arg("opacities"), py::arg("scales"), py::arg("binoms"),
          py::arg("min_opacity"), py::arg("occurrences"), py::arg("keep"),
          py::arg("revised_opacities"), py::arg("revised_scales"));
}
