# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"Wrapper macro to define default layer groups for python applications"

def default_layer_groups():
    return {
        "000_interpreter": "\\\\.runfiles/[^/]+\\\\+\\\\+python\\\\+python_[^/]+/",  # Python interpreter, typically at .runfiles/rules_python++python+python_3_13_x86_64-unknown-linux-gnu
        "001_base_system": "\\\\.runfiles/.*/site-packages/(numpy|scipy|pillow|requests|urllib3|setuptools|wheel|pip)",  # Stable system packages
        "002_ml_frameworks": "\\\\.runfiles/.*/site-packages/(torch|torchvision|pytorch_lightning|kornia|einops)",  # ML frameworks (large, stable)
        "003_cuda_libs": "\\\\.runfiles/.*/site-packages/nvidia_.*",  # NVIDIA CUDA libraries (large, stable)
        "004_cuda_toolkit": "\\\\.runfiles/_main/bazel/cuda_toolkit/",  # Hermetic CUDA toolkit (large, stable)
        "005_model_files_trt": "\\\\.runfiles/.*trt_engine_.*\\\\.(engine|ckpt|pt|pth|onnx)$$",  # TensorRT model files (large, stable)
        "006_model_files": "\\\\.runfiles/[^/]+_weights/file/.*\\\\.(engine|ckpt|pt|pth|onnx)$$",  # Other model files (large, stable)
        "007_rest_packages_a_p": "\\\\.runfiles/.*/site-packages/[a-p].*",  # Rest of the packages, package names starting with A-P
        "008_rest_packages_q_z": "\\\\.runfiles/.*/site-packages/[q-z].*",  # Rest of the packages, package names starting with Q-Z
        "009_libtorch": "\\\\.runfiles/.*/libtorch.*",  # LibTorch
        "010_system_so": "\\\\.runfiles/_main/_solib_.*$$",  # System shared libraries
        "011_nurec_libs": "\\\\.runfiles/_main/.*\\\\.(slang-module|so)$$",  # Compiled NuRec libraries and modules
        "012_nurec_code_proto": "\\\\.runfiles/_main/.*\\\\.(proto)$$",  # NuRec Proto code (changes frequently)
        "013_nurec_code_slang": "\\\\.runfiles/_main/.*\\\\.slang$$",  # NuRec Slang code (changes frequently)
        "014_nurec_markdown": "\\\\.runfiles/_main/.*\\\\.(md)$$",  # NuRec Markdown code (changes frequently)
        "015_nurec_code_cu": "\\\\.runfiles/_main/.*\\\\.(cu|cuh)$$",  # NuRec CUDA code (changes frequently)
        "016_nurec_code_cpp": "\\\\.runfiles/_main/.*\\\\.(cpp|h|hpp)$$",  # NuRec C++ code (changes frequently)
        "017_nurec_code_py": "\\\\.runfiles/_main/.*\\\\.(py|pyi)$$",  # NuRec Python code (changes frequently)
        "018_nurec_code_sh": "\\\\.runfiles/_main/.*\\\\.(sh|bash)$$",  # NuRec Shell code (changes frequently)
        "019_configs": "\\\\.runfiles/_main/([^/]+/)?configs/.*",  # Configuration files (changes frequently)
        "020_nurec_other": "\\\\.runfiles/_main/.*",  # Remaining NuRec files (changes frequently)
    }
