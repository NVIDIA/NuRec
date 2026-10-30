// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

constexpr size_t kMaxBatchSize = 16;

class NvjpegOutOfMemoryError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class NvjpegImageEncodingError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class NvjpegEncoderPoisonedError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void check_cuda(cudaError_t status, const char* call) {
    if (status != cudaSuccess) {
        const std::string message = std::string(call) + " failed: " + cudaGetErrorString(status);
        if (status == cudaErrorMemoryAllocation) {
            throw NvjpegOutOfMemoryError(message);
        }
        throw std::runtime_error(message);
    }
}

void check_nvjpeg(nvjpegStatus_t status, const char* call) {
    if (status != NVJPEG_STATUS_SUCCESS) {
        const std::string message =
            std::string(call) + " failed with nvjpeg status " + std::to_string(status);
        if (status == NVJPEG_STATUS_ALLOCATOR_FAILURE) {
            throw NvjpegOutOfMemoryError(message);
        }
        throw std::runtime_error(message);
    }
}

void check_nvjpeg_image(nvjpegStatus_t status, const char* call) {
    if (status != NVJPEG_STATUS_SUCCESS) {
        const std::string message =
            std::string(call) + " failed with nvjpeg status " + std::to_string(status);
        if (status == NVJPEG_STATUS_ALLOCATOR_FAILURE) {
            throw NvjpegOutOfMemoryError(message);
        }
        throw NvjpegImageEncodingError(message);
    }
}

void validate_image(const torch::Tensor& image, int expected_device) {
    if (!image.is_cuda()) {
        throw NvjpegImageEncodingError("nvJPEG C++ encoder expects CUDA tensors.");
    }
    if (image.scalar_type() != torch::kUInt8) {
        throw NvjpegImageEncodingError("nvJPEG C++ encoder expects uint8 tensors.");
    }
    if (image.dim() != 3 || image.size(2) != 3) {
        throw NvjpegImageEncodingError("nvJPEG C++ encoder expects HWC RGB tensors.");
    }
    if (image.size(0) <= 0 || image.size(1) <= 0) {
        throw NvjpegImageEncodingError("nvJPEG C++ encoder expects non-empty images.");
    }
    if (!image.is_contiguous()) {
        throw NvjpegImageEncodingError("nvJPEG C++ encoder expects contiguous tensors.");
    }
    if (image.get_device() != expected_device) {
        throw NvjpegImageEncodingError("All images in a batch must be on the same CUDA device.");
    }
}

struct ImageInfo {
    unsigned char* data = nullptr;
    unsigned int pitch  = 0;
    int width           = 0;
    int height          = 0;
};

} // namespace

class NvjpegBatchEncoder {
public:
    explicit NvjpegBatchEncoder(int device)
        : device_(device) {
        c10::cuda::CUDAGuard device_guard(device_);

        nvjpegHandle_t handle = nullptr;
        check_nvjpeg(nvjpegCreateSimple(&handle), "nvjpegCreateSimple");
        try {
            nvjpegEncoderParams_t params = nullptr;
            check_nvjpeg(nvjpegEncoderParamsCreate(handle, &params, nullptr), "nvjpegEncoderParamsCreate");
            handle_ = handle;
            params_ = params;
        } catch (...) {
            nvjpegDestroy(handle);
            throw;
        }
    }

    NvjpegBatchEncoder(const NvjpegBatchEncoder&)            = delete;
    NvjpegBatchEncoder& operator=(const NvjpegBatchEncoder&) = delete;

    ~NvjpegBatchEncoder() {
        try {
            c10::cuda::CUDAGuard device_guard(device_);
            for (auto state : states_) {
                if (state != nullptr) {
                    nvjpegEncoderStateDestroy(state);
                }
            }
            if (params_ != nullptr) {
                nvjpegEncoderParamsDestroy(params_);
            }
            if (handle_ != nullptr) {
                nvjpegDestroy(handle_);
            }
        } catch (...) {
            // Destructors must not throw. CUDA process teardown can make setting the
            // original device fail, in which case there is no safe recovery path.
        }
    }

    std::vector<py::bytes> encode_batch(const std::vector<torch::Tensor>& images, int quality) {
        if (images.empty()) {
            return {};
        }
        if (quality <= 0 || quality > 100) {
            throw std::runtime_error("JPEG quality must be in [1, 100].");
        }
        if (images.size() > kMaxBatchSize) {
            throw std::runtime_error(
                "nvJPEG batch size exceeds the maximum of " + std::to_string(kMaxBatchSize) + ".");
        }
        if (!images.front().is_cuda()) {
            throw std::runtime_error("nvJPEG C++ encoder expects CUDA tensors.");
        }

        const int device = images.front().get_device();
        if (device != device_) {
            throw std::runtime_error(
                "NvjpegBatchEncoder was created for CUDA device " + std::to_string(device_) +
                " but received an image on device " + std::to_string(device) + ".");
        }
        std::vector<ImageInfo> image_infos;
        image_infos.reserve(images.size());
        for (const torch::Tensor& image : images) {
            validate_image(image, device);
            ImageInfo image_info;
            image_info.data   = static_cast<unsigned char*>(image.data_ptr<std::uint8_t>());
            image_info.pitch  = static_cast<unsigned int>(image.size(1) * image.size(2));
            image_info.width  = static_cast<int>(image.size(1));
            image_info.height = static_cast<int>(image.size(0));
            image_infos.push_back(image_info);
        }

        std::vector<size_t> encoded_sizes(images.size());
        std::vector<std::vector<unsigned char>> encoded_buffers(images.size());
        {
            py::gil_scoped_release release;
            // Acquire the encoder mutex without holding the GIL. The mutex must also be released
            // before release reacquires the GIL, otherwise two Python threads can deadlock:
            // one waits for this mutex while holding the GIL and its owner waits for the GIL while
            // still holding the mutex.
            std::lock_guard<std::mutex> lock(mutex_);

            c10::cuda::CUDAGuard device_guard(device);
            cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();

            if (poisoned_) {
                throw NvjpegEncoderPoisonedError(
                    "NvjpegBatchEncoder is unusable after an asynchronous CUDA failure.");
            }

            try {
                ensure_state_count(images.size(), stream);
                configure_params(quality, stream);

                for (size_t i = 0; i < image_infos.size(); ++i) {
                    const ImageInfo& image_info = image_infos[i];
                    nvjpegImage_t nvjpeg_image{};
                    nvjpeg_image.channel[0] = image_info.data;
                    nvjpeg_image.pitch[0]   = image_info.pitch;

                    check_nvjpeg_image(
                        nvjpegEncodeImage(
                            handle_,
                            states_[i],
                            params_,
                            &nvjpeg_image,
                            NVJPEG_INPUT_RGBI,
                            image_info.width,
                            image_info.height,
                            stream),
                        "nvjpegEncodeImage");
                }

                for (size_t i = 0; i < image_infos.size(); ++i) {
                    check_nvjpeg_image(
                        nvjpegEncodeRetrieveBitstream(handle_, states_[i], nullptr, &encoded_sizes[i], stream),
                        "nvjpegEncodeRetrieveBitstream(size)");
                }

                // The size retrievals are stream-ordered, so the encodes must complete before the
                // host may read encoded_sizes to allocate the output buffers.
                check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(sizes)");

                for (size_t i = 0; i < image_infos.size(); ++i) {
                    encoded_buffers[i].resize(encoded_sizes[i]);
                    check_nvjpeg_image(
                        nvjpegEncodeRetrieveBitstream(
                            handle_,
                            states_[i],
                            encoded_buffers[i].data(),
                            &encoded_sizes[i],
                            stream),
                        "nvjpegEncodeRetrieveBitstream(data)");
                }

                check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
            } catch (...) {
                // Keep input tensors, encoder states, and output buffers alive until all work
                // submitted before the failure has drained. If the stream itself is broken,
                // prevent this encoder from reusing potentially in-flight state.
                if (cudaStreamSynchronize(stream) != cudaSuccess) {
                    poisoned_ = true;
                    throw NvjpegEncoderPoisonedError(
                        "NvjpegBatchEncoder is unusable after an asynchronous CUDA failure.");
                }
                throw;
            }
        }

        std::vector<py::bytes> outputs;
        outputs.reserve(encoded_buffers.size());
        for (size_t i = 0; i < encoded_buffers.size(); ++i) {
            outputs.emplace_back(
                reinterpret_cast<const char*>(encoded_buffers[i].data()), encoded_sizes[i]);
        }
        return outputs;
    }

private:
    void ensure_state_count(size_t count, cudaStream_t stream) {
        while (states_.size() < count) {
            nvjpegEncoderState_t state = nullptr;
            check_nvjpeg(nvjpegEncoderStateCreate(handle_, &state, stream), "nvjpegEncoderStateCreate");
            try {
                states_.push_back(state);
            } catch (...) {
                nvjpegEncoderStateDestroy(state);
                throw;
            }
        }
    }

    void configure_params(int quality, cudaStream_t stream) {
        if (configured_quality_ != quality) {
            check_nvjpeg(nvjpegEncoderParamsSetQuality(params_, quality, stream), "nvjpegEncoderParamsSetQuality");
            configured_quality_ = quality;
        }
        if (!configured_sampling_factor_) {
            check_nvjpeg(
                nvjpegEncoderParamsSetSamplingFactors(params_, NVJPEG_CSS_444, stream),
                "nvjpegEncoderParamsSetSamplingFactors");
            configured_sampling_factor_ = true;
        }
    }

    std::mutex mutex_;
    int device_                   = 0;
    nvjpegHandle_t handle_        = nullptr;
    nvjpegEncoderParams_t params_ = nullptr;
    std::vector<nvjpegEncoderState_t> states_;
    int configured_quality_          = 0;
    bool configured_sampling_factor_ = false;
    bool poisoned_                   = false;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::register_exception<NvjpegOutOfMemoryError>(m, "NvjpegOutOfMemoryError");
    py::register_exception<NvjpegImageEncodingError>(m, "NvjpegImageEncodingError");
    py::register_exception<NvjpegEncoderPoisonedError>(m, "NvjpegEncoderPoisonedError");
    m.attr("MAX_BATCH_SIZE") = py::int_(kMaxBatchSize);
    py::class_<NvjpegBatchEncoder>(m, "NvjpegBatchEncoder")
        .def(py::init<int>(), py::arg("device"))
        .def("encode_batch", &NvjpegBatchEncoder::encode_batch, py::arg("images"), py::arg("quality"));
}
