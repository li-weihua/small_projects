#include <c10/core/ScalarType.h>
#include <chrono>
#include <ratio>

#include "pocketfft_hdronly.h"
#include "torch/torch.h"

using pocketfft::shape_t;
using pocketfft::stride_t;

torch::Tensor DoRfft(const torch::Tensor& input, bool do_benchmark) {
  TORCH_CHECK(input.dtype() == torch::kFloat32);
  TORCH_CHECK(input.dim() == 1);

  auto options = torch::TensorOptions().dtype(torch::kComplexFloat);

  auto output = torch::empty({input.size(0) / 2 + 1}, options);

  shape_t shape_in = {(size_t)input.size(0)};

  stride_t stride_in = {sizeof(float)};
  stride_t stride_out = {sizeof(float) * 2};

  size_t axis = 0;
  bool forward = true;

  float* data_in = reinterpret_cast<float*>(input.data_ptr());
  std::complex<float>* data_out =
      reinterpret_cast<std::complex<float>*>(output.data_ptr());

  pocketfft::r2c<float>(shape_in, stride_in, stride_out, axis, forward, data_in,
                        data_out, 1.0f);

  if (do_benchmark) {
    constexpr int kRepeats = 10;
    float time[kRepeats] = {0};

    for (int i = 0; i < kRepeats; ++i) {
      auto start = std::chrono::high_resolution_clock::now();

      pocketfft::r2c<float>(shape_in, stride_in, stride_out, axis, forward, data_in,
                            data_out, 1.0f);

      auto stop = std::chrono::high_resolution_clock::now();

      time[i] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                    .count();
    }

    for (int i = 0; i < kRepeats; ++i) {
      std::cout << "iter [" << i << "] : " << time[i] << std::endl;
    }
  }

  return output;
}

torch::Tensor DoIrfft(const torch::Tensor& input, bool do_benchmark) {
  TORCH_CHECK(input.dtype() == torch::kComplexFloat);
  TORCH_CHECK(input.dim() == 1);

  auto options = torch::TensorOptions().dtype(torch::kFloat32);

  auto output = torch::empty({(input.size(0) - 1) * 2}, options);

  shape_t shape_out = {(size_t)output.size(0)};

  float scale = 1.0f / output.size(0);

  stride_t stride_in = {sizeof(float) * 2};
  stride_t stride_out = {sizeof(float)};

  size_t axis = 0;
  bool forward = false;

  std::complex<float>* data_in =
      reinterpret_cast<std::complex<float>*>(input.data_ptr());

  float* data_out = reinterpret_cast<float*>(output.data_ptr());

  pocketfft::c2r<float>(shape_out, stride_in, stride_out, axis, forward, data_in,
                        data_out, scale);

  if (do_benchmark) {
    constexpr int kRepeats = 10;
    float time[kRepeats] = {0};

    for (int i = 0; i < kRepeats; ++i) {
      auto start = std::chrono::high_resolution_clock::now();

      pocketfft::c2r<float>(shape_out, stride_in, stride_out, axis, forward, data_in,
                            data_out, scale);

      auto stop = std::chrono::high_resolution_clock::now();

      time[i] = std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                    .count();
    }

    for (int i = 0; i < kRepeats; ++i) {
      std::cout << "iter [" << i << "] : " << time[i] << std::endl;
    }
  }

  return output;
}

TORCH_LIBRARY_FRAGMENT(top, m) {
  m.def("rfft", DoRfft);
  m.def("irfft", DoIrfft);
}
