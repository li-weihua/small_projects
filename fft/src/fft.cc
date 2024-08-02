#include <chrono>
#include <ratio>

#include "pocketfft_hdronly.h"
#include "torch/torch.h"

torch::Tensor DoRfft(const torch::Tensor& input, bool do_benchmark) {
  TORCH_CHECK(input.dtype() == torch::kFloat32);
  TORCH_CHECK(input.dim() == 1);

  using pocketfft::shape_t;
  using pocketfft::stride_t;

  auto options = torch::TensorOptions().dtype(torch::kComplexFloat);

  auto output = torch::empty({input.size(0) / 2 + 1}, options);

  shape_t shape_in = {(size_t)input.size(0)};

  stride_t stride_in(shape_in.size(), sizeof(float));
  stride_t stride_out(shape_in.size() / 2 + 1, sizeof(float) * 2);

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

TORCH_LIBRARY_FRAGMENT(top, m) { m.def("rfft", DoRfft); }
