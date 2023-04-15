#pragma once

#include <fft.hpp>
#include <utils.h>

namespace fft {
namespace alg {
namespace cpu {

template <typename DT, typename Size>
struct edp_first : fft_functor<edp_first<DT, Size>,
                               fft_properties<DT, Size::value, false, true>> {
  static constexpr int size = Size::value;
  static void FFT_Butterflies(int m, // radix
                              std::vector<std::complex<DT>> &out,
                              std::vector<std::complex<DT>> &in, int k0, int c0,
                              int p) {
    // coefficient for k1 is coefficient for k0 divided by 1 << m
    const int c1 = (c0 >> m);
    for (int k2 = 0; k2 < c1; ++k2) {
      for (int k1 = 0; k1 < (1 << m); ++k1) {
        std::complex<DT> sum = {0.0, 0.0};
        for (int j1 = 0; j1 < (1 << m); ++j1) {
          sum += utils::one(-j1 * utils::edp_rev<DT>(k1, m)) *
                 utils::one(-j1 * utils::edp_rev<DT>((1 << m) * k0, m + p)) *
                 in[c0 * k0 + c1 * j1 + k2];
        }
        out[c0 * k0 + c1 * k1 + k2] = sum;
      }
    }
  }

  static void exec_impl(std::vector<std::complex<DT>> &data) {
    // apply the algorithm assuming out of place computation
    // and all m being 1
    constexpr auto log_size = utils::log_n(size);
    std::vector<std::vector<std::complex<DT>>> v;
    v.resize(log_size + 1);
    for (auto &vec : v) {
      vec.resize(size);
    }

    std::copy(begin(data), end(data), begin(v[0]));

    const int P = log_size;
    for (int p = 0; p < P; ++p) {
      for (int k0 = 0; k0 < (1 << p); ++k0) {
        FFT_Butterflies(1, v[p + 1], v[p], k0, 1 << (log_size - p), p);
      }
    }

    std::copy(begin(v[log_size]), end(v[log_size]), begin(data));
  }
};
} // namespace cpu
} // namespace alg
} // namespace fft
