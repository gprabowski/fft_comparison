//
// Created by gprab on 3/26/2023.
//

#include <array>
#include <complex>
#include <vector>

#ifndef FFT_COMPARISON_CPU_FFT_H
#define FFT_COMPARISON_CPU_FFT_H

namespace fft {
namespace alg {
namespace cpu {
// assumes bit reversed order of twiddles
template <typename DT, int N> struct ibb_first {
  static void execute(std::vector<std::complex<DT>> &data,
                      const std::vector<std::complex<DT>> &twiddles) {
    int pairs_in_groups = N / 2;
    int num_groups = 1;
    int distance = N / 2;
    while (num_groups < N) {
      for (int k = 0; k < num_groups; ++k) {
        const int j_first = 2 * k * pairs_in_groups;
        const int j_last = j_first + pairs_in_groups - 1;
        const int j_twiddle = k;
        const auto twiddle = twiddles[j_twiddle];
        for (int j = j_first; j <= j_last; ++j) {
          const auto temp = twiddle * data[j + distance];
          data[j + distance] = data[j] - temp;
          data[j] = data[j] + temp;
        }
      }
      pairs_in_groups /= 2;
      num_groups *= 2;
      distance /= 2;
    }
  }
};
} // namespace cpu
} // namespace alg
} // namespace fft

#endif // FFT_COMPARISON_CPU_FFT_H
