#pragma once

#include <fft.hpp>
#include <utils.h>

namespace fft {
namespace alg {
namespace cpu {

template <typename DT, typename Size>
struct ibb_first : fft_functor<ibb_first<DT, Size>,
                               fft_properties<DT, Size::value, false, true>> {
  using this_type = ibb_first<DT, Size>;
  static void exec_impl(std::vector<std::complex<DT>> &data) {
    constexpr int N = Size::value;

    const auto twiddles = utils::get_roots_of_unity<N, true, DT>();

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
