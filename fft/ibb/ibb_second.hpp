#pragma once

#include <fft.hpp>
#include <utils.h>

namespace fft {
namespace alg {
namespace cpu {
template <typename DT, typename Size>
struct ibb_second : fft_functor<ibb_second<DT, Size>,
                                fft_properties<DT, Size::value, true, false>> {
  static constexpr int size = Size::value;
  using this_type = ibb_second<DT, Size>;
  static void exec_impl(std::vector<std::complex<DT>> &data) {
    constexpr int N = Size::value;

    const auto twiddles = utils::get_roots_of_unity<N, false, DT>();

    int pairs_in_groups = N / 2;
    int num_groups = 1;
    int distance = 1;

    while (num_groups < N) {
      auto gap_to_next_pair = 2 * num_groups;
      auto gap_to_last_pair = gap_to_next_pair * (pairs_in_groups - 1);
      for (int k = 0; k < num_groups; ++k) {
        auto j = k;
        const auto j_last = k + gap_to_last_pair;
        const auto j_twiddle = k * pairs_in_groups;
        const auto twiddle = twiddles[j_twiddle];
        while (j <= j_last) {
          const auto temp = twiddle * data[j + distance];
          data[j + distance] = data[j] - temp;
          data[j] = data[j] + temp;
          j += gap_to_next_pair;
        }
      }
      pairs_in_groups /= 2;
      num_groups *= 2;
      distance *= 2;
    }
  }
};
} // namespace cpu
} // namespace alg
} // namespace fft
