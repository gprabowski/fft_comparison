#pragma once

#include <fft.hpp>
#include <utils.h>

namespace fft {
namespace alg {
namespace cpu {

template <typename DT, typename Size>
struct ibb_third : fft_functor<ibb_third<DT, Size>,
                               fft_properties<DT, Size::value, false, false>> {
  using this_type = ibb_third<DT, Size>;
  static void exec_impl(std::vector<std::complex<DT>> &data) {
    constexpr int N = Size::value;

    const auto twiddles = utils::get_roots_of_unity<N, false, DT>();

    int pairs_in_groups = N / 2;
    int num_groups = 1;
    int distance = N / 2;

    auto data_out = data.data();
    auto data2 = data;

    while (num_groups < N) {
      int l = 0;
      for (int k = 0; k < num_groups; ++k) {
        const auto j_first = 2 * k * pairs_in_groups;
        const auto j_last = j_first + pairs_in_groups - 1;
        const auto j_twiddle = k * pairs_in_groups;
        const auto twiddle = twiddles[j_twiddle];
        for (int j = j_first; j <= j_last; ++j) {
          const auto temp = twiddle * data[j + distance];
          data2[l] = data[j] + temp;
          data2[l + N / 2] = data[j] - temp;
          ++l;
        }
      }
      std::swap(data, data2);
      pairs_in_groups /= 2;
      num_groups *= 2;
      distance /= 2;
    }

    if (data.data() != data_out) {
      data2 = data;
    }
  }
};
} // namespace cpu
} // namespace alg
} // namespace fft
