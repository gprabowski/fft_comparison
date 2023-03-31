//
// Created by gprab on 3/26/2023.
//

#include <array>
#include <complex>
#include <vector>

#include <fftw3.h>

#ifndef FFT_COMPARISON_CPU_FFT_H
#define FFT_COMPARISON_CPU_FFT_H

namespace fft {
namespace alg {
namespace cpu {
// assumes bit reversed order of twiddles
template <typename DT, typename Size> struct ibb_first {
  static constexpr int size = Size::value;
  static void execute(std::vector<std::complex<DT>> &data,
                      const std::vector<std::complex<DT>> &twiddles) {
    constexpr int N = Size::value;
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

template <typename DT, typename Size> struct ibb_second {
  static constexpr int size = Size::value;
  static void execute(std::vector<std::complex<DT>> &data,
                      const std::vector<std::complex<DT>> &twiddles) {
    constexpr int N = Size::value;
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

template <typename DT, typename Size> struct ibb_third {
  static constexpr int size = Size::value;
  static void execute(std::vector<std::complex<DT>> &data,
                      const std::vector<std::complex<DT>> &twiddles) {
    constexpr int N = Size::value;
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
        for (int j = j_first; j < j_last; ++j) {
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

template <typename DT, typename Size> struct fftw {
  static constexpr int size = Size::value;
  static void execute(std::vector<std::complex<DT>> &data,
                      const std::vector<std::complex<DT>> &twiddles) {
    constexpr int N = size;

    fftw_complex *in, *out;
    fftw_plan p;

    in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);

    for (int i = 0; i < N; ++i) {
      in[i][0] = data[i].real();
      in[i][1] = data[i].imag();
    }
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p);

    for (int i = 0; i < size; ++i) {
      data[i] = {out[i][0], out[i][1]};
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
  }
};
} // namespace cpu
} // namespace alg
} // namespace fft

#endif // FFT_COMPARISON_CPU_FFT_H
