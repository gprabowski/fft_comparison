//
// Created by gprab on 3/26/2023.
//

#ifndef FFT_COMPARISON_UTILS_H
#define FFT_COMPARISON_UTILS_H

#include <array>
#include <cmath>
#include <complex>
#include <vector>

namespace fft {
namespace utils {
namespace constants {
constexpr double pi = 3.141592653;
constexpr double tau = 2.0 * pi;
} // namespace constants

template <unsigned int N> struct int_ {
  static constexpr unsigned int value = N;
};

constexpr bool is_power_of_2(unsigned int N) { return (N & (N - 1)) == 0; }

constexpr unsigned int log_n(unsigned int N) {
  int res = 0;
#pragma unroll 32
  for (int i = 0; i < 32; ++i) {
    if ((1 << res) == N)
      return res;
    ++res;
  }
  return res;
}

template <unsigned int N> constexpr unsigned int bit_reverse(unsigned int num) {
  unsigned int ret = 0;
#pragma unroll N
  for (unsigned int i = 0; i < N; ++i) {
    ret += ((num & (1 << i)) > 0) * (1 << (N - 1 - i));
  }

  return ret;
}

template <unsigned int N, typename DT = float>
std::vector<std::complex<DT>> get_roots_of_unity() {
  std::vector<std::complex<DT>> ret(N / 2);
  constexpr DT theta = constants::tau / N;
  for (int i = 0; i < N / 2; ++i) {
    ret[i] = {std::cos(i * theta), std::sin(i * theta)};
  }
  return ret;
}

template <unsigned int N, typename DT = float>
std::vector<std::complex<DT>> get_roots_of_unity_singleton() {
  std::vector<std::complex<DT>> ret(N / 2);
  constexpr DT theta = constants::tau / N;
  const DT s = std::sin(theta);
  const DT c = 1.0 - 2 * sin(theta / 2.0) * sin(theta / 2.0);
  ret[0].real(1.0);
  ret[0].imag(0.0);

#pragma unroll N / 8
  for (int k = 0; k < N / 8 - 1; ++k) {
    ret[k + 1].real(c * ret[k].real() - s * ret[k].imag());
    ret[k + 1].imag(s * ret[k].real() + c * ret[k].imag());
  }

  constexpr int l1 = N / 8;
  ret[l1].real(sqrt(2) / 2.f);
  ret[l1].imag(ret[l1].real());

#pragma unroll(N / 8 - 1)
  for (int k = 1; k < N / 8; ++k) {
    ret[l1 + k].real(ret[l1 - k].imag());
    ret[l1 + k].imag(ret[l1 - k].real());
  }

  constexpr int l2 = N / 4;
  ret[l2].real(0);
  ret[l2].imag(1);

#pragma unroll(N / 4 - 1)
  for (int k = 1; k < N / 4; ++k) {
    ret[l2 + k].real(-ret[l2 - k].real());
    ret[l2 + k].imag(ret[l2 - k].imag());
  }

  return ret;
}

template <typename DT> constexpr std::array<DT, 32> get_inv_powers() {
  std::array<DT, 32> ret;
  DT div = DT(2.0);
#pragma unroll 32
  for (int i = 0; i < 32; ++i) {
    ret[i] = 1 / div;
    div *= DT(2.0);
  }

  return ret;
}
} // namespace utils
} // namespace fft

#endif // FFT_COMPARISON_UTILS_H
