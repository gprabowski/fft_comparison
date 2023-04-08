#pragma once

#include "utils.h"
#include <algorithm>
#include <array>
#include <complex>
#include <iostream>
#include <vector>

#include <fftw3.h>

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

template <typename DT, typename Size> struct edp_first {
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

  static void execute(std::vector<std::complex<DT>> &data,
                      const std::vector<std::complex<DT>> &twiddles) {
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

template <typename DT, typename Size> struct edp_second {
  static constexpr int size = Size::value;
  static constexpr DT tau = utils::constants::tau;
  using complex_t = std::complex<DT>;
  using com_arr = std::vector<complex_t>;

  static inline int ilog2(unsigned int n) { return 31 - __builtin_clz(n); }

  struct common_weight {
    DT w1i, w1r, w2i, w2r, w3i, w3r;
  };

  using weight_vec = std::vector<common_weight>;

  static unsigned int rw(unsigned int k) {
    static const unsigned char b[256] = {
        0,   128, 64,  192, 32,  160, 96,  224, 16,  144, 80,  208, 48,  176,
        112, 240, 8,   136, 72,  200, 40,  168, 104, 232, 24,  152, 88,  216,
        56,  184, 120, 248, 4,   132, 68,  196, 36,  164, 100, 228, 20,  148,
        84,  212, 52,  180, 116, 244, 12,  140, 76,  204, 44,  172, 108, 236,
        28,  156, 92,  220, 60,  188, 124, 252, 2,   130, 66,  194, 34,  162,
        98,  226, 18,  146, 82,  210, 50,  178, 114, 242, 10,  138, 74,  202,
        42,  170, 106, 234, 26,  154, 90,  218, 58,  186, 122, 250, 6,   134,
        70,  198, 38,  166, 102, 230, 22,  150, 86,  214, 54,  182, 118, 246,
        14,  142, 78,  206, 46,  174, 110, 238, 30,  158, 94,  222, 62,  190,
        126, 254, 1,   129, 65,  193, 33,  161, 97,  225, 17,  145, 81,  209,
        49,  177, 113, 241, 9,   137, 73,  201, 41,  169, 105, 233, 25,  153,
        89,  217, 57,  185, 121, 249, 5,   133, 69,  197, 37,  165, 101, 229,
        21,  149, 85,  213, 53,  181, 117, 245, 13,  141, 77,  205, 45,  173,
        109, 237, 29,  157, 93,  221, 61,  189, 125, 253, 3,   131, 67,  195,
        35,  163, 99,  227, 19,  147, 83,  211, 51,  179, 115, 243, 11,  139,
        75,  203, 43,  171, 107, 235, 27,  155, 91,  219, 59,  187, 123, 251,
        7,   135, 71,  199, 39,  167, 103, 231, 23,  151, 87,  215, 55,  183,
        119, 247, 15,  143, 79,  207, 47,  175, 111, 239, 31,  159, 95,  223,
        63,  191, 127, 255};
    unsigned char b0 = b[(k >> (0 * 8)) & 0xff], b1 = b[(k >> (1 * 8)) & 0xff],
                  b2 = b[(k >> (2 * 8)) & 0xff], b3 = b[(k >> (3 * 8)) & 0xff];
    return (b0 << (3 * 8)) | (b1 << (2 * 8)) | (b2 << (1 * 8)) |
           (b3 << (0 * 8));
  }

  static DT r(unsigned int k) { return DT(1.) / DT(4294967296.) * rw(k); }

  static void FFT4_1WeightPerCall(com_arr &data, int k0, int c0,
                                  const common_weight &weight) {
    const int c1 = c0 >> 2;
    int k2;

    DT a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, b1r, b1i, b2r, b2i, b3r, b3i,
        c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i, d0r, d0i, d1r, d1i, d2r, d2i,
        d3r, d3i;
    for (k2 = 0; k2 < c1; ++k2) {
      // Goedecker's algorithm for easy vectorization

      // Load data into registers
      a0r = data[c0 * k0 + c1 * 0 + k2].real();
      a0i = data[c0 * k0 + c1 * 0 + k2].imag();
      a1r = data[c0 * k0 + c1 * 1 + k2].real();
      a1i = data[c0 * k0 + c1 * 1 + k2].imag();
      a2r = data[c0 * k0 + c1 * 2 + k2].real();
      a2i = data[c0 * k0 + c1 * 2 + k2].imag();
      a3r = data[c0 * k0 + c1 * 3 + k2].real();
      a3i = data[c0 * k0 + c1 * 3 + k2].imag();

      // PERFORM GOEDECKER COMPUTATION
      b1r = -a1i * weight.w1i + a1r;
      b1i = +a1r * weight.w1i + a1i;
      b2r = -a2i * weight.w2i + a2r;
      b2i = +a2r * weight.w2i + a2i;
      b1r = -a3i * weight.w3i + a3r;
      b1i = +a3r * weight.w3i + a3i;

      c0r = +b2r * weight.w2r + a0r;
      c0i = +b2i * weight.w2r + a0i;
      c2r = -b2r * weight.w2r + a0r;
      c2i = -b2i * weight.w2r + a0i;
      c1r = +b3r * weight.w3r + b1r;
      c1i = +b3i * weight.w3r + b1i;
      c3r = -b3r * weight.w3r + b1r;
      c3i = -b3i * weight.w3r + b1i;

      d0r = +c1r * weight.w1r + c0r;
      d0i = +c1i * weight.w1r + c0i;
      d1r = -c1r * weight.w1r + c0r;
      d1i = -c1i * weight.w1r + c0i;
      d2r = -c3i * weight.w1r + c2r;
      d2i = +c3r * weight.w1r + c2i;
      d3r = +c3i * weight.w1r + c2r;
      d3i = -c3r * weight.w1r + c2i;
      // END OF GOEDECKER COMPUTATION

      // Store results
      data[c0 * k0 + c1 * 0 + k2].real(d0r);
      data[c0 * k0 + c1 * 0 + k2].imag(d0i);
      data[c0 * k0 + c1 * 1 + k2].real(d1r);
      data[c0 * k0 + c1 * 1 + k2].imag(d1i);
      data[c0 * k0 + c1 * 2 + k2].real(d2r);
      data[c0 * k0 + c1 * 2 + k2].imag(d2i);
      data[c0 * k0 + c1 * 3 + k2].real(d3r);
      data[c0 * k0 + c1 * 3 + k2].imag(d3i);
    }
  }

  static void FFT4_0Weights(com_arr &data, int c0) {
    const int c1 = c0 >> 2;
    int k2;
    DT a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, c0r, c0i, c1r, c1i, c2r, c2i,
        c3r, c3i, d0r, d0i, d1r, d1i, d2r, d2i, d3r, d3i;

    for (k2 = 0; k2 < c1; ++k2) {
      // Load data
      a0r = data[c1 * 0 + k2].real();
      a0i = data[c1 * 0 + k2].imag();
      a1r = data[c1 * 1 + k2].real();
      a1i = data[c1 * 1 + k2].imag();
      a2r = data[c1 * 2 + k2].real();
      a2i = data[c1 * 2 + k2].imag();
      a3r = data[c1 * 3 + k2].real();
      a3i = data[c1 * 3 + k2].imag();

      // GOEDECKER COMPUTATION
      c0r = +a2r + a0r;
      c0i = +a2i + a0i;
      c2r = -a2r + a0r;
      c2i = -a2i + a0i;
      c1r = +a3r + a1r;
      c1i = +a3i + a1i;
      c3r = -a3r + a1r;
      c3i = -a3i + a1i;

      d0r = +c1r + c0r;
      d0i = +c1i + c0i;
      d1r = -c1r + c0r;
      d1i = -c1i + c0i;
      d2r = -c3i + c2r;
      d2i = +c3r + c2i;
      d3r = +c3i + c2r;
      d3i = -c3r + c2i;
      // END OF GOEDECKER

      // Store results
      data[c1 * 0 + k2].real(d0r);
      data[c1 * 0 + k2].imag(d0i);
      data[c1 * 1 + k2].real(d1r);
      data[c1 * 1 + k2].imag(d1i);
      data[c1 * 2 + k2].real(d2r);
      data[c1 * 2 + k2].imag(d2i);
      data[c1 * 3 + k2].real(d3r);
      data[c1 * 3 + k2].imag(d3i);
    }
  }

  static void FFT8_0Weights(com_arr &data, int c0) {
    // prepare a constant, sqrt(2) / 2
    constexpr DT sqrt2d2 = 0.7071067811865475244;
    const int c1 = c0 >> 3;

    // calculate integer base-two logarithm of n

    int k2;
    DT a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, a4r, a4i, a5r, a5i, a6r, a6i,
        a7r, a7i, b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i, b4r, b4i, b5r, b5i,
        b6r, b6i, b7r, b7i, c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i, c4r, c4i,
        c5r, c5i, c6r, c6i, c7r, c7i, d0r, d0i, d1r, d1i, d2r, d2i, d3r, d3i,
        d4r, d4i, d5r, d5i, d6r, d6i, d7r, d7i, t5r, t5i, t7r, t7i;

    for (k2 = 0; k2 < c1; ++k2) {
      // Goedecker for 8 (2^3)
      // performed as a sequence of 3 radix-2 passes

      // Load data into registers
      a0r = data[c1 * 0 + k2].real();
      a0i = data[c1 * 0 + k2].imag();
      a1r = data[c1 * 1 + k2].real();
      a1i = data[c1 * 1 + k2].imag();
      a2r = data[c1 * 2 + k2].real();
      a2i = data[c1 * 2 + k2].imag();
      a3r = data[c1 * 3 + k2].real();
      a3i = data[c1 * 3 + k2].imag();
      a4r = data[c1 * 4 + k2].real();
      a4i = data[c1 * 4 + k2].imag();
      a5r = data[c1 * 5 + k2].real();
      a5i = data[c1 * 5 + k2].imag();
      a6r = data[c1 * 6 + k2].real();
      a6i = data[c1 * 6 + k2].imag();
      a7r = data[c1 * 7 + k2].real();
      a7i = data[c1 * 7 + k2].imag();

      // Perform the Goedecker computation
      b0r = a0r + a4r; // w = 1.
      b0i = a0i + a4i;
      b1r = a1r + a5r;
      b1i = a1i + a5i;
      b2r = a2r + a6r;
      b2i = a2i + a6i;
      b3r = a3r + a7r;
      b3i = a3i + a7i;
      b4r = a0r - a4r;
      b4i = a0i - a4i;
      b5r = a1r - a5r;
      b5i = a1i - a5i;
      b6r = a2r - a6r;
      b6i = a2i - a6i;
      b7r = a3r - a7r;
      b7i = a3i - a7i;

      c0r = b0r + b2r; // w = 1.
      c0i = b0i + b2i;
      c1r = b1r + b3r;
      c1i = b1i + b3i;
      c2r = b0r - b2r;
      c2i = b0i - b2i;
      c3r = b1r - b3r;
      c3i = b1i - b3i;
      c4r = b4r - b6i; // w = i.
      c4i = b4i + b6r;
      c5r = b5r - b7i;
      c5i = b5i + b7r;
      c6r = b4r + b6i;
      c6i = b4i - b6r;
      c7r = b5r + b7i;
      c7i = b5i - b7r;

      t5r = c5r - c5i;
      t5i = c5r + c5i;
      t7r = c7r + c7i;
      t7i = c7r - c7i;

      d0r = c0r + c1r; // w = 1.
      d0i = c0i + c1i;
      d1r = c0r - c1r;
      d1i = c0i - c1i;
      d2r = c2r - c3i; // w = i.
      d2i = c2i + c3r;
      d3r = c2r + c3i;
      d3i = c2i - c3r;
      d4r = +t5r * sqrt2d2 + c4r; // w = sqrt(2)/2 * (+1+i).
      d4i = +t5i * sqrt2d2 + c4i;
      d5r = -t5r * sqrt2d2 + c4r;
      d5i = -t5i * sqrt2d2 + c4i;
      d6r = -t7r * sqrt2d2 + c6r; // w = sqrt(2)/2 * (-1+i).
      d6i = +t7i * sqrt2d2 + c6i;
      d7r = +t7r * sqrt2d2 + c6r;
      d7i = -t7i * sqrt2d2 + c6i;

      // Store back results
      data[c1 * 0 + k2].real(d0r);
      data[c1 * 0 + k2].imag(d0i);
      data[c1 * 1 + k2].real(d1r);
      data[c1 * 1 + k2].imag(d1i);
      data[c1 * 2 + k2].real(d2r);
      data[c1 * 2 + k2].imag(d2i);
      data[c1 * 3 + k2].real(d3r);
      data[c1 * 3 + k2].imag(d3i);
      data[c1 * 4 + k2].real(d4r);
      data[c1 * 4 + k2].imag(d4i);
      data[c1 * 5 + k2].real(d5r);
      data[c1 * 5 + k2].imag(d5i);
      data[c1 * 6 + k2].real(d6r);
      data[c1 * 6 + k2].imag(d6i);
      data[c1 * 7 + k2].real(d7r);
      data[c1 * 7 + k2].imag(d7i);
    }
  }

  static void FFT4_1WeightPerIteration(com_arr &data, int u0,
                                       const weight_vec &weights) {
    int k0, k2;
    DT a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, b1r, b1i, b2r, b2i, b3r, b3i,
        c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i, d0r, d0i, d1r, d1i, d2r, d2i,
        d3r, d3i;

    for (k0 = 0; k0 < u0; ++k0) {
      // load values for current weight
      // into registers
      const common_weight weight = weights[k0];
      for (k2 = 0; k2 < 4; ++k2) {
        // load data into registers
        a0r = data[16 * k0 + 4 * 0 + k2].real();
        a0i = data[16 * k0 + 4 * 0 + k2].imag();
        a1r = data[16 * k0 + 4 * 1 + k2].real();
        a1i = data[16 * k0 + 4 * 1 + k2].imag();
        a2r = data[16 * k0 + 4 * 2 + k2].real();
        a2i = data[16 * k0 + 4 * 2 + k2].imag();
        a3r = data[16 * k0 + 4 * 3 + k2].real();
        a3i = data[16 * k0 + 4 * 3 + k2].imag();

        // goedecker for arbitrary weights
        b1r = -a1i * weight.w1i + a1r;
        b1i = +a1r * weight.w1i + a1i;
        b2r = -a2i * weight.w2i + a2r;
        b2i = +a2r * weight.w2i + a2i;
        b3r = -a3i * weight.w3i + a3r;
        b3i = +a3r * weight.w3i + a3i;

        c0r = +b2r * weight.w2r + a0r;
        c0i = +b2i * weight.w2r + a0i;
        c2r = -b2r * weight.w2r + a0r;
        c2i = -b2i * weight.w2r + a0i;
        c1r = +b3r * weight.w3r + b1r;
        c1i = +b3i * weight.w3r + b1i;
        c3r = -b3r * weight.w3r + b1r;
        c3i = -b3i * weight.w3r + b1i;

        d0r = +c1r * weight.w1r + c0r;
        d0i = +c1i * weight.w1r + c0i;
        d1r = -c1r * weight.w1r + c0r;
        d1i = -c1i * weight.w1r + c0i;
        d2r = -c3i * weight.w1r + c2r;
        d2i = +c3r * weight.w1r + c2i;
        d3r = +c3i * weight.w1r + c2r;
        d3i = -c3r * weight.w1r + c2i;

        // store results
        data[16 * k0 + 4 * 0 + k2].real(d0r);
        data[16 * k0 + 4 * 0 + k2].imag(d0i);
        data[16 * k0 + 4 * 1 + k2].real(d1r);
        data[16 * k0 + 4 * 1 + k2].imag(d1i);
        data[16 * k0 + 4 * 2 + k2].real(d2r);
        data[16 * k0 + 4 * 2 + k2].imag(d2i);
        data[16 * k0 + 4 * 3 + k2].real(d3r);
        data[16 * k0 + 4 * 3 + k2].imag(d3i);
      }
    }
  }

  static void FFT4_Final(com_arr &data, int u0, const weight_vec &weights) {
    int k0;
    DT a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, b1r, b1i, b2r, b2i, b3r, b3i,
        c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i, d0r, d0i, d1r, d1i, d2r, d2i,
        d3r, d3i;

    for (k0 = 0; k0 < u0; ++k0) {
      // load values for current weight
      // into registers
      common_weight weight = weights[k0];

      // load data into registers
      a0r = data[4 * k0 + 0].real();
      a0i = data[4 * k0 + 0].imag();
      a1r = data[4 * k0 + 1].real();
      a1i = data[4 * k0 + 1].imag();
      a2r = data[4 * k0 + 2].real();
      a2i = data[4 * k0 + 2].imag();
      a3r = data[4 * k0 + 3].real();
      a3i = data[4 * k0 + 3].imag();

      // perform goedecker computation
      b1r = -a1i * weight.w1i + a1r;
      b1i = +a1r * weight.w1i + a1i;
      b2r = -a2i * weight.w2i + a2r;
      b2i = +a2r * weight.w2i + a2i;
      b3r = -a3i * weight.w3i + a3r;
      b3i = +a3r * weight.w3i + a3i;

      c0r = +b2r * weight.w2r + a0r;
      c0i = +b2i * weight.w2r + a0i;
      c2r = -b2r * weight.w2r + a0r;
      c2i = -b2i * weight.w2r + a0i;
      c1r = +b3r * weight.w3r + b1r;
      c1i = +b3i * weight.w3r + b1i;
      c3r = -b3r * weight.w3r + b1r;
      c3i = -b3i * weight.w3r + b1i;

      d0r = +c1r * weight.w1r + c0r;
      d0i = +c1i * weight.w1r + c0i;
      d1r = -c1r * weight.w1r + c0r;
      d1i = -c1i * weight.w1r + c0i;
      d2r = -c3i * weight.w1r + c2r;
      d2i = +c3r * weight.w1r + c2i;
      d3r = +c3i * weight.w1r + c2r;
      d3i = -c3r * weight.w1r + c2i;

      // store back results
      data[4 * k0 + 0].real(d0r);
      data[4 * k0 + 0].imag(d0i);
      data[4 * k0 + 1].real(d1r);
      data[4 * k0 + 1].imag(d1i);
      data[4 * k0 + 2].real(d2r);
      data[4 * k0 + 2].imag(d2i);
      data[4 * k0 + 3].real(d3r);
      data[4 * k0 + 3].imag(d3i);
    }
  }

  static void generate_common_weights(weight_vec &weights, int length) {
    int k0;

    weights.resize(length / 16);

    for (k0 = 0; k0 < length / 16; ++k0) {
      const double x = tau * r(4 * k0);
      weights[k0].w1r = std::cos(x);
      weights[k0].w1i = std::tan(x);
      weights[k0].w2r = std::cos(x + x);
      weights[k0].w2i = std::tan(x + x);
      weights[k0].w3r = 2. * weights[k0].w2r - 1.;
      weights[k0].w3i = std::tan(3. * x);
    }
  }

  static void execute(com_arr &data, const com_arr &twiddles) {
    // apply the algorithm assuming out of place computation
    // and all m being 1
    constexpr auto log_size = utils::log_n(size);

    weight_vec weights;

    generate_common_weights(weights, size * 16);

    const int first_radix = (size & 1) ? 3 : 2;
    const int P = (log_size - first_radix) / 2 + 1;

    int n = 0;
    if (size & 1) {
      FFT8_0Weights(data, log_size);
    } else {
      FFT4_0Weights(data, log_size);
    }
    n += first_radix;

    int p;

    for (p = 1; p < P - 2; ++p) {
      for (int k0 = 0; k0 < (1 << n); ++k0) {
        FFT4_1WeightPerCall(data, k0, 1 << (log_size - n), weights[k0]);
      }
      n += 2;
    }

    if (p < P - 1) {
      FFT4_1WeightPerIteration(data, 1 << (log_size - 4), weights);
      n += 2;
    }

    p = P - 1;
    FFT4_Final(data, 1 << (log_size - 2), weights);
    n += 2;
  }
};

template <typename DT, typename Size> struct fftw {
  static constexpr int size = Size::value;
  static void execute(std::vector<std::complex<DT>> &data,
                      const std::vector<std::complex<DT>> &twiddles) {

    fftw_complex *in, *out;
    fftw_plan p;

    in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size);
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size);

    for (int i = 0; i < size; ++i) {
      in[i][0] = data[i].real();
      in[i][1] = data[i].imag();
    }
    p = fftw_plan_dft_1d(size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

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
