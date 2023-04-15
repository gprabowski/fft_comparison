#pragma once

#include <algorithm>
#include <fft.hpp>
#include <immintrin.h>
#include <system_detail.hpp>
#include <utils.h>

#include <iostream>

namespace fft {
namespace alg {
namespace cpu {

template <typename DT, typename Size>
struct edp_third : fft_functor<edp_third<DT, Size>,
                               fft_properties<DT, Size::value, false, false>> {

  static constexpr int size = Size::value;
  static constexpr DT tau = utils::constants::tau;
  using complex_t = std::complex<DT>;
  using com_arr = std::vector<complex_t>;

  constexpr static inline int ilog2(unsigned int n) {
    return 31 - __builtin_clz(n);
  }

  struct common_weight {
    DT w1i, w1r, w2i, w2r, w3i, w3r;
  };

  struct final_weight {
    DT w1r[4], w1i[4], w2r[4], w2i[4], w3r[4], w3i[4];
  };

  using weight_vec = std::vector<common_weight>;
  using final_weight_vec = std::vector<final_weight>;

  // helper array for rw function
  constexpr static const unsigned char b[256] = {
      0,  128, 64, 192, 32, 160, 96,  224, 16, 144, 80, 208, 48, 176, 112, 240,
      8,  136, 72, 200, 40, 168, 104, 232, 24, 152, 88, 216, 56, 184, 120, 248,
      4,  132, 68, 196, 36, 164, 100, 228, 20, 148, 84, 212, 52, 180, 116, 244,
      12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124, 252,
      2,  130, 66, 194, 34, 162, 98,  226, 18, 146, 82, 210, 50, 178, 114, 242,
      10, 138, 74, 202, 42, 170, 106, 234, 26, 154, 90, 218, 58, 186, 122, 250,
      6,  134, 70, 198, 38, 166, 102, 230, 22, 150, 86, 214, 54, 182, 118, 246,
      14, 142, 78, 206, 46, 174, 110, 238, 30, 158, 94, 222, 62, 190, 126, 254,
      1,  129, 65, 193, 33, 161, 97,  225, 17, 145, 81, 209, 49, 177, 113, 241,
      9,  137, 73, 201, 41, 169, 105, 233, 25, 153, 89, 217, 57, 185, 121, 249,
      5,  133, 69, 197, 37, 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245,
      13, 141, 77, 205, 45, 173, 109, 237, 29, 157, 93, 221, 61, 189, 125, 253,
      3,  131, 67, 195, 35, 163, 99,  227, 19, 147, 83, 211, 51, 179, 115, 243,
      11, 139, 75, 203, 43, 171, 107, 235, 27, 155, 91, 219, 59, 187, 123, 251,
      7,  135, 71, 199, 39, 167, 103, 231, 23, 151, 87, 215, 55, 183, 119, 247,
      15, 143, 79, 207, 47, 175, 111, 239, 31, 159, 95, 223, 63, 191, 127, 255};

  constexpr static unsigned int rw(unsigned int k) noexcept {
    unsigned char b0 = b[(k >> (0 * 8)) & 0xff], b1 = b[(k >> (1 * 8)) & 0xff],
                  b2 = b[(k >> (2 * 8)) & 0xff], b3 = b[(k >> (3 * 8)) & 0xff];
    return (b0 << (3 * 8)) | (b1 << (2 * 8)) | (b2 << (1 * 8)) |
           (b3 << (0 * 8));
  }

  constexpr static inline DT r(unsigned int k) noexcept {
    return rw(k) / DT(4294967296.);
  }

  static void FFT4_1WeightPerCall(com_arr &data, int k0, int c0,
                                  const common_weight &weight) {
    const int c1 = c0 >> 2;
    DT a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, b1r, b1i, b2r, b2i, b3r, b3i,
        c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i, d0r, d0i, d1r, d1i, d2r, d2i,
        d3r, d3i;

    for (int k2 = 0; k2 < c1; ++k2) {
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

  static void FFT4_1WeightPerIteration(com_arr &data, int k0_lb, int u0,
                                       const weight_vec &weights) {
    int k0, k2;
    DT a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, b1r, b1i, b2r, b2i, b3r, b3i,
        c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i, d0r, d0i, d1r, d1i, d2r, d2i,
        d3r, d3i;

    for (k0 = k0_lb; k0 < u0; ++k0) {
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

  static void generate_common_weights(weight_vec &weights,
                                      unsigned int length) {
    unsigned int k0;

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

  struct final_index {
    unsigned int read, write;
  };

  static inline final_index construct(unsigned int read, unsigned int write) {
    final_index ret;
    ret.read = read;
    ret.write = write;
    return ret;
  }

  static void generate_final_weights(final_weight_vec &weights, int length,
                                     std::vector<final_index> &indices) {
    const double dlen = static_cast<double>(length);

    weights.resize(length / 16);

    for (int q = 0; q < length / 16; ++q) {
      const int kl = indices[q].read;
      const double r4kl = r(4 * kl);
      for (int khprime = 0; khprime < 4; ++khprime) {
        const double x = tau * (r4kl + khprime / dlen);
        weights[q].w1r[khprime] = std::cos(x);
        weights[q].w1i[khprime] = std::tan(x);
        weights[q].w2r[khprime] = std::cos(x + x);
        weights[q].w2i[khprime] = std::tan(x + x);
        weights[q].w3r[khprime] = 2. * weights[q].w2r[khprime] - 1.;
        weights[q].w3i[khprime] = std::tan(3. * x);
      }
    }
  }

  static void generate_final_indices(std::vector<final_index> &indices,
                                     int length) {
    const int shift = 32 - (ilog2(length) - 4);
    int kl;
    indices.resize(length / 16);

    int idx = 0;
    for (kl = 0; kl < length / 16; ++kl) {
      // rw(kL) reverses kl as a 32-bit number. To get it as
      // the reversal of an N-4 bit number, shift right to
      // remove 32-(N-4) bits.
      const int klprime = rw(kl) >> shift;

      // if klprime < kl then kl in a previous iteration
      // had the value klprime has now, and we do not want to
      // repeat it
      if (kl <= klprime) {
        // If kL == kLprime, add one table entry.
        // If kL != kLprime, add table entries in both orders.
        indices[idx++] = construct(kl, klprime);
        if (kl < klprime)
          indices[idx++] = construct(klprime, kl);
      }
    }
  }

  static void generate_final_indices_cache(std::vector<final_index> &indices,
                                           int len) {
    // prepare to bit reverse a  number of N - 4 bits
    const int shift = 32 - (ilog2(len) - 4);
    int kl;

    indices.resize(len / 16);

    auto p = begin(indices);

    // handle small sizes
    if (len < 64) {
      indices[0] = construct(0, 0);
      if (16 < len) {
        indices[1] = construct(1, 1);
      }
    } else {
      const unsigned int h0 = 0, h1 = (rw(1) >> shift);

      for (kl = 0; kl < len / 32; kl += 2) {
        const int klprime = rw(kl) >> shift;

        if (kl <= klprime) {
          const unsigned int F = kl, R = klprime;
          *(p++) = construct(h0 | F | 0, h0 | R | 0);
          if (kl != klprime) {
            *(p++) = construct(h0 | R | 0, h0 | F | 0);
            *(p++) = construct(h1 | F | 0, h0 | R | 1);
            *(p++) = construct(h0 | R | 1, h1 | F | 0);
            *(p++) = construct(h1 | F | 1, h1 | R | 1);
          }
          *(p++) = construct(h1 | R | 1, h1 | F | 1);
          *(p++) = construct(h0 | F | 1, h1 | R | 0);
          *(p++) = construct(h1 | R | 0, h0 | F | 1);
        }
      }
    }
  }

  // Store 4 integers from SSE vector using offsets from another vector
  static inline void scatter(double *rdi, __m128i idx, __m256d data) {
    const auto tmp = _mm256_extractf128_pd(data, 0);
    const auto tmp2 = _mm256_extractf128_pd(data, 1);
    rdi[(uint32_t)_mm_cvtsi128_si32(idx)] = _mm_cvtsd_f64(tmp);

    rdi[(uint32_t)_mm_extract_epi32(idx, 1)] =
        _mm_cvtsd_f64(_mm_unpacklo_pd(tmp, tmp));

    rdi[(uint32_t)_mm_extract_epi32(idx, 2)] = _mm_cvtsd_f64(tmp2);
    rdi[(uint32_t)_mm_extract_epi32(idx, 3)] =
        _mm_cvtsd_f64(_mm_unpacklo_pd(tmp2, tmp2));
  }

  static void FFT4_Final_AVX2(com_arr &data, int u0,
                              const std::vector<final_index> &indices,
                              const final_weight_vec &weights) {
    __m256d a0r, a1r, a2r, a3r, a0i, a1i, a2i, a3i;
    __m128i offsets = _mm_setr_epi32(2, 2, 2, 2);
    __m128i offsetsw = _mm_setr_epi32(0, 2, 4, 6);
    __m128i offsetsw2 = _mm_setr_epi32(0, 1, 2, 3);

    const auto read_elements = [&](auto kl) {
      const auto index = 4 * kl;

      double *hb00r = reinterpret_cast<double *>(&data[u0 * 0 + index]);
      double *hb01r = reinterpret_cast<double *>(&data[u0 * 2 + index]);
      double *hb10r = reinterpret_cast<double *>(&data[u0 * 1 + index]);
      double *hb11r = reinterpret_cast<double *>(&data[u0 * 3 + index]);
      double *hb00i = reinterpret_cast<double *>(&data[u0 * 0 + index]) + 1;
      double *hb01i = reinterpret_cast<double *>(&data[u0 * 2 + index]) + 1;
      double *hb10i = reinterpret_cast<double *>(&data[u0 * 1 + index]) + 1;
      double *hb11i = reinterpret_cast<double *>(&data[u0 * 3 + index]) + 1;

      const auto y0r = _mm256_i32gather_pd(hb00r, offsetsw, 1);
      const auto y1r = _mm256_i32gather_pd(hb01r, offsetsw, 1);
      const auto y2r = _mm256_i32gather_pd(hb10r, offsetsw, 1);
      const auto y3r = _mm256_i32gather_pd(hb11r, offsetsw, 1);

      const auto y0i = _mm256_i32gather_pd(hb00i, offsetsw, 1);
      const auto y1i = _mm256_i32gather_pd(hb01i, offsetsw, 1);
      const auto y2i = _mm256_i32gather_pd(hb10i, offsetsw, 1);
      const auto y3i = _mm256_i32gather_pd(hb11i, offsetsw, 1);

      const auto z0r = _mm256_unpackhi_pd(y0r, y1r);
      const auto z1r = _mm256_unpacklo_pd(y0r, y1r);
      const auto z2r = _mm256_unpackhi_pd(y2r, y3r);
      const auto z3r = _mm256_unpacklo_pd(y2r, y3r);

      a0r = _mm256_unpackhi_pd(z0r, z2r);
      a1r = _mm256_unpacklo_pd(z0r, z2r);
      a2r = _mm256_unpackhi_pd(z1r, z3r);
      a3r = _mm256_unpacklo_pd(z1r, z3r);

      const auto z0i = _mm256_unpackhi_pd(y0i, y1i);
      const auto z1i = _mm256_unpacklo_pd(y0i, y1i);
      const auto z2i = _mm256_unpackhi_pd(y2i, y3i);
      const auto z3i = _mm256_unpacklo_pd(y2i, y3i);

      a0i = _mm256_unpackhi_pd(z0i, z2i);
      a1i = _mm256_unpacklo_pd(z0i, z2i);
      a2i = _mm256_unpackhi_pd(z1i, z3i);
      a3i = _mm256_unpacklo_pd(z1i, z3i);
    };

    const auto write_reversed_elements = [&](auto klprime) {
      const auto index = 4 * klprime;

      double *hb00r = reinterpret_cast<double *>(&data[u0 * 0 + index]);
      double *hb01r = reinterpret_cast<double *>(&data[u0 * 2 + index]);
      double *hb10r = reinterpret_cast<double *>(&data[u0 * 1 + index]);
      double *hb11r = reinterpret_cast<double *>(&data[u0 * 3 + index]);
      double *hb00i = reinterpret_cast<double *>(&data[u0 * 0 + index]) + 1;
      double *hb01i = reinterpret_cast<double *>(&data[u0 * 2 + index]) + 1;
      double *hb10i = reinterpret_cast<double *>(&data[u0 * 1 + index]) + 1;
      double *hb11i = reinterpret_cast<double *>(&data[u0 * 3 + index]) + 1;

      scatter(hb00r, offsetsw, a0r);
      scatter(hb01r, offsetsw, a1r);
      scatter(hb10r, offsetsw, a2r);
      scatter(hb11r, offsetsw, a3r);
      scatter(hb00i, offsetsw, a0i);
      scatter(hb01i, offsetsw, a1i);
      scatter(hb10i, offsetsw, a2i);
      scatter(hb11i, offsetsw, a3i);
    };

    const auto perform_butterflies = [&](auto weight) {
      const auto w1i = _mm256_i32gather_pd(&weight.w1i[0], offsetsw2, 1);
      const auto w2i = _mm256_i32gather_pd(&weight.w2i[0], offsetsw2, 1);
      const auto w3i = _mm256_i32gather_pd(&weight.w3i[0], offsetsw2, 1);
      const auto w1r = _mm256_i32gather_pd(&weight.w1r[0], offsetsw2, 1);
      const auto w2r = _mm256_i32gather_pd(&weight.w2r[0], offsetsw2, 1);
      const auto w3r = _mm256_i32gather_pd(&weight.w3r[0], offsetsw2, 1);

      const auto b1r = _mm256_fnmadd_pd(a1i, w1i, a1r);
      const auto b1i = _mm256_fmadd_pd(a1r, w1i, a1i);
      const auto b2r = _mm256_fnmadd_pd(a2i, w2i, a2r);
      const auto b2i = _mm256_fmadd_pd(a2r, w2i, a2i);
      const auto b3r = _mm256_fnmadd_pd(a3i, w3i, a3r);
      const auto b3i = _mm256_fmadd_pd(a3r, w3i, a3i);
      const auto c0r = _mm256_fmadd_pd(b2r, w2r, a0r);
      const auto c0i = _mm256_fmadd_pd(b2i, w2r, a0i);
      const auto c2r = _mm256_fnmadd_pd(b2r, w2r, a0r);
      const auto c2i = _mm256_fnmadd_pd(b2i, w2r, a0i);
      const auto c1r = _mm256_fmadd_pd(b3r, w3r, b1r);
      const auto c1i = _mm256_fmadd_pd(b3i, w3r, b1i);
      const auto c3r = _mm256_fnmadd_pd(b3r, w3r, b1r);
      const auto c3i = _mm256_fnmadd_pd(b3i, w3r, b1i);

      a0r = _mm256_fmadd_pd(c1r, w1r, c0r);
      a0i = _mm256_fmadd_pd(c1i, w1r, c0i);
      a1r = _mm256_fnmadd_pd(c1r, w1r, c0i);
      a1i = _mm256_fnmadd_pd(c1i, w1r, c0i);
      a2r = _mm256_fnmadd_pd(c3i, w1r, c2r);
      a2i = _mm256_fmadd_pd(c3r, w1r, c2i);
      a3r = _mm256_fmadd_pd(c3i, w1r, c2r);
      a3i = _mm256_fnmadd_pd(c3r, w1r, c2i);
    };

    int q = 0;

    read_elements(indices[q].read);
    perform_butterflies(weights[q]);
    for (q = 1; q < (u0 >> 2); ++q) {
      read_elements(indices[q].read);
      write_reversed_elements(indices[q - 1].write);
      perform_butterflies(weights[q]);
    }
    write_reversed_elements(indices[q - 1].write);
  }
  static void FFT4_Final(com_arr &data, int u0,
                         const std::vector<final_index> &indices,
                         const final_weight_vec &weights) {
    using float4 = float[4];
    float4 a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, b1r, b1i, b2r, b2i, b3r, b3i,
        c0r, c0i, c1r, c1i, c2r, c2i, c3r, c3i, d0r, d0i, d1r, d1i, d2r, d2i,
        d3r, d3i;
    int q = 0;

    const auto read_elements = [&](auto kl) {
      a0r[0] = data[u0 * 0 + 4 * kl + 0].real();
      a1r[0] = data[u0 * 0 + 4 * kl + 1].real();
      a2r[0] = data[u0 * 0 + 4 * kl + 2].real();
      a3r[0] = data[u0 * 0 + 4 * kl + 3].real();
      a0r[1] = data[u0 * 2 + 4 * kl + 0].real();
      a1r[1] = data[u0 * 2 + 4 * kl + 1].real();
      a2r[1] = data[u0 * 2 + 4 * kl + 2].real();
      a3r[1] = data[u0 * 2 + 4 * kl + 3].real();
      a0r[2] = data[u0 * 1 + 4 * kl + 0].real();
      a1r[2] = data[u0 * 1 + 4 * kl + 1].real();
      a2r[2] = data[u0 * 1 + 4 * kl + 2].real();
      a3r[2] = data[u0 * 1 + 4 * kl + 3].real();
      a0r[3] = data[u0 * 3 + 4 * kl + 0].real();
      a1r[3] = data[u0 * 3 + 4 * kl + 1].real();
      a2r[3] = data[u0 * 3 + 4 * kl + 2].real();
      a3r[3] = data[u0 * 3 + 4 * kl + 3].real();

      a0i[0] = data[u0 * 0 + 4 * kl + 0].imag();
      a1i[0] = data[u0 * 0 + 4 * kl + 1].imag();
      a2i[0] = data[u0 * 0 + 4 * kl + 2].imag();
      a3i[0] = data[u0 * 0 + 4 * kl + 3].imag();
      a0i[1] = data[u0 * 2 + 4 * kl + 0].imag();
      a1i[1] = data[u0 * 2 + 4 * kl + 1].imag();
      a2i[1] = data[u0 * 2 + 4 * kl + 2].imag();
      a3i[1] = data[u0 * 2 + 4 * kl + 3].imag();
      a0i[2] = data[u0 * 1 + 4 * kl + 0].imag();
      a1i[2] = data[u0 * 1 + 4 * kl + 1].imag();
      a2i[2] = data[u0 * 1 + 4 * kl + 2].imag();
      a3i[2] = data[u0 * 1 + 4 * kl + 3].imag();
      a0i[3] = data[u0 * 3 + 4 * kl + 0].imag();
      a1i[3] = data[u0 * 3 + 4 * kl + 1].imag();
      a2i[3] = data[u0 * 3 + 4 * kl + 2].imag();
      a3i[3] = data[u0 * 3 + 4 * kl + 3].imag();
    };

    const auto write_reversed_elements = [&](auto klprime) {
      int khprime;
      for (khprime = 0; khprime < 4; ++khprime) {
        data[u0 * 0 + 4 * klprime + khprime].real(d0r[khprime]);
        data[u0 * 2 + 4 * klprime + khprime].real(d1r[khprime]);
        data[u0 * 1 + 4 * klprime + khprime].real(d2r[khprime]);
        data[u0 * 3 + 4 * klprime + khprime].real(d3r[khprime]);
        data[u0 * 0 + 4 * klprime + khprime].imag(d0i[khprime]);
        data[u0 * 2 + 4 * klprime + khprime].imag(d1i[khprime]);
        data[u0 * 1 + 4 * klprime + khprime].imag(d2i[khprime]);
        data[u0 * 3 + 4 * klprime + khprime].imag(d3i[khprime]);
      }
    };

    const auto perform_butterflies = [&](auto weight) {
      for (int i = 0; i < 4; ++i) {
        b1r[i] = -a1i[i] * weight.w1i[i] + a1r[i];
        b1i[i] = +a1r[i] * weight.w1i[i] + a1i[i];
        b2r[i] = -a2i[i] * weight.w2i[i] + a2r[i];
        b2i[i] = +a2r[i] * weight.w2i[i] + a2i[i];
        b3r[i] = -a3i[i] * weight.w3i[i] + a3r[i];
        b3i[i] = +a3r[i] * weight.w3i[i] + a3i[i];
        c0r[i] = +b2r[i] * weight.w2r[i] + a0r[i];
        c0i[i] = +b2i[i] * weight.w2r[i] + a0i[i];
        c2r[i] = -b2r[i] * weight.w2r[i] + a0r[i];
        c2i[i] = -b2i[i] * weight.w2r[i] + a0i[i];
        c1r[i] = +b3r[i] * weight.w3r[i] + b1r[i];
        c1i[i] = +b3i[i] * weight.w3r[i] + b1i[i];
        c3r[i] = -b3r[i] * weight.w3r[i] + b1r[i];
        c3i[i] = -b3i[i] * weight.w3r[i] + b1i[i];
        d0r[i] = +c1r[i] * weight.w1r[i] + c0r[i];
        d0i[i] = +c1i[i] * weight.w1r[i] + c0i[i];
        d1r[i] = -c1r[i] * weight.w1r[i] + c0r[i];
        d1i[i] = -c1i[i] * weight.w1r[i] + c0i[i];
        d2r[i] = -c3i[i] * weight.w1r[i] + c2r[i];
        d2i[i] = +c3r[i] * weight.w1r[i] + c2i[i];
        d3r[i] = +c3i[i] * weight.w1r[i] + c2r[i];
        d3i[i] = -c3r[i] * weight.w1r[i] + c2i[i];
      }
    };

    read_elements(indices[q].read);
    perform_butterflies(weights[q]);
    for (q = 1; q < (u0 >> 2); ++q) {
      read_elements(indices[q].read);
      write_reversed_elements(indices[q - 1].write);
      perform_butterflies(weights[q]);
    }
    write_reversed_elements(indices[q - 1].write);
  }

  static void gather_buffer(com_arr &dst, com_arr &src, int u1, int c1, int k2,
                            int cluster) {
    int k1, c;

    for (k1 = 0; k1 < u1; ++k1) {
      for (c = 0; c < cluster; ++c) {
        dst[cluster * k1 + c] = src[c1 * k1 + k2 + c];
      }
    }
  }

  static void scatter_buffer(com_arr &dst, com_arr &src, int u1, int c1, int k2,
                             int cluster) {
    int k1, c;

    for (k1 = 0; k1 < u1; ++k1) {
      for (c = 0; c < cluster; ++c) {
        dst[c1 * k1 + k2 + c] = src[cluster * k1 + c];
      }
    }
  }

  static void FFT_first_stage(int m, com_arr &data, int c0,
                              const std::vector<common_weight> &weights) {
    const int c1 = c0 >> m;
    const int u1 = 1 << m;

    constexpr auto max_c =
        ((system_detail::l1_cache_size * 1024) / (2 * sizeof(DT))) / 2;
    constexpr unsigned int b = std::min(1 << ilog2(max_c), size);

    std::vector<std::complex<DT>> buffer(b);

    const int cluster = b >> m;

    int n, n_lb, k, k0, k2;

    for (k2 = 0; k2 < c1; k2 += cluster) {
      gather_buffer(buffer, data, u1, c1, k2, cluster);

      if (m & 1) {
        FFT8_0Weights(buffer, b);
      } else {
        FFT4_0Weights(buffer, b);
      }

      n_lb = m & 1 ? 3 : 2;

      for (n = n_lb; n < m; n += 2) {
        FFT4_0Weights(buffer, b >> n);
      }

      for (k0 = 1; n_lb < m; n_lb += 2) {
        for (; k0 < 1 << n_lb; ++k0) {
          for (n = n_lb; n < m; n += 2) {
            FFT4_1WeightPerCall(buffer, k0, b >> n, weights[k0]);
          }
        }
      }

      scatter_buffer(data, buffer, u1, c1, k2, cluster);
    }
  }

  static void FFT_penultimate_stage(com_arr &data, int n_stage, int N,
                                    const std::vector<common_weight> &weights) {
    constexpr auto log_size = utils::log_n(size);
    int n, n_lb, k, k0;
    n_lb = n_stage;

    constexpr auto max_c =
        ((system_detail::l1_cache_size * 1024) / (2 * sizeof(DT))) / 2;
    constexpr int g = std::min(size, 1 << ilog2(max_c));

    for (n = n_lb; n < log_size - 4; n += 2) {
      FFT4_0Weights(data, 1 << (log_size - n));
    }

    for (k0 = 1; n_lb < log_size - 4; n_lb += 2) {
      for (; k0 < g >> (N - n_lb); ++k0) {
        for (n = n_lb; n < log_size - 4; n += 2) {
          FFT4_1WeightPerCall(data, k0, 1 << (log_size - n), weights[k0]);
        }
      }
    }

    FFT4_1WeightPerIteration(data, 0, g >> 4, weights);

    for (k = g; k < 1 << log_size; k += g) {
      for (n = n_stage; n < log_size - 4; n += 2) {
        for (k0 = k >> (log_size - n); k0 < (k + g) >> (log_size - n); ++k0) {
          FFT4_1WeightPerCall(data, k0, 1 << (log_size - n), weights[k0]);
        }
      }
      FFT4_1WeightPerIteration(data, k >> 4, (k + g) >> 4, weights);
    }
  }

  static void FFT_final_stage(com_arr &data, int u0,
                              const std::vector<final_index> &final_indices,
                              const std::vector<final_weight> &final_weights) {
    FFT4_Final(data, u0, final_indices, final_weights);
  }

  static void exec_impl(com_arr &data) {
    // apply the algorithm assuming out of place computation
    // and all m being 1
    constexpr auto log_size = utils::log_n(size);

    weight_vec weights;
    final_weight_vec final_weights;
    std::vector<final_index> final_indices;

    generate_common_weights(weights, size);
    generate_final_indices_cache(final_indices, size);
    generate_final_weights(final_weights, size, final_indices);

    // here we have to make a split of our data into 3 stages
    // we know that the last stage will be a radix-4 stage (m_2 = 2)
    //
    // L1 size is 192KiB = 2 * 6144 complex<double> numbers
    // but we limit ourselfes to half, which is 6144 and
    // round down to the closest power of 2, which is 4096
    //
    constexpr auto max_c =
        ((system_detail::l1_cache_size * 1024) / (2 * sizeof(DT))) / 2;
    constexpr int b = std::min(size, 1 << ilog2(max_c));

    // first we choose m_0
    // if log_n & 1 then m_0 is at least 3
    // else m_0 >= 2
    //
    // m_0 <= log_2(b)
    // m_0 <= 12
    //
    constexpr int m_2 = 2;
    // m_0 = min(log_n - 2, 12)
    constexpr int m_0 = std::min(log_size - m_2, (log_size & 1) ? 7u : 8u);
    //
    // m_1 = max(0, log_n - m_0 - m_2)
    constexpr int m_1 = std::max(0u, log_size - m_0 - m_2);

    FFT_first_stage(m_0, data, 1 << log_size, weights);

    if (m_1 > 0) {
      FFT_penultimate_stage(data, m_0, log_size, weights);
    }

    FFT_final_stage(data, 1 << (log_size - 2), final_indices, final_weights);
  }
};
} // namespace cpu
} // namespace alg
} // namespace fft
