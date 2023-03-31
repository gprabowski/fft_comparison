//
// Created by gprab on 3/26/2023.
//

#ifndef FFT_COMPARISON_UTILS_H
#define FFT_COMPARISON_UTILS_H

#include <array>
#include <cmath>
#include <complex>

namespace fft {
    namespace utils {
        namespace constants {
            constexpr double pi = 3.14159265358979323846264338327950288419716939937510;
            constexpr double tau = 2.0 * pi;
        }

        constexpr bool is_power_of_2(int N) {
            return (N & (N - 1)) == 0;
        }

        constexpr int highest_bit(unsigned int N) {
            int res = 0;
#pragma unroll 32
            for(int i = 0; i < 32; ++i) {
                if(N == 0) return res;
                N >>= 1;
                ++res;
            }
	    return res;
	}

        template<int N>
        constexpr int bit_reverse(int num) {
            int ret = 0;
#pragma unroll N
            for (int i = 0; i < N; ++i) {
                ret += ((num & (1 << i)) > 0) * (1 << (N - 1 - i));
            }

            return ret;
        }

        template<int N, typename DT = float>
        std::array<std::complex<DT>, N / 2> get_roots_of_unity() {
            std::array<std::complex<DT>, N / 2> ret;
            constexpr DT theta = constants::tau / N;
            for(int i = 0; i < N / 2; ++i) {
                ret[i] = {std::cos(i * theta), std::sin(i * theta)};
            }
            return ret;
        }

        template<int N, typename DT = float>
        std::array<std::complex<DT>, N / 2> get_roots_of_unity_singleton() {
            std::array<std::complex<DT>, N / 2> ret;
            constexpr DT theta = constants::tau / N;
            const DT s = std::sin(theta); const DT c = 1.0 - 2 * sin(theta / 2.0) * sin(theta / 2.0);
            ret[0].real(1.0);
            ret[0].imag(0.0);

#pragma unroll N / 8
            for(int k = 0; k < N / 8 - 1; ++k) {
                ret[k + 1].real(c * ret[k].real() - s * ret[k].imag());
                ret[k + 1].imag(s * ret[k].real() + c * ret[k].imag());
            }

            constexpr int l1 = N / 8;
            ret[l1].real(sqrt(2) / 2.f);
            ret[l1].imag(ret[l1].real());

#pragma unroll (N / 8 - 1)
            for(int k = 1; k < N / 8; ++k) {
                ret[l1 + k].real(ret[l1 - k].imag());
                ret[l1 + k].imag(ret[l1 - k].real());
            }

            constexpr int l2 = N / 4;
            ret[l2].real(0);
            ret[l2].imag(1);

#pragma unroll (N / 4 - 1)
            for(int k = 1; k < N / 4; ++k) {
                ret[l2 + k].real(-ret[l2 - k].real());
                ret[l2 + k].imag(ret[l2 - k].imag());
            }

            return ret;
        }
    }
}

#endif //FFT_COMPARISON_UTILS_H