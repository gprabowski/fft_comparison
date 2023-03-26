//
// Created by gprab on 3/26/2023.
//

#include <array>
#include <vector>
#include <complex>

#ifndef FFT_COMPARISON_CPU_FFT_H
#define FFT_COMPARISON_CPU_FFT_H

namespace fft {
    namespace alg {
        namespace cpu {
            namespace dit {
                namespace nr {
                    // assumes bit reversed order of twiddles
                    template <typename DT, int N>
                    void basic_fft(std::array<std::complex<DT>, N>& data, const std::array<std::complex<DT>, N / 2>& twiddles) {
                        int pairs_in_groups = N / 2;
                        int num_groups = 1;
                        int distance  = N / 2;
                        while(num_groups < N) {
                            for(int k = 0; k < num_groups; ++k) {
                                const int j_first = 2 * k * pairs_in_groups;
                                const int j_last = j_first + pairs_in_groups - 1;
                                const int j_twiddle = k;
                                const auto twiddle = twiddles[j_twiddle];
                                for(int j = j_first; j <= j_last; ++j) {
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
                }
            }
        }
    }
}

#endif //FFT_COMPARISON_CPU_FFT_H
