#include <iostream>
#include <vector>
#include <random>

#include <utils.h>
#include <cpu_fft.h>

int main() {
    std::cout << "Testing the bit reverse function" << std::endl;
    std::vector<int> results;

    for(int i = 0; i < 8; ++i) {
        std::cout << "Bit reversed " << i << " is: " << fft::utils::bit_reverse<3>(i) << std::endl;
    };

    constexpr int N = 1024;
    constexpr int num_bits = fft::utils::highest_bit(N);

    std::cout << "Number of bits: " << num_bits << " for FFT size " << N << std::endl;

    using DT = double;
    std::cout << "Testing twiddle factor generation" << std::endl;
    const auto twiddle_factors = fft::utils::get_roots_of_unity_singleton<N, DT>();

    for(int i = 0; i < N / 2; ++i) {
        std::cout << "Twiddle factor: " << twiddle_factors[i] << " its 16th power: " << std::pow(twiddle_factors[i], N) << std::endl;
    }

    std::cout << "Testing FFT correctness when compared to FFTW" << std::endl;
    std::array<std::complex<DT>, N> data;

    std::random_device rd;
    std::uniform_real_distribution<DT> dist(0.0, 1.0);

    for(int i = 0; i < N; ++i) { data[i] = dist(rd); }

    // reverse order of twiddles
    std::array<std::complex<DT>, N / 2> twiddles_reversed;
    for(int i = 0; i < N / 2; ++i) {
        twiddles_reversed[fft::utils::bit_reverse<num_bits - 2>(i)] = twiddle_factors[i];
    }

    fft::alg::cpu::dit::nr::basic_fft<DT, N>(data, twiddles_reversed);

    return 0;
}
