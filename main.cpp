#include <iostream>
#include <random>
#include <vector>

#include <cpu_fft.h>
#include <fftw3.h>
#include <utils.h>

int main() {
  constexpr int N = 16;
  constexpr int num_bits = fft::utils::highest_bit(N);

  std::cout << "Number of bits: " << num_bits << " for FFT size " << N
            << std::endl;

  using DT = double;
  const auto twiddle_factors =
      fft::utils::get_roots_of_unity_singleton<N, DT>();

  std::cout << "Testing FFT correctness when compared to FFTW" << std::endl;
  std::array<std::complex<DT>, N> data;

  std::random_device rd;
  std::uniform_real_distribution<DT> dist(0.0, 1.0);

  for (int i = 0; i < N; ++i) {
    data[i] = dist(rd);
  }

  // reverse order of twiddles
  std::array<std::complex<DT>, N / 2> twiddles_reversed;
  for (int i = 0; i < N / 2; ++i) {
    twiddles_reversed[fft::utils::bit_reverse<num_bits - 2>(i)] =
        twiddle_factors[i];
  }

  // FFTW PREPARATION

  fftw_complex *in, *out;
  fftw_plan p;

  in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
  out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);

  for (int i = 0; i < N; ++i) {
    in[i][0] = data[i].real();
    in[i][1] = data[i].imag();
  }

  // OUR ALG RUN
  fft::alg::cpu::dit::nr::basic_fft<DT, N>(data, twiddles_reversed);

  auto data_reversed = data;
  // unscramble data order
  for (int i = 0; i < N; ++i) {
    data_reversed[fft::utils::bit_reverse<num_bits - 1>(i)] = data[i];
  }

  // FFTW RUN
  p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);

  // COMPARE HERE
  for (int i = 0; i < N; ++i) {
    std::cout << "R: " << out[i][0] << " " << data_reversed[i].real()
              << " I: " << out[i][1] << " " << data_reversed[i].imag()
              << std::endl;
  }

  fftw_destroy_plan(p);

  fftw_free(in);
  fftw_free(out);
  return 0;
}
