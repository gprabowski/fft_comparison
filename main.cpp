#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <cpu_fft.h>
#include <fftw3.h>
#include <utils.h>

int main() {
  using DT = double;

  const unsigned int N = (1 << 20);

  const unsigned int num_bits = fft::utils::highest_bit(N);

  const auto twiddle_factors =
      fft::utils::get_roots_of_unity_singleton<N, DT>();

  std::cout << "Testing FFT correctness when compared to FFTW" << std::endl;
  std::vector<std::complex<DT>> data(N);

  std::random_device rd;
  std::uniform_real_distribution<DT> dist(0.0, 1.0);

  for (int i = 0; i < N; ++i) {
    data[i] = dist(rd);
  }

  // reverse order of twiddles
  std::vector<std::complex<DT>> twiddles_reversed(N / 2);
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
  auto t1 = std::chrono::high_resolution_clock::now();
  fft::alg::cpu::ibb_first<DT, N>::execute(data, twiddles_reversed);
  auto t2 = std::chrono::high_resolution_clock::now();

  const auto basic_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

  auto data_reversed = data;
  // unscramble data order
  for (int i = 0; i < N; ++i) {
    data_reversed[fft::utils::bit_reverse<num_bits - 1>(i)] = data[i];
  }

  // FFTW RUN

  t1 = std::chrono::high_resolution_clock::now();
  p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_execute(p);
  t2 = std::chrono::high_resolution_clock::now();

  const auto fftw_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

  fftw_destroy_plan(p);

  fftw_free(in);
  fftw_free(out);

  std::cout << "Basic milliseconds: " << basic_ms.count()
            << " FFTW ms: " << fftw_ms.count() << std::endl;
  return 0;
}
