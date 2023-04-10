#include "runner.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <cpu_fft.h>
#include <utils.h>

int main() {
  using fft::utils::int_;
  namespace fac = fft::alg::cpu;
  using DT = double;
  const unsigned int N = (1 << 14);
  std::random_device rd;
  std::uniform_real_distribution<DT> dist(0.0, 1.0);

  std::vector<std::complex<DT>> data(N);

  std::transform(begin(data), end(data), begin(data),
                 [&](auto) { return std::complex<DT>(dist(rd), dist(rd)); });

  using fft_list =
      fft::type_list<fac::fftw<DT, int_<N>>, fac::ibb_first<DT, int_<N>>,
                     fac::ibb_second<DT, int_<N>>, fac::ibb_third<DT, int_<N>>,
                     fac::edp_first<DT, int_<N>>, fac::edp_second<DT, int_<N>>>;

  using final_list =
      fft::type_list<fac::fftw<DT, int_<N>>, fac::edp_second<DT, int_<N>>>;

  if (!fft::test_all_ffts_correctness<final_list>::execute(data)) {
    std::cerr << "FAILED CORRECTNESS TEST" << std::endl;
    return -1;
  }

  fft::test_all_ffts_perf<final_list>::execute(data);

  return 0;
}
