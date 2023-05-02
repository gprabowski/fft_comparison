#include "edp/edp_third.hpp"
#include "runner.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <config.hpp>

#include <cpu_fft.h>
#include <utils.h>

constexpr bool test_correctness = true;

int main() {
  using fft::utils::int_;
  namespace fac = fft::alg::cpu;
  using DT = conf::DT;
  std::random_device rd;
  std::uniform_real_distribution<DT> dist(0.0, 1.0);

  std::vector<std::complex<DT>> data(conf::size);

  std::transform(begin(data), end(data), begin(data),
                 [&](auto) { return std::complex<DT>(dist(rd), dist(rd)); });

  using final_list = fft::type_list<fac::fftwc, fac::ibbc_first>;

  fft::test_all_ffts_correctness<final_list>::execute(data);
  fft::test_all_ffts_perf<final_list>::execute(data);

  return 0;
}
