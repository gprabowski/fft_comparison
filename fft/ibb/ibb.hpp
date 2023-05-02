#pragma once

#include <config.hpp>

#include <complex>
#include <vector>

namespace fft {
namespace alg {
namespace cpu {

struct ibbc_first {
  static void execute(std::vector<std::complex<conf::DT>> &data);
};

struct ibbc_second {
  static void execute(std::vector<std::complex<conf::DT>> &data);
};

struct ibbc_third {
  static void execute(std::vector<std::complex<conf::DT>> &data);
};

} // namespace cpu
} // namespace alg
} // namespace fft
