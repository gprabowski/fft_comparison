#pragma once

#include <config.hpp>

#include <complex>
#include <vector>

namespace fft {
namespace alg {
namespace cpu {

struct fftwc {
  static void execute(std::vector<std::complex<conf::DT>> &data);
};

} // namespace cpu
} // namespace alg
} // namespace fft
