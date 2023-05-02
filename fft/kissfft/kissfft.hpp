#pragma once

#include <complex>
#include <vector>

#include <config.hpp>

namespace fft {
namespace alg {
namespace cpu {
struct kissfftc {
  static void execute(std::vector<std::complex<conf::DT>> &data);
};
} // namespace cpu
} // namespace alg
} // namespace fft
