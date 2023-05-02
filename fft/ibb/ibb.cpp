#include "ibb.hpp"

#include "ibb_first.hpp"
#include "ibb_second.hpp"
#include "ibb_third.hpp"

#include <config.hpp>
#include <utils.h>

namespace fft {
namespace alg {
namespace cpu {

void ibbc_first::execute(std::vector<std::complex<conf::DT>> &data) {
  ibb_first<conf::DT, utils::int_<conf::size>>::execute(data);
};

void ibbc_second::execute(std::vector<std::complex<conf::DT>> &data) {
  ibb_second<conf::DT, utils::int_<conf::size>>::execute(data);
};

void ibbc_third::execute(std::vector<std::complex<conf::DT>> &data) {
  ibb_third<conf::DT, utils::int_<conf::size>>::execute(data);
};

} // namespace cpu
} // namespace alg
} // namespace fft
