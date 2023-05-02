#include "edp.hpp"

#include "edp_first.hpp"
#include "edp_second.hpp"
#include "edp_third.hpp"

#include <config.hpp>
#include <utils.h>

namespace fft {
namespace alg {
namespace cpu {

void edpc_first::execute(std::vector<std::complex<conf::DT>> &data) {
  edp_first<conf::DT, utils::int_<conf::size>>::execute(data);
};

void edpc_second::execute(std::vector<std::complex<conf::DT>> &data) {
  edp_second<conf::DT, utils::int_<conf::size>>::execute(data);
};

void edpc_third::execute(std::vector<std::complex<conf::DT>> &data) {
  edp_third<conf::DT, utils::int_<conf::size>>::execute(data);
};

} // namespace cpu
} // namespace alg
} // namespace fft
