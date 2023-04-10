#pragma once

#include "utils.h"
#include <vector>
#include <complex>

namespace fft {

    template<typename DT, int Size, bool ReverseInput, bool ReverseOutput>
    struct fft_properties {
        using dt = DT;
        static constexpr auto size = Size;
        static constexpr auto reverse_input = ReverseInput;
        static constexpr auto reverse_output = ReverseOutput;
    };

    template<typename FFTSpec, typename FFTProps>
    struct fft_functor {
        using DT = typename FFTProps::dt;
        static constexpr auto N = FFTProps::size;
  static void execute(std::vector<std::complex<DT>> &data) {

    // arrange input data into correct order
    if(FFTProps::reverse_input) {
        utils::rearrange_data<N>(data);
    }

    // execute fft
    FFTSpec::exec_impl(data);

    // arrange output data into correct order
    if(FFTProps::reverse_output) {
        utils::rearrange_data<N>(data);
    }
  }
    };
};
