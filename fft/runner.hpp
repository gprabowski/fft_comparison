#pragma once

#include <chrono>
#include <complex>
#include <iostream>
#include <vector>

#include "fft_traits.hpp"
#include "utils.h"

namespace fft {

template <typename... Ts> struct type_list;

template <typename FFT, typename DT, bool SaveOutput = false>
void exec_fft(std::vector<std::complex<DT>> &data,
              const std::vector<std::complex<DT>> &twiddles) {

  // if some order is wrong then reverse
  std::vector<std::complex<DT>> local_data, local_twiddles;

  if (fft::fft_trait_order<FFT>::value == fft::fft_order::RN) {
    local_data.resize(FFT::size);
    constexpr auto hb = fft::utils::highest_bit(FFT::size);
    for (int i = 0; i < FFT::size; ++i) {
      local_data[i] = data[fft::utils::bit_reverse<hb - 1>(i)];
    }
  } else {
    local_data = data;
  }

  if (fft::fft_trait_twiddle_order<FFT>::value ==
      fft::twiddle_order::reversed) {
    local_twiddles.resize(FFT::size / 2);
    constexpr auto hb = fft::utils::highest_bit(FFT::size);
    for (int i = 0; i < FFT::size / 2; ++i) {
      local_twiddles[i] = twiddles[fft::utils::bit_reverse<hb - 2>(i)];
    }
  } else {
    local_twiddles = twiddles;
  }

  if constexpr (!SaveOutput) {
    // print name
    std::cout << "Now running: " << fft_trait_name<FFT>::value << std::endl;
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  FFT::execute(data, twiddles);
  auto t2 = std::chrono::high_resolution_clock::now();

  if constexpr (SaveOutput) {
    if (fft::fft_trait_order<FFT>::value == fft::fft_order::RN) {
      constexpr auto hb = fft::utils::highest_bit(FFT::size);
      for (int i = 0; i < FFT::size; ++i) {
        data[i] = local_data[fft::utils::bit_reverse<hb - 1>(i)];
      }
    } else {
      data = local_data;
    }
  }

  if constexpr (!SaveOutput) {
    const auto res_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    std::cout << "That took " << res_ms.count()
              << " milliseconds: " << std::endl;
  }
}

template <typename> struct run_tests;

template <typename FFT> struct run_tests<type_list<FFT>> {
  template <typename DT>
  static void execute(std::vector<std::complex<DT>> &data,
                      const std::vector<std::complex<DT>> &twiddles) {
    exec_fft<FFT, DT>(data, twiddles);
  }
};

template <typename FFT, typename... FFTs>
struct run_tests<type_list<FFT, FFTs...>> {
  template <typename DT>
  static void execute(std::vector<std::complex<DT>> &data,
                      const std::vector<std::complex<DT>> &twiddles) {
    exec_fft<FFT, DT>(data, twiddles);
    run_tests<type_list<FFTs...>>::execute(data, twiddles);
  }
};

template <typename FFTList> struct test_all_ffts_perf {
  template <typename DT>
  static void execute(std::vector<std::complex<DT>> &data,
                      const std::vector<std::complex<DT>> &twiddles) {
    run_tests<FFTList>::execute(data, twiddles);
  }
};

template <typename FFTRef, typename FFT> struct compare_results {
  template <typename DT>
  static bool execute(std::vector<std::complex<DT>> &data,
                      const std::vector<std::complex<DT>> &twiddles) {
    // get first results
    auto data1 = data;
    exec_fft<FFTRef, DT, true>(data1, twiddles);
    // get second results
    auto data2 = data;
    exec_fft<FFT, DT, true>(data2, twiddles);

    // compare
    for (int i = 0; i < data.size(); ++i) {
      const auto ratior = data1[i].real() / data2[i].real();
      const auto distr = std::abs(ratior - DT(1.0));
      if (distr > DT(1e-4)) {
        return false;
      }

      // depending on the defition of the DFT used we can
      // get conjugated value, thus the std::abs here
      const auto ratioi = std::abs(data1[i].imag() / data2[i].imag());
      const auto disti = std::abs(ratioi - DT(1.0));
      if (disti > DT(1e-4)) {
        return false;
      }
    }

    return true;
  }
};

template <typename> struct check_perf;

template <typename FFTRef, typename FFT>
struct check_perf<type_list<FFTRef, FFT>> {
  template <typename DT>
  static bool execute(std::vector<std::complex<DT>> &data,
                      const std::vector<std::complex<DT>> &twiddles) {
    const bool bthis = compare_results<FFTRef, FFT>::execute(data, twiddles);
    return bthis;
  }
};

template <typename FFTRef, typename FFT, typename... FFTs>
struct check_perf<type_list<FFTRef, FFT, FFTs...>> {
  template <typename DT>
  static bool execute(std::vector<std::complex<DT>> &data,
                      const std::vector<std::complex<DT>> &twiddles) {
    const bool bthis = compare_results<FFTRef, FFT>::execute(data, twiddles);
    return bthis &&
           check_perf<type_list<FFTRef, FFTs...>>::execute(data, twiddles);
  }
};

template <typename FFTList> struct test_all_ffts_correctness {
  template <typename DT>
  static bool execute(std::vector<std::complex<DT>> &data,
                      const std::vector<std::complex<DT>> &twiddles) {
    return check_perf<FFTList>::execute(data, twiddles);
  }
};
} // namespace fft
