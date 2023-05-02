#pragma once

#include <chrono>
#include <complex>
#include <iostream>
#include <type_traits>
#include <vector>

#include "cpu_fft.h"
#include "fft_traits.hpp"
#include "utils.h"

namespace fft {

template <typename... Ts> struct type_list;

template <typename FFT, typename DT, bool SaveOutput = false>
void exec_fft(std::vector<std::complex<DT>> &data) {
  if constexpr (!SaveOutput) {
    // print name
    // flush cache
    const size_t bigger_than_cache = 100 * 1024 * 1024;
    long *p = new long[bigger_than_cache];

    for (int i = 0; i < bigger_than_cache; ++i) {
      p[i] = rand();
    }

    // compute

    std::cout << "Now running: " << fft_trait_name<FFT>::value << std::endl;
    std::vector<std::complex<DT>> local_data = data;
    auto t1 = std::chrono::high_resolution_clock::now();
    FFT::execute(local_data);
    auto t2 = std::chrono::high_resolution_clock::now();
    const auto res_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

    std::cout << "That took " << res_ms.count()
              << " microseconds: " << std::endl;
    delete[] p;
  } else {
    FFT::execute(data);
  }
}

template <typename> struct run_tests;

template <typename FFT> struct run_tests<type_list<FFT>> {
  template <typename DT>
  static void execute(std::vector<std::complex<DT>> &data) {
    exec_fft<FFT, DT>(data);
  }
};

template <typename FFT, typename... FFTs>
struct run_tests<type_list<FFT, FFTs...>> {
  template <typename DT>
  static void execute(std::vector<std::complex<DT>> &data) {
    exec_fft<FFT, DT>(data);
    run_tests<type_list<FFTs...>>::execute(data);
  }
};

template <typename FFTList> struct test_all_ffts_perf {
  template <typename DT>
  static void execute(std::vector<std::complex<DT>> &data) {
    run_tests<FFTList>::execute(data);
  }
};

template <typename FFTRef, typename FFT> struct compare_results {
  template <typename DT>
  static void execute(const std::vector<std::complex<DT>> &data) {

    std::cout << "TESTING CORRECTNESS " << fft_trait_name<FFTRef>::value
              << " and " << fft_trait_name<FFT>::value << std::endl;

    // get first results
    std::vector<std::complex<DT>> data1 = data;
    exec_fft<FFTRef, DT, true>(data1);

    // get second results
    std::vector<std::complex<DT>> data2 = data;
    exec_fft<FFT, DT, true>(data2);

    // compare
    DT mse{0.0f};
    for (unsigned int i = 0; i < data.size(); ++i) {
      const auto current_se =
          std::norm(data1[i] - ((i == 0) ? data2[i] : data2[data2.size() - i]));
      if constexpr (false) {
        if (i == 0) {
          std::cout << "R: " << data1[i].real() << " " << data2[i].real()
                    << std::endl;
        } else {
          std::cout << "R: " << data1[i].real() << " "
                    << data2[data2.size() - i].real() << std::endl;
        }
        std::cout << "MSE: " << mse << std::endl;
      }
      mse += current_se;
    }
    mse = mse / DT(data.size());

    std::cout << "MSE of " << fft_trait_name<FFTRef>::value << " and "
              << fft_trait_name<FFT>::value << " : " << mse << std::endl;
  }
};

template <typename> struct check_correctness;

template <typename FFTRef, typename FFT>
struct check_correctness<type_list<FFTRef, FFT>> {
  template <typename DT>
  static void execute(std::vector<std::complex<DT>> &data) {
    compare_results<FFTRef, FFT>::execute(data);
  }
};

template <typename FFTRef, typename FFT, typename... FFTs>
struct check_correctness<type_list<FFTRef, FFT, FFTs...>> {
  template <typename DT>
  static void execute(std::vector<std::complex<DT>> &data) {
    compare_results<FFTRef, FFT>::execute(data);
    check_correctness<type_list<FFTRef, FFTs...>>::execute(data);
  }
};

template <typename FFTList> struct test_all_ffts_correctness {
  template <typename DT>
  static void execute(std::vector<std::complex<DT>> &data) {
    check_correctness<FFTList>::execute(data);
  }
};
} // namespace fft
