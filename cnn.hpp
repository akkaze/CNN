#ifndef CNN_HPP
#define CNN_HPP

#include <cblas.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

#define DOUBLE_PRECISION

#ifdef DOUBLE_PRECISION
#define real_t double
#else
#define real_t float
#endif

#ifdef DOUBLE_PRECISION
#define cblas_gemm cblas_dgemm
#define cblas_axpy cblas_daxpy
#define cblas_axpby cblas_daxpby
#else
#define cblas_gemm cblas_sgemm
#define cblas_axpy cblas_saxpy
#define cblas_axpby cblas_saxpby
#endif

#include "batch.h"
#include "blas.hpp"
#include "im2col.hpp"
#include "numeric.hpp"
#include "layers.hpp"
#include "solve.hpp"
#include "net.hpp"
#endif
