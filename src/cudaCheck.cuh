/**
 * Defines CUDA runtime error checking
 */

#ifndef CUDACHECK_CUH_
#define CUDACHECK_CUH_

#include <cusolver_common.h>
#include <cusparse_v2.h>
#include <stdio.h>
#include <stdlib.h>

//#define NDEBUG // include to remove asserts and cudaCheck
#define cudaCheck(call) __cudaCheck(call, __FILE__, __LINE__)

inline void __cudaCheck(cudaError err, const char* file, int line) {
#ifndef NDEBUG
	if (err != cudaSuccess) {
		fprintf(stderr, "%s(%d): CUDA error: %s\n", file, line,
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#endif
}

#define cuSparseCheck(call) __cuSparseCheck(call, __FILE__, __LINE__)

inline void __cuSparseCheck(cusparseStatus_t err, const char* file, int line) {
#ifndef NDEBUG
	if (err != CUSPARSE_STATUS_SUCCESS) {
		fprintf(stderr, "%s(%d): CUSPARSE error: %s\n", file, line, err);
		exit(EXIT_FAILURE);
	}
#endif
}

#define cuSolverCheck(call) __cuSolverCheck(call, __FILE__, __LINE__)

inline void __cuSolverCheck(cusolverStatus_t err, const char* file, int line) {
#ifndef NDEBUG
	if (err != CUSOLVER_STATUS_SUCCESS) {
		fprintf(stderr, "%s(%d): CUSOLVER error: %s\n", file, line, err);
		exit(EXIT_FAILURE);
	}
#endif
}

#endif /* CUDACHECK_CUH_ */
