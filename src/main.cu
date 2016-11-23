#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudaCheck.cuh"
#include "lpProblem.cuh"
#include "print.cuh"
#include "simplex.cuh"

int32_t main(int32_t argc, char *argv[]) {
	cusolverSpHandle_t cusolverHandle;
	cuSolverCheck(cusolverSpCreate(&cusolverHandle));
	cusparseHandle_t cusparseHandle;
	cuSparseCheck(cusparseCreate(&cusparseHandle));

	cusparseMatDescr_t matrixDescriptor;
	cuSparseCheck(cusparseCreateMatDescr(&matrixDescriptor));
	cuSparseCheck(
			cusparseSetMatType(matrixDescriptor, CUSPARSE_MATRIX_TYPE_GENERAL));
	cuSparseCheck(
			cusparseSetMatIndexBase(matrixDescriptor,
					CUSPARSE_INDEX_BASE_ZERO));

	LPProblem *lpProblem = (LPProblem *) malloc(sizeof(LPProblem));
	readMPS(argv[1], lpProblem);
	LPProblem *lpProblemMod = (LPProblem *) malloc(sizeof(LPProblem));
	convertToStandardform(lpProblem, lpProblemMod);
	findBFS(lpProblemMod, cusolverHandle, cusparseHandle, matrixDescriptor);
	LPProblem *lpProblemCopy = (LPProblem *) malloc(sizeof(LPProblem));
	copyLPProblem(lpProblemMod, lpProblemCopy);
	deleteLPProblem(lpProblem);
	deleteLPProblem(lpProblemMod);
	deleteLPProblem(lpProblemCopy);

	cudaCheck(cudaDeviceReset());

	return EXIT_SUCCESS;
}
