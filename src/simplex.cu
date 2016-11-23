/**
 * Implements simplex algorithm.
 */

#include "simplex.cuh"

#include <float.h>
#include <stdio.h>

#include "cudaCheck.cuh"
#include "print.cuh"

__device__ double atomicAdd(double *address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull, assumed;
	if (val == 0.0)
		return __longlong_as_double(old);
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__device__ void minimumIndex(double *value1, int32_t *index1, double value2,
		int32_t index2) {
	if (*value1 > value2) {
		*value1 = value2;
		*index1 = index2;
	}
}

__device__ void warpReduceMinIndex(double *value, int32_t *index) {
	// see https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
	for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
		double shuffleValue = __shfl_down(*value, offset, warpSize);
		double shuffleIndex = __shfl_down(*index, offset, warpSize);
		minimumIndex(value, index, shuffleValue, shuffleIndex);
	};
}

__device__ void blockReduceMinGIndex(double *g, int32_t columns,
		double *blockData, int32_t *blockIndex, double *localData,
		int32_t *localIndex) {
	// see https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
	int32_t warpIndex = threadIdx.x / warpSize;
	int32_t warpLane = threadIdx.x % warpSize;

	*localData = DBL_MAX;
	*localIndex = -1;
	for (int32_t i = threadIdx.x; i < columns; i += blockDim.x) {
		// not 0 because of numerical errors
		if (g[i] < -1.e-4) {
			minimumIndex(localData, localIndex, g[i], i);
		}
	}
	warpReduceMinIndex(localData, localIndex);
	if (warpLane == 0) {
		blockData[warpIndex] = *localData;
		blockIndex[warpIndex] = *localIndex;
	}

	__syncthreads();

	if (threadIdx.x < blockDim.x / warpSize) {
		*localData = blockData[warpLane];
		*localIndex = blockIndex[warpLane];
	} else {
		*localData = DBL_MAX;
		*localIndex = -1;
	}
	if (warpIndex == 0) {
		warpReduceMinIndex(localData, localIndex);
	}
}

__device__ void blockReduceMinSIndex(double *s, double *xB, int32_t rows,
		double *blockData, int32_t *blockIndex, double *localData,
		int32_t *localIndex) {
	// see https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
	int32_t warpIndex = threadIdx.x / warpSize;
	int32_t warpLane = threadIdx.x % warpSize;

	*localData = DBL_MAX;
	*localIndex = -1;
	for (int32_t i = threadIdx.x; i < rows; i += blockDim.x) {
		// not 0 because of numerical errors
		if (s[i] > 1.e-4) {
			minimumIndex(localData, localIndex, xB[i] / s[i], i);
		}
	}
	warpReduceMinIndex(localData, localIndex);
	if (warpLane == 0) {
		blockData[warpIndex] = *localData;
		blockIndex[warpIndex] = *localIndex;
	}

	__syncthreads();

	if (threadIdx.x < blockDim.x / warpSize) {
		*localData = blockData[warpLane];
		*localIndex = blockIndex[warpLane];
	} else {
		*localData = DBL_MAX;
		*localIndex = -1;
	}
	if (warpIndex == 0) {
		warpReduceMinIndex(localData, localIndex);
	}
}

__global__ void copyColumnDouble(double *fromMatrix, double *to, int32_t rows,
		int32_t columns, int32_t column) {
	int32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int32_t i = id; i < rows; i += gridDim.x * blockDim.x) {
		to[i] = fromMatrix[i * columns + column];
	}
}

__global__ void copyColumnDouble2(double *fromANB, double *toANB, int32_t rows,
		int32_t fromColumns, int32_t toColumns, double *fromCNB, double *toCNB,
		int32_t *fromCNBIndex, int32_t *toCNBIndex, int32_t fromColumn, int32_t toColumn) {
	int32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int32_t i = id; i < rows; i += gridDim.x * blockDim.x) {
		toANB[i * toColumns + toColumn] = fromANB[i * fromColumns + fromColumn];
	}
	if (id == 0) {
		toCNB[toColumn] = fromCNB[fromColumn];
		toCNBIndex[toColumn] = fromCNBIndex[fromColumn];
	}
}

__global__ void copyNegativeDouble(double *from, double *to, int32_t elements) {
	int32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int32_t i = id; i < elements; i += gridDim.x * blockDim.x) {
		to[i] = -from[i];
	}
}

__global__ void minG(double *g, int32_t columns, int32_t *column) {
	extern __shared__ double blockData[];
	int32_t *blockIndex = (int32_t *) &blockData[blockDim.x / 32];

	double localData;
	int32_t localIndex;
	blockReduceMinGIndex(g, columns, blockData, blockIndex, &localData,
			&localIndex);

	if (threadIdx.x == 0) {
		*column = localIndex;
	}
}

__global__ void minS(double *s, double *xB, int32_t columns, int32_t *row) {
	extern __shared__ double blockData[];
	int32_t *blockIndex = (int32_t *) &blockData[blockDim.x / 32];

	double localData;
	int32_t localIndex;
	blockReduceMinSIndex(s, xB, columns, blockData, blockIndex, &localData,
			&localIndex);

	if (threadIdx.x == 0) {
		*row = localIndex;
	}
}

__global__ void steepestEdge(double *gTemp, int32_t *ANBColumnIndices,
		double *ANBValues, int32_t columns, int32_t nnz) {
	int32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int32_t i = id; i < nnz; i += gridDim.x * blockDim.x) {
		atomicAdd(&gTemp[ANBColumnIndices[i]],
				ANBValues[ANBColumnIndices[i]]
						* ANBValues[ANBColumnIndices[i]]);
	}
}

__global__ void steepestEdge2(double *g, double *gTemp, int32_t columns) {
	int32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int32_t i = id; i < columns; i += gridDim.x * blockDim.x) {
		g[i] *= rsqrt(gTemp[i]);
	}
}

__global__ void swapColumnDouble(double *AB, double *ANB, int32_t rows,
		int32_t columnsAB, int32_t columnsANB, int32_t columnAB, int32_t columnANB,
		double *cB, int32_t *cBIndex, double *cNB, int32_t *cNBIndex,
		int32_t *xIndex) {
	int32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int32_t i = id; i < rows; i += gridDim.x * blockDim.x) {
		double temp = AB[i * columnsAB + columnAB];
		AB[i * columnsAB + columnAB] = ANB[i * columnsANB + columnANB];
		ANB[i * columnsANB + columnANB] = temp;
	}

	if (id == 0) {
		double tempValue = cB[columnAB];
		cB[columnAB] = cNB[columnANB];
		cNB[columnANB] = tempValue;

		int32_t tempIndex = cBIndex[columnAB];
		cBIndex[columnAB] = cNBIndex[columnANB];
		cNBIndex[columnANB] = tempIndex;

		tempIndex = xIndex[columnAB + columnsANB];
		xIndex[columnAB + columnsANB] = xIndex[columnANB];
		xIndex[columnANB] = tempIndex;
	}
}

__global__ void updateXB(double *xB, double *s, int32_t rows, int32_t row) {
	int32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	//xBi=xBi-si*xBrow/srow xBrow=xBrow/srow
	for (int32_t i = id; i < rows; i += gridDim.x * blockDim.x) {
		if (i == row) {
			xB[i] /= s[i];
		} else {
			xB[i] -= s[i] * xB[row] / s[row];
		}
	}
}

__global__ void zero(double *pointer, int32_t length) {
	int32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	for (int32_t i = id; i < length; i += gridDim.x * blockDim.x) {
		pointer[i] = 0.;
	}
}

void findBFS(LPProblem *lpProblem, cusolverSpHandle_t cusolverHandle,
		cusparseHandle_t cusparseHandle, cusparseMatDescr_t matrixDescriptor) {
	LPProblem *lpTemp = (LPProblem *) malloc(sizeof(LPProblem));
	initializeLPProblem(lpTemp, lpProblem->rows, lpProblem->columns + lpProblem->rows, lpProblem->nnz + lpProblem->rows);

	lpTemp->isBasisAllocated = false;
	lpTemp->rows = lpProblem->rows;
	lpTemp->columns = lpProblem->columns + lpProblem->rows;
	lpTemp->nnz = lpProblem->nnz + lpProblem->rows;
	cudaCheck(cudaMemcpy(lpTemp->ANB, lpProblem->A, lpTemp->rows * (lpTemp->columns - lpTemp->rows)* sizeof(double), cudaMemcpyDeviceToDevice));
	cuSparseCheck(
			cusparseDnnz(cusparseHandle, CUSPARSE_DIRECTION_COLUMN, lpTemp->columns - lpTemp->rows, lpTemp->rows,
					matrixDescriptor, lpTemp->ANB, lpTemp->columns - lpTemp->rows, lpTemp->nnzPerRow,
					&lpTemp->nnzANB));
	cuSparseCheck(
			cusparseDdense2csc(cusparseHandle, lpTemp->columns - lpTemp->rows, lpTemp->rows,
					matrixDescriptor, lpTemp->ANB, lpTemp->columns - lpTemp->rows, lpTemp->nnzPerRow, lpTemp->ANBValues,
					lpTemp->ANBColumnIndices, lpTemp->ANBRowPointer));
	cudaCheck(cudaDeviceSynchronize());
	for (int32_t i = 0; i < lpTemp->rows; i++) {
		for (int32_t j = 0; j < lpTemp->rows; j++) {
			if (i == j) {
				lpTemp->AB[i * lpTemp->rows + j] = 1;
			} else {
				lpTemp->AB[i * lpTemp->rows + j] = 0;
			}
		}
	}
	cuSparseCheck(
			cusparseDnnz(cusparseHandle, CUSPARSE_DIRECTION_COLUMN, lpTemp->rows, lpTemp->rows,
					matrixDescriptor, lpTemp->AB, lpTemp->rows, lpTemp->nnzPerRow, &lpTemp->nnzAB));
	cuSparseCheck(
			cusparseDdense2csc(cusparseHandle, lpTemp->rows, lpTemp->rows, matrixDescriptor,
					lpTemp->AB, lpTemp->rows, lpTemp->nnzPerRow, lpTemp->ABValues, lpTemp->ABColumnIndices,
					lpTemp->ABRowPointer));
	cudaCheck(cudaMemcpy(lpTemp->b, lpProblem->b, lpTemp->rows * sizeof(double), cudaMemcpyDeviceToDevice));
	cudaCheck(cudaMemcpy(lpTemp->cNB, lpProblem->c, (lpTemp->columns - lpTemp->rows) * sizeof(double), cudaMemcpyDeviceToDevice));
	for (int32_t i = 0; i < lpTemp->columns - lpTemp->rows; i++) {
		lpTemp->cNBIndex[i] = i;
	}
	for (int32_t i = 0; i < lpTemp->rows; i++) {
		lpTemp->cBIndex[i] = i + lpTemp->columns - lpTemp->rows;
		lpTemp->cB[i] = -1000;
	}
	for (int32_t i = 0; i < lpTemp->columns; i++) {
		lpTemp->xIndex[i] = i;
	}

	simplex(cusolverHandle, cusparseHandle, matrixDescriptor, lpTemp->rows,
			lpTemp->columns, lpTemp->nnz, &lpTemp->nnzAB, lpTemp->AB,
			lpTemp->ABRowPointer, lpTemp->ABColumnIndices, lpTemp->ABValues,
			lpTemp->ABTRowPointer, lpTemp->ABTColumnIndices, lpTemp->ABTValues,
			&lpTemp->nnzANB, lpTemp->ANB, lpTemp->ANBRowPointer,
			lpTemp->ANBColumnIndices, lpTemp->ANBValues, lpTemp->b, lpTemp->cB,
			lpTemp->cBIndex, lpTemp->cNB, lpTemp->cNBIndex, lpTemp->xB,
			lpTemp->xIndex, lpTemp->s, lpTemp->g, lpTemp->gTemp,
			lpTemp->ANBColumn, lpTemp->nnzPerRow, lpTemp->row, lpTemp->column);

for (int32_t i = 0; i < lpTemp->rows; i++) {
	if (lpTemp->cBIndex[i] < lpProblem->columns) {
		printf("[%d]%g(%d),", lpTemp->cBIndex[i], lpTemp->xB[i], lpProblem->columns);
	}
}
printf("\n");

	lpProblem->isSolution = true;
	for (int32_t i = 0; i < lpTemp->rows; i++) {
		if (lpTemp->cBIndex[i] >= lpProblem->columns) {
			lpProblem->isSolution = false;
			break;
		}
	}

	deleteLPProblem(lpTemp);
}

void simplex(cusolverSpHandle_t cusolverHandle, cusparseHandle_t cusparseHandle,
		cusparseMatDescr_t matrixDescriptor, int32_t rows, int32_t columns,
		int32_t nnz, int32_t *nnzAB, double *AB, int32_t *ABRowPointer,
		int32_t *ABColumnIndices, double *ABValues, int32_t *ABTRowPointer,
		int32_t *ABTColumnIndices, double *ABTValues, int32_t *nnzANB,
		double *ANB, int32_t *ANBRowPointer, int32_t *ANBColumnIndices,
		double *ANBValues, double *b, double *cB, int32_t *cBIndex, double *cNB,
		int32_t *cNBIndex, double *xB, int32_t *xIndex, double *s, double *g,
		double *gTemp, double *ANBColumn, int32_t *nnzPerRow, int32_t *row,
		int32_t *column) {
	double tolerance = 1.e-15;
	int32_t reorder = 0;
	int32_t singularity;
	double one = 1.;

	int32_t blockSize = 384;
	int32_t gridSize = 16;

	int32_t iteration = 0;
	while (true) {
		// AB*xB=b or xB_i=xB_i-s_i*xB_row/s_row xB_row=xB_row/s_row
		if (1) {//iteration % 10 == 0) {
			cuSolverCheck(
					cusolverSpDcsrlsvqr(cusolverHandle, rows, *nnzAB,
							matrixDescriptor, ABValues, ABRowPointer,
							ABColumnIndices, b, tolerance, reorder, xB,
							&singularity));
			if (singularity != -1) {
				printf("singularity at %d\n", singularity);
			}
		} else {
			cudaDeviceSynchronize();
			updateXB<<<gridSize, blockSize>>>(xB, s, rows, *row);
		}

		// y*AB=cB (als ABT*s=cB)
		cuSparseCheck(
				cusparseDcsr2csc(cusparseHandle, rows, rows, *nnzAB, ABValues,
						ABRowPointer, ABColumnIndices, ABTValues,
						ABTColumnIndices, ABTRowPointer,
						CUSPARSE_ACTION_NUMERIC,
						cusparseGetMatIndexBase(matrixDescriptor)));
		cudaDeviceSynchronize();
		cuSolverCheck(
				cusolverSpDcsrlsvqr(cusolverHandle, rows, *nnzAB,
						matrixDescriptor, ABTValues, ABTRowPointer,
						ABTColumnIndices, cB, tolerance, reorder, s,
						&singularity));
		if (singularity != -1) {
			printf("singularity at %d\n", singularity);
			break;
		}

		// g=y*ANB-cNB (als g=ANBT*s-cNB)
		copyNegativeDouble<<<gridSize, blockSize>>>(cNB, g, columns - rows);
		cuSparseCheck(
				cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
						rows, columns - rows, *nnzANB, &one, matrixDescriptor,
						ANBValues, ANBRowPointer, ANBColumnIndices, s, &one,
						g));
		// steepest edge g_i=g_i/||ANB_i||
		zero<<<gridSize, blockSize>>>(gTemp, columns - rows);
		steepestEdge<<<gridSize, blockSize>>>(gTemp, ANBColumnIndices,
				ANBValues, columns - rows, *nnzANB);
		steepestEdge2<<<gridSize, blockSize>>>(g, gTemp, columns - rows);

		// column={i|min{g_i},g_i<0}
		minG<<<1, blockSize, blockSize / 32 * 2 * sizeof(double)>>>(g,
				columns - rows, column);
		cudaCheck(cudaDeviceSynchronize());

		// !column -> optimal
		if (*column == -1) {
			printf("optimal\n");
			break;
		}

		// AB*s=ANB_column
		copyColumnDouble<<<gridSize, blockSize>>>(ANB, ANBColumn, rows,
				columns - rows, *column);
		cuSolverCheck(
				cusolverSpDcsrlsvqr(cusolverHandle, rows, *nnzAB,
						matrixDescriptor, ABValues, ABRowPointer,
						ABColumnIndices, ANBColumn, tolerance, reorder, s,
						&singularity));
		if (singularity != -1) {
			printf("singularity at %d\n", singularity);
			break;
		}

		// row={i|min{xB_i/s_i},s_i>0}
		minS<<<1, blockSize, blockSize / 32 * 2 * sizeof(double)>>>(s, xB, rows,
				row);
		cudaCheck(cudaDeviceSynchronize());

		// !row -> unbounded
		if (*row == -1) {
			printf("unbounded\n");
			break;
		}

		// swap/update variables
		// dense matrix is assumed to be stored in column-major format, need to transpose (implicitly via conversion to CSC format and reinterpreting as CSR)
		swapColumnDouble<<<gridSize, blockSize>>>(AB, ANB, rows, rows, columns - rows, *row, *column, cB, cBIndex, cNB, cNBIndex,xIndex);
		cuSparseCheck(
				cusparseDnnz(cusparseHandle, CUSPARSE_DIRECTION_COLUMN, rows, rows,
						matrixDescriptor, AB, rows, nnzPerRow, nnzAB));
		cuSparseCheck(
				cusparseDdense2csc(cusparseHandle, rows, rows, matrixDescriptor,
						AB, rows, nnzPerRow, ABValues, ABColumnIndices,
						ABRowPointer));
		cuSparseCheck(
				cusparseDnnz(cusparseHandle, CUSPARSE_DIRECTION_COLUMN, columns - rows, rows,
						matrixDescriptor, ANB, columns - rows, nnzPerRow,
						nnzANB));
		cuSparseCheck(
				cusparseDdense2csc(cusparseHandle, columns - rows, rows,
						matrixDescriptor, ANB, columns - rows, nnzPerRow, ANBValues,
						ANBColumnIndices, ANBRowPointer));

		iteration++;
	}

	// solve exact
	cuSolverCheck(
			cusolverSpDcsrlsvqr(cusolverHandle, rows, *nnzAB,
					matrixDescriptor, ABValues, ABRowPointer,
					ABColumnIndices, b, tolerance, reorder, xB,
					&singularity));
	if (singularity != -1) {
		printf("singularity at %d\n", singularity);
	}
}
