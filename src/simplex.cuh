/**
 * Defines simplex algorithm.
 */

#ifndef SIMPLEX_CUH_
#define SIMPLEX_CUH_

#include <cusolverSp.h>
#include <cusparse_v2.h>
#include <stdint.h>

#include "lpProblem.cuh"

__device__ double atomicAdd(double *address, double val);

__device__ void minimumIndex(double *value1, int32_t *index1, double value2,
		int32_t index2);

__device__ void warpReduceMinIndex(double *value, int32_t *index);

__device__ void blockReduceMinGIndex(double *g, int32_t columns,
		double *blockData, int32_t *blockIndex, double *localData,
		int32_t *localIndex);

__device__ void blockReduceMinSIndex(double *s, double *xB, int32_t rows,
		double *blockData, int32_t *blockIndex, double *localData,
		int32_t *localIndex);

__global__ void copyColumnDouble(double *fromMatrix, double *to, int32_t rows,
		int32_t columns, int32_t column);

__global__ void copyColumnDouble2(double *fromANB, double *toANB, int32_t rows,
		int32_t fromColumns, int32_t toColumns, double *fromCNB, double *toCNB,
		double *fromCNBIndex, double *toCNBIndex, int32_t fromColumn, int32_t toColumn);

__global__ void copyNegativeDouble(double *from, double *to, int32_t elements);

__global__ void minG(double *g, int32_t columns, int32_t *column);

__global__ void minS(double *s, double *xB, int32_t columns, int32_t *row);

__global__ void steepestEdge(double *gTemp, int32_t *ANBColumnIndices,
		double *ANBValues, int32_t columns, int32_t nnz);

__global__ void steepestEdge2(double *g, double *gTemp, int32_t columns);

__global__ void swapColumnDouble(double *AB, double *ANB, int32_t rows,
		int32_t columnsAB, int32_t columnsANB, int32_t columnAB, int32_t columnANB,
		double *cB, int32_t *cBIndex, double *cNB, int32_t *cNBIndex,
		int32_t *xIndex);

__global__ void updateXB(double *xB, double *s, int32_t rows, int32_t row);

__global__ void zero(double *pointer, int32_t length);

void findBFS(LPProblem *lpProblem, cusolverSpHandle_t cusolverHandle,
		cusparseHandle_t cusparseHandle, cusparseMatDescr_t matrixDescriptor);

void simplex(cusolverSpHandle_t cusolverHandle, cusparseHandle_t cusparseHandle,
		cusparseMatDescr_t matrixDescriptor, int32_t rows, int32_t columns,
		int32_t nnz, int32_t *nnzAB, double *AB, int32_t *ABRowPointer,
		int32_t *ABColumnIndices, double *ABValues, int32_t *ABTRowPointer,
		int32_t *ABTColumnIndices, double *ABTValues, int32_t *nnzANB,
		double *ANB, int32_t *ANBRowPointer, int32_t *ANBColumnIndices,
		double *ANBValues, double *b, double *cB, int32_t *cBIndex, double *cNB,
		int32_t *cNBIndex, double *xB, int32_t *xIndex, double *s, double *g,
		double *gTemp, double *ANBColumn, int32_t *nnzPerRow, int32_t *row,
		int32_t *column);

#endif /* SIMPLEX_CUH_ */
