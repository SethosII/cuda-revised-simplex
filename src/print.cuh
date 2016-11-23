/**
 * Defines print.
 */

#ifndef PRINT_CUH_
#define PRINT_CUH_

void printIntArray(int32_t *array, int32_t length);

void printDoubleArray(double *array, int32_t length);

void printCSRMatrix(int32_t *RowPointer, int32_t *ColumnIndices, double *Values,
		int32_t rows);

void printMatrix(double *matrix, int32_t rows, int32_t columns);

#endif /* PRINT_CUH_ */
