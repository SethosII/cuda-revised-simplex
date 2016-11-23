/**
 * Implements print.
 */

#include <stdio.h>

void printIntArray(int32_t *array, int32_t length) {
	for (int32_t i = 0; i < length; i++) {
		printf("%d,", array[i]);
	}
	printf("\n");
}

void printDoubleArray(double *array, int32_t length) {
	for (int32_t i = 0; i < length; i++) {
		printf("%g,", array[i]);
	}
	printf("\n");
}

void printCSRMatrix(int32_t *RowPointer, int32_t *ColumnIndices, double *Values,
		int32_t rows) {
	for (int32_t i = 0; i <= rows; i++) {
		printf("%d,", RowPointer[i]);
	}
	printf("\n");
	for (int32_t i = 0; i < RowPointer[rows]; i++) {
		printf("%d,", ColumnIndices[i]);
	}
	printf("\n");
	for (int32_t i = 0; i < RowPointer[rows]; i++) {
		printf("%g,", Values[i]);
	}
	printf("\n");
}

void printMatrix(double *matrix, int32_t rows, int32_t columns) {
	for (int32_t row = 0; row < rows; row++) {
		for (int32_t column = 0; column < columns; column++) {
			printf("%g,", matrix[row * columns + column]);
		}
		printf("\n");
	}
}
