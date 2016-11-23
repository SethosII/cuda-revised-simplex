/**
 * Implements LP problem
 */

#include <glpk.h>
#include <stdio.h>

#include "cudaCheck.cuh"
#include "lpProblem.cuh"
#include "print.cuh"

void readMPS(char *mpsFile, LPProblem *lpProblem) {
	glp_prob *lp = glp_create_prob();
	glp_read_mps(lp, GLP_MPS_FILE, NULL, mpsFile);
	lpProblem->rows = glp_get_num_rows(lp);
	lpProblem->columns = glp_get_num_cols(lp);
	lpProblem->nnz = glp_get_num_nz(lp);
	lpProblem->isBasisAllocated = false;


	cudaCheck(
			cudaMallocManaged(&lpProblem->A, lpProblem->rows * lpProblem->columns * sizeof(double)));
	cudaCheck(cudaMallocManaged(&lpProblem->b, lpProblem->rows * sizeof(double)));
	cudaCheck(cudaMallocManaged(&lpProblem->c, lpProblem->columns * sizeof(double)));
	cudaCheck(
			cudaMallocManaged(&lpProblem->lowerBound,
					lpProblem->columns * sizeof(double)));
	cudaCheck(
			cudaMallocManaged(&lpProblem->upperBound,
					lpProblem->columns * sizeof(double)));

	for (int32_t i = 0; i < lpProblem->rows * lpProblem->columns; i++) {
		lpProblem->A[i] = 0.;
	}
	int32_t *indices = (int32_t *) malloc(lpProblem->columns * sizeof(int32_t));
	double *values = (double *) malloc(lpProblem->columns * sizeof(double));
	// for glpk i + 1 (indices one-based)
	for (int32_t i = 0; i < lpProblem->rows; i++) {
		if (glp_get_row_type(lp, i + 1) == GLP_FR) {
			// ignore cost row
			printf("shouldn't be here!!\n");
			continue;
		}
		int32_t numberValues = glp_get_mat_row(lp, i + 1, indices, values);
		for (uint32_t j = 0; j < numberValues; j++) {
			lpProblem->A[i * lpProblem->columns + indices[j + 1] - 1] = values[j + 1];
		}
	}
	free(indices);
	free(values);

	for (int32_t i = 0; i < lpProblem->rows; i++) {
		// constraints are expected to be in form A*x=b
		if (glp_get_row_type(lp, i + 1) == GLP_FX) {
			lpProblem->b[i] = glp_get_row_lb(lp, i + 1);
		} else {
			printf("Can only handle constraints in form A*x=b!");
			exit(EXIT_FAILURE);
		}
	}

	for (uint32_t i = 0; i < lpProblem->columns; i++) {
		lpProblem->c[i] = glp_get_obj_coef(lp, i + 1);
	}

	for (int32_t i = 0; i < lpProblem->columns; i++) {
		lpProblem->lowerBound[i] = glp_get_col_lb(lp, i + 1);
		lpProblem->upperBound[i] = glp_get_col_ub(lp, i + 1);
	}

	glp_delete_prob(lp);
	glp_free_env();
}

void convertToStandardform(LPProblem *source, LPProblem *converted) {
	converted->isBasisAllocated = source->isBasisAllocated;
	converted->rows= source->rows + source->columns;
	converted->columns = source->columns * 2;
	converted->nnz = source->nnz + source->columns * 2;
	cudaCheck(cudaMallocManaged(&converted->A, converted->rows * converted->columns * sizeof(double)));
	for (int32_t i = 0; i < converted->rows; i++) {
		for (int32_t j = 0; j < converted->columns; j++) {
			if (i < source->rows && j < source->columns) {
				converted->A[i * converted->columns + j] = source->A[i * source->columns + j];
			} else if(i < source->rows) {
				converted->A[i * converted->columns + j] = 0;
			} else if(j < source->columns) {
				if (i - source->rows == j) {
					converted->A[i * converted->columns + j] = 1;
				} else {
					converted->A[i * converted->columns + j] = 0;
				}
			} else {
				if (i - source->rows == j - source->columns) {
					converted->A[i * converted->columns + j] = 1;
				} else {
					converted->A[i * converted->columns + j] = 0;
				}
			}
		}
	}
	cudaCheck(cudaMallocManaged(&converted->b, converted->rows * sizeof(double)));
	for (int32_t i = 0; i < converted->rows; i++) {
		if (i < source->rows) {
			converted->b[i] = source->b[i];
			for (int32_t j = 0; j < source->columns; j++) {
				converted->b[i] -= source->A[i * source->columns + j] * source->lowerBound[j];
			}
		} else {
			converted->b[i] = source->upperBound[i - source->rows] - source->lowerBound[i - source->rows];
		}
		if (converted->b[i] < 0) {
			converted->b[i] = -converted->b[i];
			for (int j = 0; j < converted->columns; j++) {
				converted->A[i * converted->columns + j] = -converted->A[i * converted->columns + j];
			}
		}
	}
	cudaCheck(cudaMallocManaged(&converted->c, converted->columns * sizeof(double)));
	for (int32_t i = 0; i < converted->columns; i++) {
		if (i < source->columns) {
			converted->c[i] = source->c[i];
		} else {
			converted->c[i] = 0;
		}
	}
}

void copyLPProblem(LPProblem *source, LPProblem *destination) {
	if (source->isBasisAllocated) {
		initializeLPProblem(destination, source->rows, source->columns, source->nnz);
		destination->isBasisAllocated = true;
		destination->rows = source->rows;
		destination->columns = source->columns;
		destination->nnz = source->nnz;
		cudaCheck(cudaMemcpy(destination->A, source->A, destination->rows * destination->columns * sizeof(double), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->b, source->b, destination->rows * sizeof(double), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->c, source->c, destination->columns * sizeof(double), cudaMemcpyDeviceToDevice));

		destination->nnzAB = source->nnzAB;
		cudaCheck(cudaMemcpy(destination->AB, source->AB, destination->rows * destination->rows * sizeof(double), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->ABRowPointer, source->ABRowPointer, (destination->rows + 1) * sizeof(int32_t), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->ABColumnIndices, source->ABColumnIndices, destination->nnz * sizeof(int32_t), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->ABValues, source->ABValues, destination->nnz * sizeof(double), cudaMemcpyDeviceToDevice));

		cudaCheck(cudaMemcpy(destination->ABTRowPointer, source->ABTRowPointer, (destination->rows + 1) * sizeof(int32_t), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->ABTColumnIndices, source->ABTColumnIndices, destination->nnz * sizeof(int32_t), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->ABTValues, source->ABTValues, destination->nnz * sizeof(double), cudaMemcpyDeviceToDevice));

		destination->nnzANB = source->nnzANB;
		cudaCheck(cudaMemcpy(destination->ANB, source->ANB, destination->rows * (destination->columns - destination->rows) * sizeof(double), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->ANBRowPointer, source->ANBRowPointer, (destination->rows + 1) * sizeof(int32_t), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->ANBColumnIndices, source->ANBColumnIndices, destination->nnz * sizeof(int32_t), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->ANBValues, source->ANBValues, destination->nnz * sizeof(double), cudaMemcpyDeviceToDevice));

		cudaCheck(cudaMemcpy(destination->cB, source->cB, destination->rows * sizeof(double), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->cBIndex, source->cBIndex, destination->rows * sizeof(int32_t), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->cNB, source->cNB, (destination->columns - destination->rows) * sizeof(double), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->cNBIndex, source->cNBIndex, (destination->columns - destination->rows) * sizeof(int32_t), cudaMemcpyDeviceToDevice));


		cudaCheck(cudaMemcpy(destination->xB, source->xB, destination->rows * sizeof(double), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->xIndex, source->xIndex, destination->columns * sizeof(int32_t), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->lowerBound, source->lowerBound, destination->columns * sizeof(double), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMemcpy(destination->upperBound, source->upperBound, destination->columns * sizeof(double), cudaMemcpyDeviceToDevice));
	} else {
		destination->isBasisAllocated = false;
		destination->rows = source->rows;
		destination->columns = source->columns;
		destination->nnz = source->nnz;
		cudaCheck(cudaMallocManaged(&destination->A, destination->rows * destination->columns * sizeof(double)));
		cudaCheck(cudaMemcpy(destination->A, source->A, destination->rows * destination->columns * sizeof(double), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMallocManaged(&destination->b, destination->rows * sizeof(double)));
		cudaCheck(cudaMemcpy(destination->b, source->b, destination->rows * sizeof(double), cudaMemcpyDeviceToDevice));
		cudaCheck(cudaMallocManaged(&destination->c, destination->columns * sizeof(double)));
		cudaCheck(cudaMemcpy(destination->c, source->c, destination->columns * sizeof(double), cudaMemcpyDeviceToDevice));
	}
}

void deleteLPProblem(LPProblem *lpProblem) {
	cudaFree(lpProblem->A);
	cudaFree(lpProblem->b);
	cudaFree(lpProblem->c);
	cudaFree(lpProblem->lowerBound);
	cudaFree(lpProblem->upperBound);
	if (lpProblem->isBasisAllocated) {
		cudaFree(lpProblem->AB);
		cudaFree(lpProblem->ABRowPointer);
		cudaFree(lpProblem->ABColumnIndices);
		cudaFree(lpProblem->ABValues);
		cudaFree(lpProblem->ABTRowPointer);
		cudaFree(lpProblem->ABTColumnIndices);
		cudaFree(lpProblem->ABTValues);
		cudaFree(lpProblem->ANB);
		cudaFree(lpProblem->ANBRowPointer);
		cudaFree(lpProblem->ANBColumnIndices);
		cudaFree(lpProblem->ANBValues);
		cudaFree(lpProblem->cB);
		cudaFree(lpProblem->cBIndex);
		cudaFree(lpProblem->cNB);
		cudaFree(lpProblem->cNBIndex);
		cudaFree(lpProblem->xB);
		cudaFree(lpProblem->xIndex);
		cudaFree(lpProblem->s);
		cudaFree(lpProblem->g);
		cudaFree(lpProblem->gTemp);
		cudaFree(lpProblem->ANBColumn);
		cudaFree(lpProblem->nnzPerRow);
		cudaFree(lpProblem->row);
		cudaFree(lpProblem->column);
	}
	free(lpProblem);
}

void initializeLPProblem(LPProblem *lpProblem, int32_t rows, int32_t columns,
		int32_t nnz) {
	lpProblem->isBasisAllocated = true;
	lpProblem->rows = rows;
	lpProblem->columns = columns;
	lpProblem->nnz = nnz;
	cudaCheck(
			cudaMallocManaged(&lpProblem->A, rows * columns * sizeof(double)));
	cudaCheck(cudaMallocManaged(&lpProblem->b, rows * sizeof(double)));
	cudaCheck(cudaMallocManaged(&lpProblem->c, columns * sizeof(double)));
	cudaCheck(
			cudaMallocManaged(&lpProblem->lowerBound,
					columns * sizeof(double)));
	cudaCheck(
			cudaMallocManaged(&lpProblem->upperBound,
					columns * sizeof(double)));

	cudaCheck(cudaMallocManaged(&lpProblem->AB, rows * rows * sizeof(double)));
	cudaCheck(
			cudaMallocManaged(&lpProblem->ABRowPointer,
					(rows + 1) * sizeof(int32_t)));
	cudaCheck(
			cudaMallocManaged(&lpProblem->ABColumnIndices,
					nnz * sizeof(int32_t)));
	cudaCheck(cudaMallocManaged(&lpProblem->ABValues, nnz * sizeof(double)));

	cudaCheck(
			cudaMallocManaged(&lpProblem->ABTRowPointer,
					(rows + 1) * sizeof(int32_t)));
	cudaCheck(
			cudaMallocManaged(&lpProblem->ABTColumnIndices,
					nnz * sizeof(int32_t)));
	cudaCheck(cudaMallocManaged(&lpProblem->ABTValues, nnz * sizeof(double)));

	cudaCheck(
			cudaMallocManaged(&lpProblem->ANB,
					rows * (columns - rows) * sizeof(double)));
	cudaCheck(
			cudaMallocManaged(&lpProblem->ANBRowPointer,
					(rows + 1) * sizeof(int32_t)));
	cudaCheck(
			cudaMallocManaged(&lpProblem->ANBColumnIndices,
					nnz * sizeof(int32_t)));
	cudaCheck(cudaMallocManaged(&lpProblem->ANBValues, nnz * sizeof(double)));

	cudaCheck(cudaMallocManaged(&lpProblem->cB, rows * sizeof(double)));
	cudaCheck(cudaMallocManaged(&lpProblem->cBIndex, rows * sizeof(int32_t)));
	cudaCheck(
			cudaMallocManaged(&lpProblem->cNB,
					(columns - rows) * sizeof(double)));
	cudaCheck(
			cudaMallocManaged(&lpProblem->cNBIndex,
					(columns - rows) * sizeof(int32_t)));

	cudaCheck(cudaMallocManaged(&lpProblem->xB, rows * sizeof(double)));
	cudaCheck(cudaMallocManaged(&lpProblem->xIndex, columns * sizeof(int32_t)));
	cudaCheck(cudaMallocManaged(&lpProblem->s, rows * sizeof(double)));
	cudaCheck(
			cudaMallocManaged(&lpProblem->g,
					(columns - rows) * sizeof(double)));
	cudaCheck(
			cudaMallocManaged(&lpProblem->gTemp,
					(columns - rows) * sizeof(double)));
	cudaCheck(cudaMallocManaged(&lpProblem->ANBColumn, rows * sizeof(double)));
	cudaCheck(cudaMallocManaged(&lpProblem->nnzPerRow, rows * sizeof(int32_t)));

	cudaCheck(cudaMallocManaged(&lpProblem->row, sizeof(int32_t)));
	cudaCheck(cudaMallocManaged(&lpProblem->column, sizeof(int32_t)));
}
