/**
 * Defines LP Problem.
 */

#ifndef LPPROBLEM_CUH_
#define LPPROBLEM_CUH_

#include <stdbool.h>
#include <stdint.h>

typedef struct LPProblem {
	bool isBasisAllocated;
	bool isOptimal;
	bool isUnbounded;
	bool isSolution;

	int32_t rows;
	int32_t columns;
	int32_t nnz;
	double *A;

	int32_t nnzAB;
	double *AB;
	int32_t *ABRowPointer;
	int32_t *ABColumnIndices;
	double *ABValues;

	int32_t *ABTRowPointer;
	int32_t *ABTColumnIndices;
	double *ABTValues;

	int32_t nnzANB;
	double *ANB;
	int32_t *ANBRowPointer;
	int32_t *ANBColumnIndices;
	double *ANBValues;

	double *b;

	double *c;
	double *cB;
	int32_t *cBIndex;
	double *cNB;
	int32_t *cNBIndex;

	double *xB;
	int32_t *xIndex;
	double *lowerBound;
	double *upperBound;

	double *s;
	double *g;
	double *gTemp;
	double *ANBColumn;
	int32_t *nnzPerRow;
	int32_t *row;
	int32_t *column;
} LPProblem;

void readMPS(char *mpsFile, LPProblem *lpProblem);

void convertToStandardform(LPProblem *source, LPProblem *converted);

void copyLPProblem(LPProblem *source, LPProblem *destination);

void deleteLPProblem(LPProblem *lpProblem);

void initializeLPProblem(LPProblem *lpProblem, int32_t rows, int32_t columns,
		int32_t nnz);

#endif /* LPPROBLEM_CUH_ */
