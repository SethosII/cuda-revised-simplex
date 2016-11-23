# cuda-revised-simplex

An implementation of the revised simplex algorithm in CUDA for solving linear optimization problems in the form `max{c*x | A*x=b, l<=x<=u}`. The LP problem is read from an MPS file by GLPK.

The implementations uses the following data structures. The coefficient matrix and objective function values are splitted in two parts. One contains the basic variables and the other the nonbasic variables. The matrices are stored in array format and CSR format. The CSR format is used for calculations and the array format for copying of columns. The CSR representation is updated at the end of each iteration. Temporary variables are allocated once at the beginning and reused throughout the iterations. All values are stored in double precision.

The LP problem is transformed into standard form. Lower bounds are eliminated via a shift. Upper bounds are integrated by addition of new equations. Negative right hand size are eliminated by multiplication of the row with -1.

The implemented simplex algorithm is as follows. The current basis solution is calculated by solving the system of linear equations `A_B*x_B=b` with a QR decomposition from the cuSolver library. The basis solution can also be calculated by updating the old basis solution via `x_B_i=x_B_i-s_i*x_B_row/s_row, x_B_row=x_B_row/s_row$`. Afterwards reduced costs are calculated in two steps. First the system of linear equations `A_B^T*s=c_B` is solved with QR decomposition. The routine can't handle matrix transpose implicitly, so the CSR matrix is transposed explicitly by converting it into a CSC format and reinterpreting it as CSR format.  Afterwards the reduced costs are calculated with a matrix-vector product with cuSparse library `g=A_NB^T*s-c_NB`. The outgoing variable is chosen via Dantzig's or steepest-edge rule. Dantzig's rule chooses the variable with the most negative reduced cost `column={i|min{g_i}, g_i<0}`. This is implemented as a parallel reduction over the values of `g`. The values aren't compared directly to zero but rather to a small tolerance like `-10^-12`. This is necessary because of numerical inaccuracy whereas very small values can arise that would be zero when solved exactly. The steepest-edge rule scales the values of `g` with the norm of the corresponding column of the basis matrix `g_i=g_i/||A_NB_i||`. The norm is calculated as the square root of the sum of squares `||v||=sum_i v_i^2` with the CSR representation. For this purpose the square of each element is added atomically on the corresponding position. After all elements are processed the values of the vector `g` are divided by the square root of the calculated values. The steepest edge rule has in general a better convergence behaviour and therefore takes less iterations. Here the smallest negative value is also used which is calculated by a parallel reduction. The current solution is optimal if such a value isn't found. The entering variable is calculated with the minimum ratio test. Therefore the equation system `A_B*s=A_NB_column` is solved by with a QR decomposition from the cuSolver library. The needed column of the non nonbasic matrix is copied from the array representation. The entering column is calculated by a parallel reduction of the ratio of basic variable and the corresponding values of `s`: `row={i|min{x_B_i/s_i}, s_i>0}`. The values of `s` are also not directly checked for positivity but rather if they are above a certain tolerance like `10^-12`. The lp problem is unbounded if such a value isn't found. After choosing entering and leaving variable, the data structures need to be updated. For this the corresponding columns in the array format of basic and nonbasic matrix are swapped. Afterwards the CSR representation is updated. Also the corresponding entries in basic and nonbasic objective function values are swapped.

## Build

```
nvcc -o cuda-revised-simplex src/*.cu -lcusolver -lcusparse -lglpk -arch=sm_35 --relocatable-device-code=true -O3
```

## Run

```
./cuda-revised-simplex simpleProblem.mps
```

The solution for the simple example should be 3.3333 for X8 (index [7]) (the objective function). Different values are possible for other variables because the problem has multiple optimal solutions.
