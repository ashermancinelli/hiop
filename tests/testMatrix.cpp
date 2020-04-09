#include <iostream>
#include <cassert>

#include <hiopVector.hpp>
#include <hiopMatrix.hpp>
#include "LinAlg/matrixTestsDense.hpp"

int main(int argc, char** argv)
{
    int rank=0, numRanks=1;
    MPI_Comm comm = MPI_COMM_NULL;

#ifdef HIOP_USE_MPI
    int err;
    err = MPI_Init(&argc, &argv);                  assert(MPI_SUCCESS==err);
    comm = MPI_COMM_WORLD;
    err = MPI_Comm_rank(comm,&rank);     assert(MPI_SUCCESS==err);
    err = MPI_Comm_size(comm,&numRanks); assert(MPI_SUCCESS==err);
    if(0 == rank)
        printf("Support for MPI is enabled\n");
#endif

    global_ordinal_type M = 10;  // rows
    global_ordinal_type N = 100; // columns
    long long M_local   = 10;  // local rows

    // all distribution occurs column-wise
    long long N_local   = 100; // local columns
    long long N_global  = N_local * numRanks; // global columns
    
    auto partition = new long long[numRanks+1];
    partition[0] = 0;
    for(int i = 1; i < numRanks + 1; ++i)
        partition[i] = i*N_local;

    int fail = 0;

    // Test dense matrix
    {
        // initialize MxN matrices
        hiop::hiopMatrixDense A_mxn(M_local, N_global, partition, comm);
        hiop::hiopMatrixDense* B_mxn = A_mxn.alloc_clone();
        hiop::hiopMatrixDense* C_mxn = A_mxn.alloc_clone();

        // NxN matrices
        hiop::hiopMatrixDense A_nxn(N_global, N_global, partition, comm);
        hiop::hiopMatrixDense* B_nxn = A_nxn.alloc_clone();
        hiop::hiopMatrixDense* C_nxn = A_nxn.alloc_clone();

        // set up distributed vectors of size N
        hiop::hiopVectorPar x_n(N_global, partition, comm);
        hiop::hiopVectorPar* y_n= x_n.alloc_clone();
        // set up local vectors of size M
        hiop::hiopVectorPar x_m(M_local);
        hiop::hiopVectorPar* y_m = x_m.alloc_clone();
        hiop::tests::MatrixTestsDense test;

        fail += test.matrixNumRows(A_mxn, M_local, rank);
        fail += test.matrixNumCols(A_mxn, N_global, rank);
        fail += test.matrixSetToZero(A_mxn, rank);
        fail += test.matrixSetToConstant(A_mxn, rank);
        fail += test.matrixTimesVec(A_mxn, x_m, x_n, rank);
        fail += test.matrixTransTimesVec(A_mxn, x_m, x_n, rank);

        fail += test.matrixTimesMat(A_mxn, A_nxn, *B_mxn, rank);
        fail += test.matrixTransTimesMat(A_mxn, *B_nxn, *C_nxn, rank);
        fail += test.matrixTimesMatTrans(A_mxn, *B_nxn, *C_nxn, rank);
    }

    // Test RAJA matrix
    {
        // Code here ...
    }

    if (rank == 0)
    {
        if(fail)
        {
            std::cout << "Matrix tests failed\n";
        }
        else
        {
            std::cout << "Matrix tests passed\n";
        }
    }

#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif

    return fail;
}
