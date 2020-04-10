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
        n_partition[i] = i*N_local;

    auto k_partition = new global_ordinal_type[numRanks+1];
    k_partition[0] = 0;
    for(int i = 1; i < numRanks + 1; ++i)
        k_partition[i] = i*K_local;

    int fail = 0;

    // Test dense matrix
    {
        // Matrix dimensions denoted by subscript
        hiop::hiopMatrixDense A_mxn(M_global, N_global, n_partition, comm);
        hiop::hiopMatrixDense A_kxn(K_global, N_global, n_partition, comm);
        hiop::hiopMatrixDense A_mxk(M_global, K_global, k_partition, comm);

        hiop::hiopVectorPar x_n(N_global, n_partition, comm);
        //                   ^^^
        hiop::hiopVectorPar x_m(M_global);
        //                   ^^^
        hiop::tests::MatrixTestsDense test;

        fail += test.matrixNumRows(A_mxn, M_global, rank);
        fail += test.matrixNumCols(A_mxn, N_global, rank);
        fail += test.matrixSetToZero(A_mxn, rank);
        fail += test.matrixSetToConstant(A_mxn, rank);
        fail += test.matrixTimesVec(A_mxn, x_m, x_n, rank);
        fail += test.matrixTransTimesVec(A_mxn, x_m, x_n, rank);

        if (numRanks == 1)
        {
            fail += test.matrixTimesMat(A_mxk, A_kxn, A_mxn, rank);
            // fail += test.matrixTransTimesMat(A_mxk, A_kxn, A_mxn, rank);
            // fail += test.matrixTimesMatTrans(A_mxn, *B_nxn, *C_nxn, rank);
        }
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
