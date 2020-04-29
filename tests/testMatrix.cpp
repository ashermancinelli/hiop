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

    global_ordinal_type M_global  = 10;
    global_ordinal_type K_local   = 15;
    global_ordinal_type N_local   = 100;

    // all distribution occurs column-wise, so any length 
    // that will be used as a column of a matrix will have
    // to be scaled up by numRanks
    global_ordinal_type N_global = N_local * numRanks;
    global_ordinal_type K_global = K_local * numRanks;
    
    auto n_partition = new global_ordinal_type[numRanks+1];
    n_partition[0] = 0;
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
        hiop::hiopMatrixDense B_mxn(M_global, N_global, n_partition, comm);
        hiop::hiopMatrixDense A_nxn(N_global, N_global, n_partition, comm);
        hiop::hiopMatrixDense B_nxn(N_global, N_global, n_partition, comm);
        hiop::hiopMatrixDense A_kxn(K_global, N_global, n_partition, comm);
        hiop::hiopMatrixDense A_mxk(M_global, K_global, k_partition, comm);
        //                      ^^^
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
            /* 
             * Note that these ended up not being used
             * and are not as high of a priority
            fail += test.matrixTimesMat(A_mxk, A_kxn, A_mxn, rank);
            fail += test.matrixTransTimesMat(A_mxk, A_kxn, A_mxn, rank);
            fail += test.matrixTimesMatTrans(A_mxk, A_kxn, A_mxn, rank);
            */
            fail += test.matrixAddDiagonal(A_nxn, x_n, rank);
            fail += test.matrixAddSubDiagonal(A_nxn, x_n, rank);
        }

        fail += test.matrixAddToSymDenseMatrixUpperTriangle(A_nxn, A_mxk, rank);
        fail += test.matrixAddMatrix(A_mxn, B_mxn, rank);
        fail += test.matrixAssertSymmetry(A_nxn, rank);
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
