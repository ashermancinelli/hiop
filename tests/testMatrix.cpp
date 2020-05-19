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

    global_ordinal_type M_local = 10;
    global_ordinal_type K_local = 15;
    global_ordinal_type N_local = 100;

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
        // Distributed matrices:
        hiop::hiopMatrixDense A_mxn(M_local, N_global, n_partition, comm);
        hiop::hiopMatrixDense B_mxn(M_local, N_global, n_partition, comm);
        hiop::hiopMatrixDense A_nxn(N_global, N_global, n_partition, comm);
        hiop::hiopMatrixDense B_nxn(N_global, N_global, n_partition, comm);
        hiop::hiopMatrixDense A_kxn(K_global, N_global, n_partition, comm);
        hiop::hiopMatrixDense A_kxk(K_global, K_global, k_partition, comm);
        hiop::hiopMatrixDense A_mxk(M_local, K_global, k_partition, comm);

        // Local matrices:
        hiop::hiopMatrixDense A_mxn_local(M_local, N_local);
        hiop::hiopMatrixDense A_mxk_local(M_local, K_local);
        hiop::hiopMatrixDense A_kxn_local(K_local, N_local);

        hiop::hiopVectorPar x_n(N_global, n_partition, comm);
        //                   ^^^
        hiop::hiopVectorPar x_m(M_local);
        //                   ^^^
        hiop::tests::MatrixTestsDense test;

        fail += test.matrixNumRows(A_mxn, M_local, rank);
        fail += test.matrixNumCols(A_mxn, N_global, rank);
        fail += test.matrixSetToZero(A_mxn, rank);
        fail += test.matrixSetToConstant(A_mxn, rank);
        fail += test.matrixTimesVec(A_mxn, x_m, x_n, rank);
        fail += test.matrixTransTimesVec(A_mxn, x_m, x_n, rank);

        if (numRanks == 1)
        {
            fail += test.matrixTimesMatLocal(A_mxk_local, A_kxn_local, A_mxn_local, rank);
            fail += test.matrixTransTimesMat(A_mxk, A_kxn, A_mxn, rank);
            fail += test.matrixTimesMatTrans(A_mxk, A_kxn, A_mxn, rank);

            // This function is only meant to be called locally
            fail += test.matrixAddDiagonal(A_nxn, x_n, rank);
            fail += test.matrixAddSubDiagonalLocal(A_nxn, x_m, rank);
        }
        else
        {
            fail += test.matrixAddSubDiagonalDistributed(A_nxn, x_m, rank);
        }

        fail += test.matrixAddMatrix(A_mxn, B_mxn, rank);
        fail += test.matrixAddToSymDenseMatrixUpperTriangle(A_nxn, A_mxk, rank);
        fail += test.matrixTransAddToSymDenseMatrixUpperTriangle(A_nxn, A_mxk, rank);
        fail += test.matrixAddUpperTriangleToSymDenseMatrixUpperTriangle(A_nxn, A_kxk, rank);
        fail += test.matrixMaxAbsValue(A_mxn, rank);
        fail += test.matrixIsFinite(A_mxn, rank);
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
