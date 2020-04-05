#include <iostream>
#include <assert.h>

// This header contains HiOp's MPI definitions
#include <hiopVector.hpp>

#include "LinAlg/vectorTestsPar.hpp"
// #include "LinAlg/vectorTestsRAJA.hpp"


/**
 * @brief Main body of vector implementation testing code.
 *
 * @todo The size of the vector should be passed on the command line.
 *
 * @pre All test functions should return the same boolean value on all ranks.
 *
 */
int main(int argc, char** argv)
{
    int rank=0;
    int numRanks=1;
    long long* partition = nullptr;
    MPI_Comm comm = MPI_COMM_NULL;

    if constexpr (USE_MPI)
    {
        int err;
        err = MPI_Init(&argc, &argv);         assert(MPI_SUCCESS == err);
        comm = MPI_COMM_WORLD;
        err = MPI_Comm_rank(comm, &rank);     assert(MPI_SUCCESS == err);
        err = MPI_Comm_size(comm, &numRanks); assert(MPI_SUCCESS == err);
        if(0 == rank)
            std::cout << "Running MPI enabled tests ...\n";
    }

    long long Nlocal = 1000;
    long long Nglobal = Nlocal*numRanks;
    partition = new long long [numRanks + 1];
    partition[0] = 0;
    for(int i = 1; i < numRanks + 1; ++i)
        partition[i] = i*Nlocal;

    int fail = 0;

    // Test parallel vector
    {
        hiop::hiopVectorPar x(Nglobal, partition, comm);
        hiop::hiopVectorPar* y = x.alloc_clone();
        hiop::hiopVectorPar* z = x.alloc_clone();
        hiop::hiopVectorPar* a = x.alloc_clone();
        hiop::hiopVectorPar* b = x.alloc_clone();
        hiop::tests::VectorTestsPar test;

        fail += test.vectorGetSize(x, Nglobal, rank);
        fail += test.vectorSetToZero(x, rank);
        fail += test.vectorSetToConstant(x, rank);
        fail += test.vectorSetToConstant_w_patternSelect(x, *y, rank);
        fail += test.vectorCopyFrom(x, *y, rank);
        fail += test.vectorCopyTo(x, *y, rank);

        if (numRanks == 1)
        {
            fail += test.vectorCopyFromStarting(x, *y, rank);
            fail += test.vectorStartingAtCopyFromStartingAt(x, *y, rank);
            fail += test.vectorCopyToStarting(x, *y, rank);
            fail += test.vectorStartingAtCopyToStartingAt(x, *y, rank);
        }

        fail += test.vectorTwonorm(x, rank);
        fail += test.vectorInfnorm(x, rank);
        fail += test.vectorOnenorm(x, rank);
        fail += test.vectorComponentMult(x, *y, rank);
        fail += test.vectorComponentDiv(x, *y, rank);
        fail += test.vectorComponentDiv_p_selectPattern(x, *y, *z, rank);
        fail += test.vectorScale(x, rank);

        fail += test.vectorAxpy(x, *y, rank);
        fail += test.vectorAxzpy(x, *y, *z, rank);
        fail += test.vectorAxdzpy(x, *y, *z, rank);

        fail += test.vectorAddConstant(x, rank);
        fail += test.vectorAddConstant_w_patternSelect(x, *y, rank);
        fail += test.vectorDotProductWith(x, *y, rank);
        fail += test.vectorNegate(x, rank);
        fail += test.vectorInvert(x, rank);
        fail += test.vectorLogBarrier(x, *y, rank);
        fail += test.vectorAddLogBarrierGrad(x, *y, *z, rank);
        fail += test.vectorLinearDampingTerm(x, *y, *z, rank);

        fail += test.vectorAllPositive(x, rank);
        fail += test.vectorAllPositive_w_patternSelect(x, *y, rank);

        // fail += test.vectorMin(x, rank);
        fail += test.vectorProjectIntoBounds(x, *y, *z, *a, *b, rank);
        fail += test.vectorFractionToTheBdry(x, *y, rank);
        fail += test.vectorFractionToTheBdry_w_pattern(x, *y, *z, rank);

        fail += test.vectorSelectPattern(x, *y, rank);
        fail += test.vectorMatchesPattern(x, *y, rank);
        fail += test.vectorAdjustDuals_plh(x, *y, *z, *a, rank);
        fail += test.vectorIsnan(x, rank);
        fail += test.vectorIsinf(x, rank);
        fail += test.vectorIsfinite(x, rank);
    }

    // Test RAJA vector
    {
        //         hiop::hiopVectorRAJA x(N);
        //         hiop::tests::VectorTestsRAJA test;
        //
        //         fail += test.testGetSize(x, N);
        //         fail += test.testSetToConstant(x);
    }


    if (rank == 0)
    {
        if(fail)
            std::cout << fail << " tests failed\n";
        else
            std::cout << "All tests passed\n";
    }

    if constexpr (USE_MPI)
    {
        MPI_Finalize();
    }

    return fail;
}
