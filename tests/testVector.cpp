#include <iostream>
#include <assert.h>

// This header contains HiOp's MPI definitions
#include <hiopVector.hpp>
#include <hiopVectorRajaPar.hpp>

#include "LinAlg/vectorTestsPar.hpp"
#include "LinAlg/vectorTestsRajaPar.hpp"


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
    global_ordinal_type* partition = nullptr;
    MPI_Comm comm = MPI_COMM_NULL;

#ifdef HIOP_USE_MPI
    int err;
    err = MPI_Init(&argc, &argv);         assert(MPI_SUCCESS == err);
    comm = MPI_COMM_WORLD;
    err = MPI_Comm_rank(comm, &rank);     assert(MPI_SUCCESS == err);
    err = MPI_Comm_size(comm, &numRanks); assert(MPI_SUCCESS == err);
    if(0 == rank)
        std::cout << "Running MPI enabled tests ...\n";
#endif

    global_ordinal_type Nlocal = 1000;
    global_ordinal_type Nglobal = Nlocal*numRanks;
    partition = new global_ordinal_type [numRanks + 1];
    partition[0] = 0;
    for(int i = 1; i < numRanks + 1; ++i)
        partition[i] = i*Nlocal;

    int fail = 0;

    // Test parallel vector
    {
        if (rank == 0)
          std::cout << "\nTesting default HiOp vector:\n";

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
        if (rank == 0)
          std::cout << "\nTesting HiOp RAJA vector:\n";

        hiop::hiopVectorRajaPar x(Nglobal, partition, comm);
        hiop::hiopVectorRajaPar y(Nglobal, partition, comm);
        hiop::hiopVectorRajaPar z(Nglobal, partition, comm);
        hiop::hiopVectorRajaPar a(Nglobal, partition, comm);
        hiop::hiopVectorRajaPar b(Nglobal, partition, comm);
        hiop::tests::VectorTestsRajaPar test;

        fail += test.vectorSetToConstant(x, rank);
        fail += test.vectorSetToConstant_w_patternSelect(x, y, rank);
        fail += test.vectorSetToZero(x, rank);
        fail += test.vectorOnenorm(x, rank);
        fail += test.vectorIsfinite(x, rank);
        fail += test.vectorIsinf(x, rank);
        fail += test.vectorIsnan(x, rank);
        fail += test.vectorAllPositive_w_patternSelect(x, y, rank);
        fail += test.vectorAdjustDuals_plh(x, y, z, a, rank);
        // fail += test.vectorAxdzpy_w_patternSelect(x, y, z, rank);
        fail += test.vectorFractionToTheBdry_w_pattern(x, y, z, rank);
        fail += test.vectorInfnorm(x, rank);
        fail += test.vectorComponentMult(x, y, rank);
        fail += test.vectorComponentDiv(x, y, rank);
        fail += test.vectorComponentDiv_p_selectPattern(x, y, z, rank);

        fail += test.vectorAxzpy(x, y, z, rank);
        fail += test.vectorAxdzpy(x, y, z, rank);

        fail += test.vectorAddConstant(x, rank);
        fail += test.vectorAddConstant_w_patternSelect(x, y, rank);
        fail += test.vectorInvert(x, rank);
        fail += test.vectorLogBarrier(x, y, rank);
        fail += test.vectorAddLogBarrierGrad(x, y, z, rank);
        fail += test.vectorLinearDampingTerm(x, y, z, rank);

        fail += test.vectorAllPositive(x, rank);
        fail += test.vectorProjectIntoBounds(x, y, z, a, b, rank);
    }


    if (rank == 0)
    {
        std::cout << "\n";
        if(fail)
            std::cout << fail << " tests failed\n";
        else
            std::cout << "All tests passed\n";
    }

#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif

    return fail;
}
