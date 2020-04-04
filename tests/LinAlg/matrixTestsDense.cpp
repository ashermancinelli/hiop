#include <hiopMatrix.hpp>
#include "matrixTestsDense.hpp"

namespace hiop::tests {

/// Method to set matrix _A_ element (i,j) to _val_.
/// First need to retrieve hiopMatrixDense from the abstract interface
void MatrixTestsDense::setElement(hiop::hiopMatrix* A, local_ordinal_type i, local_ordinal_type j, real_type val)
{
    hiop::hiopMatrixDense* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
    real_type** data = amat->get_M();
    data[i][j] = val;
}

/// Returns element (i,j) of matrix _A_.
/// First need to retrieve hiopMatrixDense from the abstract interface
real_type MatrixTestsDense::getElement(hiop::hiopMatrix* A, local_ordinal_type i, local_ordinal_type j)
{
    hiop::hiopMatrixDense* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
    return amat->local_data()[i][j];
}

local_ordinal_type MatrixTestsDense::getNumLocRows(hiop::hiopMatrix* A)
{
    hiop::hiopMatrixDense* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
    return amat->get_local_size_m();
    //                         ^^^
}

local_ordinal_type MatrixTestsDense::getNumLocCols(hiop::hiopMatrix* A)
{
    hiop::hiopMatrixDense* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
    return amat->get_local_size_n();
    //                         ^^^
}

#ifdef HIOP_USE_MPI
/// Get communicator
MPI_Comm MatrixTestsDense::getMPIComm(hiop::hiopMatrix* _A)
{
    const hiop::hiopMatrixDense* A = dynamic_cast<const hiop::hiopMatrixDense*>(_A);
    return A->get_mpi_comm();
}
#endif

// Every rank returns failure if any individual rank fails
bool MatrixTestsDense::reduceReturn(int failures, hiop::hiopMatrix* A)
{
    int fail = 0;

    if constexpr (USE_MPI)
    {
        MPI_Allreduce(&failures, &fail, 1, MPI_INT, MPI_SUM, getMPIComm(A));
    }
    else
    {
        fail = failures;
    }

    return (fail != 0);
}

[[nodiscard]]
int MatrixTestsDense::verifyAnswer(hiop::hiopMatrix* A, const double answer)
{
    const int M = getNumLocRows(A);
    const int N = getNumLocCols(A);
    int fail = 0;
    for (int i=0; i<M; i++)
        for (int j=0; j<N; j++)
            if (!isEqual(getElement(A, i, j), answer))
                fail++;
    return fail;
}

} // namespace hiop::tests
