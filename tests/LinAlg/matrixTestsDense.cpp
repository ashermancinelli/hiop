// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp 
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause). 
// Please also read “Additional BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list 
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice, 
// this list of conditions and the disclaimer (as noted below) in the documentation and/or 
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to 
// endorse or promote products derived from this software without specific prior written 
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR 
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS 
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department 
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under 
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC 
// nor any of their employees, makes any warranty, express or implied, or assumes any 
// liability or responsibility for the accuracy, completeness, or usefulness of any 
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by 
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or 
// imply its endorsement, recommendation, or favoring by the United States Government or 
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed 
// herein do not necessarily state or reflect those of the United States Government or 
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or 
// product endorsement purposes.

#include <hiopMatrix.hpp>
#include "matrixTestsDense.hpp"

namespace hiop::tests {

// Start hiopMatrixDense matrix tests
int MatrixTestsDense::matrixCopyFrom(
    hiopMatrixDense& dst,
    hiopMatrixDense& src,
    const int rank)
{
    assert(dst.n() == src.n() && "Did you pass in matrices of the same size?");
    assert(dst.m() == src.m() && "Did you pass in matrices of the same size?");
    assert(getNumLocRows(&dst) == getNumLocRows(&src) && "Did you pass in matrices of the same size?");
    assert(getNumLocCols(&dst) == getNumLocCols(&src) && "Did you pass in matrices of the same size?");
    const real_type src_val = one;

    // Test copying src another matrix
    src.setToConstant(src_val);
    dst.setToZero();

    dst.copyFrom(src);
    int fail = verifyAnswer(&dst, src_val);

    // test copying src a raw buffer
    const size_t buf_len = getNumLocRows(&src) * getNumLocCols(&src);
    real_type* src_buf = new real_type[buf_len];
    std::fill_n(src_buf, buf_len, src_val);
    dst.setToZero();

    dst.copyFrom(src_buf);
    fail += verifyAnswer(&dst, src_val);
    delete[] src_buf;

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &dst);
}

int MatrixTestsDense::matrixAppendRow(
    hiopMatrixDense& A,
    hiopVectorPar& vec,
    const int rank)
{
    assert(A.n() == vec.get_size() 
        && "Did you pass in a vector with the same length as the number of columns of the matrix?");
    assert(getNumLocCols(&A) == vec.get_local_size()
        && "Did you pass in a vector with the same length as the number of columns of the matrix?");
    const local_ordinal_type init_num_rows = A.m();
    const real_type A_val = one;
    const real_type vec_val = two;
    int fail = 0;

    A.setToConstant(A_val);
    vec.setToConstant(vec_val);
    A.appendRow(vec);
    
    // Ensure A's num rows is updated
    if (A.m() != init_num_rows+1)
        fail++;

    // Ensure vec's values are copied over to A's last row
    fail += verifyAnswer(&A,
        [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
        {
            (void)j; // j is unused
            const bool isLastRow = (i == init_num_rows);
            return isLastRow ? vec_val : A_val;
        });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
}

int MatrixTestsDense::matrixCopyRowsFrom(
    hiopMatrixDense& dst,
    hiopMatrixDense& src,
    const int rank)
{
    assert(dst.n() == src.n());
    assert(dst.m() > src.m());
    assert(getNumLocCols(&dst) == getNumLocCols(&src));
    assert(getNumLocRows(&dst) > getNumLocRows(&src));
    const real_type dst_val = one;
    const real_type src_val = two;
    const local_ordinal_type dst_start_idx = dst.m() - src.m();
    local_ordinal_type num_rows_to_copy = src.m();
    const local_ordinal_type src_num_rows = src.m();

    // Test copying continuous rows from matrix
    dst.setToConstant(dst_val);
    src.setToConstant(src_val);

    dst.copyRowsFrom(src, num_rows_to_copy, dst_start_idx);

    int fail = verifyAnswer(&dst,
        [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
        {
            (void)j; // j is unused
            const bool isRowCopiedOver = (
              i >= dst_start_idx &&
              i < dst_start_idx + src_num_rows);
            return isRowCopiedOver ? src_val : dst_val;
        });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &dst);
}

int MatrixTestsDense::matrixCopyBlockFromMatrix(
    hiopMatrixDense& src,
    hiopMatrixDense& dst,
    const int rank)
{
    assert(src.n() < dst.n()
        && "Src mat must be smaller than dst mat");
    assert(src.m() < dst.m()
        && "Src mat must be smaller than dst mat");
    assert(getNumLocCols(&src) < getNumLocCols(&dst)
        && "Src mat must be smaller than dst mat");
    const real_type src_val = one;
    const real_type dst_val = two;

    // Copy matrix block into downmost and rightmost location
    // possible
    const local_ordinal_type src_num_rows = getNumLocRows(&src);
    const local_ordinal_type src_num_cols = getNumLocCols(&src);
    const local_ordinal_type dst_start_row = getNumLocRows(&dst) - src_num_rows;
    const local_ordinal_type dst_start_col = getNumLocCols(&dst) - src_num_cols;

    src.setToConstant(src_val);
    dst.setToConstant(dst_val);
    dst.copyBlockFromMatrix(dst_start_row, dst_start_col, src);

    const int fail = verifyAnswer(&dst,
        [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
        {
            const bool isIdxCopiedFromSource = (
                i >= dst_start_row && i < dst_start_row+src_num_rows &&
                j >= dst_start_col && j < dst_start_col+src_num_cols);
            return isIdxCopiedFromSource ? src_val : dst_val;
        });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &dst);
}

int MatrixTestsDense::matrixCopyFromMatrixBlock(
    hiopMatrixDense& src,
    hiopMatrixDense& dst,
    const int rank)
{
    assert(src.n() > dst.n()
        && "Src mat must be larger than dst mat");
    assert(src.m() > dst.m()
        && "Src mat must be larger than dst mat");
    assert(getNumLocCols(&src) > getNumLocCols(&dst)
        && "Src mat must be larger than dst mat");
    const local_ordinal_type dst_m = getNumLocRows(&dst);
    const local_ordinal_type dst_n = getNumLocCols(&dst);
    const local_ordinal_type src_m = getNumLocRows(&src);
    const local_ordinal_type src_n = getNumLocCols(&src);
    const local_ordinal_type block_start_row = (src_m - getNumLocRows(&dst)) - 1;
    const local_ordinal_type block_start_col = (src_n - getNumLocCols(&dst)) - 1;

    const real_type src_val = one;
    const real_type dst_val = two;
    src.setToConstant(src_val);
    if (rank == 0)
        setLocalElement(&src, src_m-1, src_n-1, zero);
    dst.setToConstant(dst_val);

    dst.copyFromMatrixBlock(src, block_start_row, block_start_col);

    const int fail = verifyAnswer(&dst,
        [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
        {
            // This is the element set to zero in src
            // before being copied over
            if (i == dst_m && j == dst_n && rank == 0)
                return zero;
            else
                return src_val;
        });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &dst);
}

/*
 * shiftRows does not wrap around when shifted past the end
 * of the matrix.
 *
 *    2 2 2                   2 2 2
 *    1 1 1                   2 2 2
 *    1 1 1  shiftRows(1) ->  1 1 1
 *
 *  The uppermost row is not overwritten by the last row
 */
int MatrixTestsDense::matrixShiftRows(
    hiopMatrixDense& A,
    const int rank)
{
    printMessage(SKIP_TEST, __func__, rank); return 0;
    const local_ordinal_type M = getNumLocRows(&A);
    local_ordinal_type uniq_row_idx = 0;
    local_ordinal_type shift = M-1;
    int fail = 0;

    const real_type A_val = one;
    const real_type uniq_row_val = two;

    // First check positive shift
    // Set one row to a unique value
    A.setToConstant(A_val);
    setLocalRow(&A, uniq_row_idx, uniq_row_val);
    A.shiftRows(shift);

    fail += verifyAnswer(&A,
        [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
        {
            (void)j; // j is unused
            const bool isUniqueRow = (
                i == (uniq_row_idx + shift) ||
                i == uniq_row_idx);
            return isUniqueRow ? uniq_row_val : A_val;
        });

    // Now check negative shift
    shift *= -1;
    uniq_row_idx = M - 1;
    A.setToConstant(A_val);
    setLocalRow(&A, uniq_row_idx, uniq_row_val);
    A.shiftRows(shift);

    fail += verifyAnswer(&A,
        [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
        {
            (void)j; // j is unused
            const bool isUniqueRow = (
                i == (uniq_row_idx + shift) ||
                i == uniq_row_idx);
            return isUniqueRow ? uniq_row_val : A_val;
        });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
}

int MatrixTestsDense::matrixReplaceRow(
    hiopMatrixDense& A,
    hiopVectorPar& vec,
    const int rank)
{
    const local_ordinal_type N = getNumLocCols(&A);
    const local_ordinal_type M = getNumLocRows(&A);
    assert(N == vec.get_local_size() && "Did you pass a vector and matrix of compatible lengths?");
    assert(A.n() == vec.get_size() && "Did you pass a vector and matrix of compatible lengths?");

    const local_ordinal_type row_idx = M - 1;
    const local_ordinal_type col_idx = 1;
    const real_type A_val = one;
    const real_type vec_val = two;
    A.setToConstant(A_val);
    vec.setToConstant(vec_val);
    setLocalElement(&vec, col_idx, zero);

    A.replaceRow(row_idx, vec);
    const int fail = verifyAnswer(&A,
        [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
        {
            // Was the row replaced?
            if (i == row_idx)
            {
                // Was the index of the row set to zero?
                if (j == col_idx)
                    return zero;
                else
                    return vec_val;
            }
            // The matrix should be otherwise unchanged
            else
            {
                return A_val;
            }
        });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
}

int MatrixTestsDense::matrixGetRow(
    hiopMatrixDense& A,
    hiopVectorPar& vec,
    const int rank)
{
    const local_ordinal_type N = getNumLocCols(&A);
    const local_ordinal_type M = getNumLocRows(&A);
    assert(N == vec.get_local_size() && "Did you pass a vector and matrix of compatible lengths?");
    assert(A.n() == vec.get_size() && "Did you pass a vector and matrix of compatible lengths?");

    // Set one value in the matrix row to be retrieved to be a unique value
    const local_ordinal_type row_idx = M - 1;
    const local_ordinal_type col_idx = 1;
    const real_type A_val = one;
    const real_type vec_val = two;
    A.setToConstant(A_val);
    setLocalElement(&A, row_idx, col_idx, zero);
    vec.setToConstant(vec_val);
    A.getRow(row_idx, vec);

    const int fail = verifyAnswer(&vec,
        [=] (local_ordinal_type i) -> real_type
        {
            if (i == col_idx)
                return zero;
            else
                return A_val;
        });

    printMessage(fail, __func__, rank);
    return reduceReturn(fail, &A);
}
// End hiopMatrixDense matrix tests

// Start helper methods
/// Method to set matrix _A_ element (i,j) to _val_.
/// First need to retrieve hiopMatrixDense from the abstract interface
void MatrixTestsDense::setLocalElement(
        hiop::hiopMatrix* A,
        local_ordinal_type i,
        local_ordinal_type j,
        real_type val)
{
    hiop::hiopMatrixDense* amat = dynamic_cast<hiop::hiopMatrixDense*>(A);
    real_type** data = amat->get_M();
    data[i][j] = val;
}

void MatrixTestsDense::setLocalElement(
        hiop::hiopVector* _x,
        const local_ordinal_type i,
        const real_type val)
{
    auto x = dynamic_cast<hiop::hiopVectorPar*>(_x);
    real_type* data = x->local_data();
    data[i] = val;
}

/// Method to set a single row of matrix to a constant value
void MatrixTestsDense::setLocalRow(
        hiop::hiopMatrixDense* A,
        const local_ordinal_type row,
        const real_type val)
{
    const local_ordinal_type N = getNumLocCols(A);
    for (int i=0; i<N; i++)
    {
        setLocalElement(A, row, i, val);
    }
}

/// Returns element (i,j) of matrix _A_.
/// First need to retrieve hiopMatrixDense from the abstract interface
real_type MatrixTestsDense::getLocalElement(
        const hiop::hiopMatrix* A,
        local_ordinal_type i,
        local_ordinal_type j)
{
    const hiop::hiopMatrixDense* amat = dynamic_cast<const hiop::hiopMatrixDense*>(A);
    return amat->local_data()[i][j];
}

/// Returns element _i_ of vector _x_.
/// First need to retrieve hiopVectorPar from the abstract interface
real_type MatrixTestsDense::getLocalElement(
        const hiop::hiopVector* x,
        local_ordinal_type i)
{
    const hiop::hiopVectorPar* xvec = dynamic_cast<const hiop::hiopVectorPar*>(x);
    return xvec->local_data_const()[i];
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

/// Returns size of local data array for vector _x_
int MatrixTestsDense::getLocalSize(const hiop::hiopVector* x)
{
    const hiop::hiopVectorPar* xvec = dynamic_cast<const hiop::hiopVectorPar*>(x);
    return static_cast<int>(xvec->get_local_size());
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

#ifdef HIOP_USE_MPI
    MPI_Allreduce(&failures, &fail, 1, MPI_INT, MPI_SUM, getMPIComm(A));
#else
    fail = failures;
#endif

    return (fail != 0);
}

[[nodiscard]]
int MatrixTestsDense::verifyAnswer(hiop::hiopMatrix* A, const double answer)
{
    const local_ordinal_type M = getNumLocRows(A);
    const local_ordinal_type N = getNumLocCols(A);
    int fail = 0;
    for (local_ordinal_type i=0; i<M; i++)
    {
        for (local_ordinal_type j=0; j<N; j++)
        {
            if (!isEqual(getLocalElement(A, i, j), answer))
            {
                std::cout << i << " " << j << " = " << getLocalElement(A, i, j) << " != " << answer << "\n";
                fail++;
            }
        }
    }
    return fail;
}

/*
 * Pass a function-like object to calculate the expected
 * answer dynamically, based on the row and column
 */
[[nodiscard]]
int MatrixTestsDense::verifyAnswer(
            hiop::hiopMatrix* A,
            std::function<real_type(local_ordinal_type, local_ordinal_type)> expect)
{
    const local_ordinal_type M = getNumLocRows(A);
    const local_ordinal_type N = getNumLocCols(A);
    int fail = 0;
    for (local_ordinal_type i=0; i<M; i++)
    {
        for (local_ordinal_type j=0; j<N; j++)
        {
            if (!isEqual(getLocalElement(A, i, j), expect(i, j)))
            {
                fail++;
            }
        }
    }

    return fail;
}

/// Checks if _local_ vector elements are set to `answer`.
[[nodiscard]]
int MatrixTestsDense::verifyAnswer(hiop::hiopVector* x, double answer)
{
    const local_ordinal_type N = getLocalSize(x);

    int local_fail = 0;
    for(local_ordinal_type i=0; i<N; ++i)
    {
        if(!isEqual(getLocalElement(x, i), answer))
        {
            ++local_fail;
        }
    }

    return local_fail;
}

[[nodiscard]]
int MatrixTestsDense::verifyAnswer(
    hiop::hiopVector* x,
    std::function<real_type(local_ordinal_type)> expect)
{
    const local_ordinal_type N = getLocalSize(x);

    int local_fail = 0;
    for (int i=0; i<N; i++)
    {
        if(!isEqual(getLocalElement(x, i), expect(i)))
        {
            ++local_fail;
        }
    }
    return local_fail;
}

bool MatrixTestsDense::globalToLocalMap(
    hiop::hiopMatrix* A,
    const global_ordinal_type row,
    const global_ordinal_type col,
    local_ordinal_type& local_row,
    local_ordinal_type& local_col)
{
#ifdef HIOP_USE_MPI
    int rank = 0;
    MPI_Comm comm = getMPIComm(A);
    MPI_Comm_rank(comm, &rank);
    const local_ordinal_type n_local = getNumLocCols(A);
    const global_ordinal_type local_col_start = n_local * rank;
    if (col >= local_col_start && col < local_col_start + n_local)
    {
        local_row = row;
        local_col = col % n_local;
        return true;
    }
    else
    {
        return false;
    }
#else
    local_row = row;
    local_col = col;
    return true;
#endif
}

// End helper methods

} // namespace hiop::tests
