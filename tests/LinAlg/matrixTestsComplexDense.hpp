// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop.
// HiOp is released under the BSD 3-clause license
// (https://opensource.org/licenses/BSD-3-Clause). Please also read “Additional
// BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the disclaimer below. ii. Redistributions in
// binary form must reproduce the above copyright notice, this list of
// conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
// THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S.
// Department of Energy (DOE). This work was produced at Lawrence Livermore
// National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National
// Security, LLC nor any of their employees, makes any warranty, express or
// implied, or assumes any liability or responsibility for the accuracy,
// completeness, or usefulness of any information, apparatus, product, or
// process disclosed, or represents that its use would not infringe
// privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or
// services by trade name, trademark, manufacturer or otherwise does not
// necessarily constitute or imply its endorsement, recommendation, or favoring
// by the United States Government or Lawrence Livermore National Security, LLC.
// The views and opinions of authors expressed herein do not necessarily state
// or reflect those of the United States Government or Lawrence Livermore
// National Security, LLC, and shall not be used for advertising or product
// endorsement purposes.

/**
 * @file matrixTestsDense.hpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>,  PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 *
 */

#pragma once

#include <complex>

#include "matrixTestsDense.hpp"

namespace hiop
{
namespace tests
{

  class MatrixTestsComplexDense : public MatrixTestsDense
  {
  public:
    MatrixTestsComplexDense() {}
    virtual ~MatrixTestsComplexDense() {}

    // Start hiopMatrixComplexDense matrix tests
    virtual int matrixSetToZero(hiop::hiopMatrixComplexDense &A, const int rank)
    {
      A.setToZero();
      const int fail = verifyAnswer(&A, std::complex<real_type>{zero});
      printMessage(fail, __func__, rank);
      return reduceReturn(fail, &A);
    }

    virtual int matrixSetToConstant(hiop::hiopMatrixComplexDense &A,
                                    const int rank)
    {
      const local_ordinal_type M = getNumLocRows(&A);
      const local_ordinal_type N = getNumLocCols(&A);

      std::complex<real_type> aval{two};
      A.setToZero();
      A.setToConstant(aval);
      int fail = verifyAnswer(&A, aval);

      real_type aval1{two};
      A.setToZero();
      A.setToConstant(aval1);
      fail += verifyAnswer(&A, aval);

      printMessage(fail, __func__, rank);
      return reduceReturn(fail, &A);
    }

    int matrixCopyFrom(hiopMatrixComplexDense &dst,
                       hiopMatrixComplexDense &src,
                       const int rank)
    {
      assert(dst.n() == src.n()
             && "Did you pass in matrices of the same size?");
      assert(dst.m() == src.m()
             && "Did you pass in matrices of the same size?");
      assert(getNumLocRows(&dst) == getNumLocRows(&src)
             && "Did you pass in matrices of the same size?");
      assert(getNumLocCols(&dst) == getNumLocCols(&src)
             && "Did you pass in matrices of the same size?");
      std::complex<real_type> src_val{one};

      // Test copying src another matrix
      src.setToConstant(src_val);
      dst.setToZero();

      dst.copyFrom(src);
      int fail = verifyAnswer(&dst, src_val);

      // test copying src a raw buffer
      const size_t buf_len = getNumLocRows(&src) * getNumLocCols(&src);
      auto *src_buf = new std::complex<real_type>[buf_len];
      for (int i = 0; i < buf_len; i++)
        src_buf[i] = src_val;
      dst.setToZero();

      dst.copyFrom(src_buf);
      fail += verifyAnswer(&dst, src_val);
      delete[] src_buf;

      printMessage(fail, __func__, rank);
      return reduceReturn(fail, &dst);
    }

    /**
     * Tests function that copies rows from source to destination starting from
     * `dst_start_idx` in the same order.
     *
     */
    int matrixCopyRowsFrom(hiopMatrixComplexDense &dst,
                           hiopMatrixComplexDense &src,
                           const int rank)
    {
      assert(dst.n() == src.n());
      assert(dst.m() > src.m());
      assert(getNumLocCols(&dst) == getNumLocCols(&src));
      assert(getNumLocRows(&dst) > getNumLocRows(&src));
      std::complex<real_type> dst_val{one, one};
      std::complex<real_type> src_val{two, two};
      const local_ordinal_type dst_start_idx = dst.m() - src.m();
      local_ordinal_type num_rows_to_copy = src.m();
      const local_ordinal_type src_num_rows = src.m();

      // Test copying continuous rows from matrix
      dst.setToConstant(dst_val);
      src.setToConstant(src_val);

      dst.copyRowsFrom(src, num_rows_to_copy, dst_start_idx);

      int fail =
          verifyAnswer(&dst,
                       [=](local_ordinal_type i,
                           local_ordinal_type j) -> std::complex<real_type> {
                         (void)j;  // j is unused
                         const bool isRowCopiedOver =
                             (i >= dst_start_idx
                              && i < dst_start_idx + src_num_rows);
                         return isRowCopiedOver ? src_val : dst_val;
                       });

      printMessage(fail, __func__, rank);
      return reduceReturn(fail, &dst);
    }

    /**
     * @brief Tests both the real-only overload and the imaginary-defined
     * overload.
     */
    int matrixAddMatrixComplex(
        hiop::hiopMatrixComplexDense& A,
        hiop::hiopMatrixComplexDense& B,
        const int rank)
    {
      int                     fail{0};
      std::complex<real_type> alpha{half, half};
      std::complex<real_type> A_val{half, half};
      std::complex<real_type> B_val{one, one};

      // Test the real-only method first
      fail += MatrixTests::matrixAddMatrix(A, B, rank);

      // Then test the imaginary-defined method
      A.setToConstant(A_val);
      B.setToConstant(B_val);
      A.addMatrix(alpha, B);
      fail += verifyAnswer(&A, A_val + B_val * alpha);

      printMessage(fail, __func__, rank);
      return reduceReturn(fail, &A);
    }

    /**
     * @brief Tests method `addSparseMatrix' method. The assertions in th
     * method will fail if both matrices are not square, so both test matrices
     * must also be square.
     *
     * TODO: Clean up all the nasty inline crap in here. Cameron implemented
     * most of this in his branch, so as soon as that branch is pulled I'll
     * have access to those methods here.
     */
    int matrixAddSparseMatrix(
        hiop::hiopMatrixComplexDense& A,
        hiop::hiopMatrixComplexSparseTriplet& B,
        const int rank=0)
    {
      assert(A.m() == A.n());
      assert(B.m() == B.n());
      assert(A.m() == B.m());
      const local_ordinal_type M    = getNumLocRows(&A);
      const local_ordinal_type N    = getNumLocCols(&A);
      const int                nnz  = B.numberOfNonZeros();
      int                      fail = 0;
      std::complex<real_type>  B_val{one, one};
      std::complex<real_type>  A_val{two, two};
      std::complex<real_type>  alpha{half, half};

      A.setToConstant(A_val);
      B.setToConstant(B_val);
      A.addSparseMatrix(zero, B);
      fail += verifyAnswer(&A, A_val);

      A.setToConstant(A_val);
      B.setToZero();
      A.addSparseMatrix(alpha, B);
      fail += verifyAnswer(&A, A_val);

      A.setToConstant(A_val);
      B.setToConstant(B_val);

      // On master rank, set one index to be zero
      // but it has to already be a nonzero of the sparse matrix
      local_ordinal_type zeroRowIdx{-1};
      local_ordinal_type zeroColIdx{-1};
      auto irow = B.i_row();
      auto jcol = B.j_col();

      /// TODO: use cameron's helper methods for this sort of thing
      if (rank == 0)
      {
        auto* values = B.M();
        values[0] = zero;
        zeroRowIdx = irow[0];
        zeroColIdx = jcol[0];
      }

      A.addSparseMatrix(alpha, B);
      fail += verifyAnswer(&A,
        [=](local_ordinal_type i,
            local_ordinal_type j
            ) -> std::complex<real_type>
        {
          const bool isZerodIdx = rank == 0 && 
                                  i == zeroRowIdx && 
                                  j == zeroColIdx;
          /// TODO: fix this ugle crap with cameron's helper methods
          bool isNonZero{false};
          for(int ii=0; ii<nnz; ii++)
          {
            if (irow[ii] == i && jcol[ii] == j)
            {
              isNonZero = true;
              break;
            }
          }
          if (isZerodIdx || !isNonZero)
            return A_val;
          else
            return A_val + alpha * B_val;
        });

      printMessage(fail, __func__, rank);
      return reduceReturn(fail, &A);
    }

    int matrixAddSparseSymUpperTriangleToSymDenseMatrixUpperTriangle(
        hiop::hiopMatrixComplexDense& A,
        hiop::hiopMatrixComplexSparseTriplet& B,
        const int rank=0)
    {
      assert(A.m() == A.n());
      assert(B.m() == B.n());
      assert(A.m() == B.m());
      const local_ordinal_type M          = getNumLocRows(&A);
      const local_ordinal_type N          = getNumLocCols(&A);
      const local_ordinal_type M_sparse   = getNumLocRows(&B);
      const local_ordinal_type N_sparse   = getNumLocCols(&B);
      const int                nnz        = B.numberOfNonZeros();
      int                      fail       = 0;
      local_ordinal_type       zeroRowIdx = -1;
      local_ordinal_type       zeroColIdx = -1;
      auto                     irow       = B.i_row();
      auto                     jcol       = B.j_col();
      std::complex<real_type>  B_val{one, one};
      std::complex<real_type>  A_val{two, two};
      std::complex<real_type>  alpha{half, half};

      A.setToConstant(A_val);
      B.setToZero();
      A.addSparseSymUpperTriangleToSymDenseMatrixUpperTriangle(alpha, B);
      fail += verifyAnswer(&A, A_val);

      A.setToConstant(A_val);
      B.setToConstant(B_val);
      A.addSparseSymUpperTriangleToSymDenseMatrixUpperTriangle(zero, B);
      fail += verifyAnswer(&A, A_val);

      /// TODO: use cameron's helper methods for this sort of thing
      if (rank == 0)
      {
        auto* values = B.M();
        values[0] = zero;
        zeroRowIdx = irow[0];
        zeroColIdx = jcol[0];
      }

      A.setToConstant(A_val);
      B.setToConstant(B_val);
      A.addSparseSymUpperTriangleToSymDenseMatrixUpperTriangle(alpha, B);
      fail += verifyAnswer(&A,
        [=](local_ordinal_type row,
            local_ordinal_type col)
        {
          const bool isUpperTriangle = row <= col;
          const bool isZerodIdx      = rank == 0 && 
                                       row  == zeroRowIdx && 
                                       col  == zeroColIdx;

          bool isNonZero{false};
          for(int i=0; i<nnz; i++)
          {
            if (irow[i] == row && jcol[i] == col)
            {
              isNonZero = true;
            }
          }
          return isUpperTriangle && isNonZero && !isZerodIdx ?
                 A_val + alpha * B_val :
                 A_val;
        });

      printMessage(fail, __func__, rank);
      return reduceReturn(fail, &A);
    }
    // End hiopMatrixComplexDense matrix tests

  private:
    virtual void setLocalElement(hiop::hiopMatrix *a,
        local_ordinal_type i,
        local_ordinal_type j,
        real_type val) override;
    virtual void setLocalElement(hiop::hiopMatrix *a,
        local_ordinal_type i,
        local_ordinal_type j,
        std::complex<real_type> val);
    virtual void setLocalRow(hiop::hiopMatrixComplexDense *A,
        const local_ordinal_type row,
        const std::complex<real_type> val);
    virtual real_type getLocalElement(const hiop::hiopMatrix *a,
        local_ordinal_type i,
        local_ordinal_type j) override;
    virtual std::complex<real_type> getLocalElementComplex(
        const hiop::hiopMatrix* a, local_ordinal_type i, local_ordinal_type j);
    virtual real_type getLocalElement(const hiop::hiopVector *x,
        local_ordinal_type i);
    virtual local_ordinal_type getNumLocRows(hiop::hiopMatrix *a) override;
    virtual local_ordinal_type getNumLocCols(hiop::hiopMatrix *a) override;
    virtual local_ordinal_type getLocalSize(const hiop::hiopVector *x) override;
    virtual int verifyAnswer(hiop::hiopMatrix *A,
        std::complex<real_type> answer);
    virtual int verifyAnswer(
        hiop::hiopMatrix *A,
        std::function<std::complex<real_type>(local_ordinal_type,
          local_ordinal_type)> expect);
    virtual int verifyAnswer(hiop::hiopVector *x,
        real_type answer);
    virtual int verifyAnswer(
        hiop::hiopVector *x,
        std::function<real_type(local_ordinal_type)> expect);
    virtual bool reduceReturn(int failures, hiop::hiopMatrix *A) override;
    virtual bool globalToLocalMap(hiop::hiopMatrix *A,
        const global_ordinal_type row,
        const global_ordinal_type col,
        local_ordinal_type &local_row,
        local_ordinal_type &local_col) override;

#ifdef HIOP_USE_MPI
    MPI_Comm getMPIComm(hiop::hiopMatrix *A);
#endif
  };

}  // namespace tests
}  // namespace hiop
