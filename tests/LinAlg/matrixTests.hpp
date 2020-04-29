#pragma once

#include <iostream>
#include <cassert>
#include <hiopVector.hpp>
#include <hiopMatrix.hpp>
#include "testBase.hpp"

namespace hiop::tests {

class MatrixTests : public TestBase
{
public:
    MatrixTests() {}
    virtual ~MatrixTests(){}

    int matrixNumRows(hiop::hiopMatrix& A, global_ordinal_type M, const int rank)
    {
        const bool fail = A.m() == M ? 0 : 1;
        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    int matrixNumCols(hiop::hiopMatrix& A, global_ordinal_type N, const int rank)
    {
        const bool fail = A.n() == N ? 0 : 1;
        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    int matrixSetToZero(hiop::hiopMatrix& A, const int rank)
    {
        local_ordinal_type M = getNumLocRows(&A);
        local_ordinal_type N = getNumLocCols(&A);

        A.setToZero();

        int fail = 0;
        for(local_ordinal_type i=0; i<M; ++i)
            for(local_ordinal_type j=0; j<N; ++j)
                if(getLocalElement(&A,i,j) != 0)
                {
                    std::cerr << "Element (" << i << "," << j << ") not set to zero\n";
                    fail++;
                }

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    int matrixSetToConstant(hiop::hiopMatrix& A, const int rank)
    {
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type N = getNumLocCols(&A);
        int fail = 0;
        for (local_ordinal_type i=0; i<M; i++)
            for (local_ordinal_type j=0; j<N; j++)
                setLocalElement(&A, i, j, one);
        A.setToConstant(two);
        fail = verifyAnswer(&A, two);
        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    /*
     * y_{glob} \leftarrow \beta y_{glob} + \alpha A_{glob \times loc} x_{loc}
     */
    int matrixTimesVec(
            hiop::hiopMatrix& A,
            hiop::hiopVector& m_vec,
            hiop::hiopVector& n_vec,
            const int rank)
    {
        int fail = 0;
        A.setToConstant(one);
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type N = getNumLocCols(&A);
        const global_ordinal_type N_glob = n_vec.get_size();
        assert(getLocalSize(&m_vec) == M && "Did you pass in vectors of the correct sizes?");
        assert(getLocalSize(&n_vec) == N && "Did you pass in vectors of the correct sizes?");

        // First, check A_{MxN} \times x_N
        // beta = zero so y \leftarrow alpha * A * x
        m_vec.setToConstant(zero);
        n_vec.setToConstant(two);
        A.setToConstant(one);
        A.timesVec(zero, m_vec, one, n_vec);
        real_type expected = two * N_glob;
        fail += verifyAnswerVec(&m_vec, expected);

        // Now, check y \leftarrow beta * y + alpha * A * x
        m_vec.setToConstant(two);
        n_vec.setToConstant(two);
        A.setToConstant(one);
        A.timesVec(one, m_vec, two, n_vec);
        //          beta   * y    +  alpha * A   * x
        expected = (one    * two) + (two   * one * two * N_glob);
        //                                                ^^^
        // Sum over num global columns <-------------------+
        fail += verifyAnswerVec(&m_vec, expected);

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    /*
     * y_{loc} \leftarrow \beta y_{loc} + \alpha A_{glob \times loc}^T x_{glob}
     *
     * Notice that since A^T, x must not be distributed in this case, whereas
     * the plain `timesVec' nessecitated that x be distributed and y not be.
     */
    int matrixTransTimesVec(
            hiop::hiopMatrix& A,
            hiop::hiopVector& m_vec,
            hiop::hiopVector& n_vec,
            const int rank)
    {
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type N = getNumLocCols(&A);
        const global_ordinal_type N_glob = n_vec.get_size();
        assert(getLocalSize(&m_vec) == M && "Did you pass in vectors of the correct sizes?");
        assert(getLocalSize(&n_vec) == N && "Did you pass in vectors of the correct sizes?");
        int fail = 0;

        // First, test with \beta = 0
        A.setToConstant(one);
        m_vec.setToConstant(one);
        n_vec.setToConstant(zero);
        A.transTimesVec(zero, n_vec, two, m_vec);
        //                 0 * y + alpha * A^T   * 1
        real_type expected =          two   * one   * one * M;
        //                                              ^^^
        // Sum over num global rows <--------------------|
        fail += verifyAnswerVec(&n_vec, expected);

        // Now test with \beta != 0 \and y != 0
        A.setToConstant(one);
        m_vec.setToConstant(one);
        n_vec.setToConstant(one);
        A.transTimesVec(two, n_vec, two, m_vec);
        //          beta * y    + alpha * A^T   * X
        expected = (two  * one) + two   * one   * one * M;
        //                                             ^^^
        // Sum over num global rows <-------------------|
        fail += verifyAnswerVec(&n_vec, expected);

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    /**
     *  W = beta * W + alpha * this * X
     *
     * Shapes:
     *   A: MxK
     *   X: KxN
     *   W: MxN
     */
    int matrixTimesMat(
            hiop::hiopMatrix& A,
            hiop::hiopMatrix& X,
            hiop::hiopMatrix& W,
            const int rank)
    {
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type K_loc = getNumLocCols(&A);
        const global_ordinal_type K_glob = A.n();
        const local_ordinal_type N_loc = getNumLocCols(&X);
        const global_ordinal_type N_glob = X.n();
        assert(K_glob == getNumLocRows(&X)  && "Matrices have mismatched shapes");
        assert(M == getNumLocRows(&W)       && "Matrices have mismatched shapes");
        assert(N_loc == getNumLocCols(&W)   && "Matrices have mismatched shapes");
        assert(N_glob == W.n()              && "Matrices have mismatched shapes");
        int fail = 0;

        A.setToConstant(one);
        W.setToConstant(one);
        X.setToConstant(one);

        // Beta = 0 to just test matmul portion
        // this fails
        A.timesMat(one, W, one, X);

        /*
         * This is commented out until we successfully
         * test the previous lines
         *
        //     W        = 0 * W + A   * X
        real_type expected =         one * one * N_glob;
        fail += verifyAnswer(&W, expected);

        A.setToConstant(one);
        W.setToConstant(two);
        X.setToConstant(half);
        A.timesMat(one, W, one, X);

        //     W = 0   * W   + \sum_0^{N_glob} A   * X
        expected = one * two + N_glob        * one * half;
        fail += verifyAnswer(&W, expected);

        */
        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    /*
     *  W = beta * W + alpha * this^T * X
     *
     *  A: mxn
     *  W: mxk
     *  X: nxk
     *
     */
    int matrixTransTimesMat(
            hiop::hiopMatrix& A,
            hiop::hiopMatrix& W,
            hiop::hiopMatrix& X,
            const int rank)
    {
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type N_loc = getNumLocCols(&A);
        const global_ordinal_type N_glob = A.n();
        assert(M == getNumLocRows(&W) && "Matrices have mismatched shapes");
        assert(N_loc == getNumLocCols(&W) && "Matrices have mismatched shapes");
        assert(N_loc == getNumLocCols(&X) && "Matrices have mismatched shapes");
        assert(N_glob == getNumLocRows(&X) && "Matrices have mismatched shapes");
        int fail = 0;

        A.setToConstant(one);
        W.setToConstant(one);
        X.setToConstant(one);

        // Beta = 0 to just test matmul portion
        // this fails
        // A.timesMat(zero, W, one, X);

        //        W        = 0 * W + A   * X
        // real_type expected =         one * one * N_glob;
        // fail += verifyAnswer(&W, expected);

        printMessage(SKIP_TEST, __func__, rank);
        return reduceReturn(fail, &A);
    }

    int matrixTimesMatTrans(
            hiop::hiopMatrix& A,
            hiop::hiopMatrix& W,
            hiop::hiopMatrix& X,
            const int rank)
    {
        printMessage(SKIP_TEST, __func__, rank);
        return 0;
    }

    /*
     * this += alpha * diag
     */
    int matrixAddDiagonal(
            hiop::hiopMatrix& A,
            hiop::hiopVector& x,
            const int rank)
    {
        int fail = 0;
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type N = getNumLocCols(&A);
        assert(x.get_size()==A.n());
        assert(x.get_size()==A.m());
        assert(N == getLocalSize(&x));
        assert(M == getLocalSize(&x));
        constexpr real_type alpha = two;

        A.setToConstant(one);
        x.setToConstant(two);
        A.addDiagonal(alpha, x);
        real_type expected = one + (alpha * two);
        real_type tmp = 0.0;
        for (local_ordinal_type i=0; i<M; i++)
            for (local_ordinal_type j=0; j<N; j++)
                if ((tmp=getLocalElement(&A, i, j)) != expected)
                    if (!(i != j && tmp == one))
                        fail++;

        A.setToConstant(one);
        A.addDiagonal(alpha);
        expected = one + alpha;
        for (local_ordinal_type i=0; i<M; i++)
            for (local_ordinal_type j=0; j<N; j++)
                if ((tmp=getLocalElement(&A, i, j)) != expected)
                    if (!(i != j && tmp == one))
                        fail++;

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    /*
     * this += alpha * subdiag
     */
    int matrixAddSubDiagonal(
            hiop::hiopMatrix& A,
            hiop::hiopVector& x,
            const int rank)
    {
        int fail = 0;
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type N = getNumLocCols(&A);
        assert(N == getLocalSize(&x));
        assert(M == getLocalSize(&x));
        assert(M == A.n());
        assert(A.n() == x.get_size());
        assert(A.m() == x.get_size());

        A.setToConstant(one);
        x.setToConstant(two);
        A.addSubDiagonal(two, 0, x);
        for (global_ordinal_type i=0; i<M; i++)
        {
            if (getLocalElement(&A, i, i) != (one + two * two))
                fail++;
        }

        A.setToConstant(one);
        x.setToConstant(two);
        A.addSubDiagonal(0, two, x, 0, N);
        for (global_ordinal_type i=0; i<M; i++)
        {
            if (getLocalElement(&A, i, i) != (one + two * two))
                fail++;
        }

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    /* 
     * W = beta*W + alpha*this*X
     * For A with shape M x N,
     * X must have shape N x L, and
     * W must have shape M x L
     *
     * Shapes:
     *   A: MxK
     *   X: KxN
     *   W: MxN
     */
    int matrixTimesMat(
            hiop::hiopMatrix& A,
            hiop::hiopMatrix& X,
            hiop::hiopMatrix& W,
            const int rank)
    {
        const int M = getNumLocRows(&A);
        const int K_loc = getNumLocCols(&A);
        const int K_glob = A.n();
        const int N_loc = getNumLocCols(&X);
        const int N_glob = X.n();
        assert(K_glob == getNumLocRows(&X)  && "Matrices have mismatched shapes");
        assert(M == getNumLocRows(&W)       && "Matrices have mismatched shapes");
        assert(N_loc == getNumLocCols(&W)   && "Matrices have mismatched shapes");
        assert(N_glob == W.n()              && "Matrices have mismatched shapes");
        int fail = 0;

        A.setToConstant(one);
        W.setToConstant(one);
        X.setToConstant(one);

        // Beta = 0 to just test matmul portion
        // this fails
        A.timesMat(one, W, one, X);

        /*
        //     W        = 0 * W + A   * X
        real_type expected =         one * one * N_glob;
        fail += verifyAnswer(&W, expected);

        A.setToConstant(one);
        W.setToConstant(two);
        X.setToConstant(half);
        A.timesMat(one, W, one, X);

        //     W = 0   * W   + \sum_0^{N_glob} A   * X
        expected = one * two + N_glob        * one * half;
        fail += verifyAnswer(&W, expected);

        */
        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    /*
     *  W = beta * W + alpha * this^T * X
     *
     *  A: mxn
     *  W: mxk
     *  X: nxk
     *
     */
    int matrixTransTimesMat(
            hiop::hiopMatrix& A,
            hiop::hiopMatrix& W,
            hiop::hiopMatrix& X,
            const int rank)
    {
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type N_loc = getNumLocCols(&A);
        const global_ordinal_type N_glob = A.n();
        assert(M == getNumLocRows(&W) && "Matrices have mismatched shapes");
        assert(N_loc == getNumLocCols(&W) && "Matrices have mismatched shapes");
        assert(N_loc == getNumLocCols(&X) && "Matrices have mismatched shapes");
        assert(N_glob == getNumLocRows(&X) && "Matrices have mismatched shapes");
        int fail = 0;

        A.setToConstant(one);
        W.setToConstant(one);
        X.setToConstant(one);

        // Beta = 0 to just test matmul portion
        // this fails
        // A.timesMat(zero, W, one, X);

        //        W        = 0 * W + A   * X
        // real_type expected =         one * one * N_glob;
        // fail += verifyAnswer(&W, expected);

        printMessage(SKIP_TEST, __func__, rank);
        return reduceReturn(fail, &A);
    }

    int matrixTimesMatTrans(
            hiop::hiopMatrix& A,
            hiop::hiopMatrix& W,
            hiop::hiopMatrix& X,
            const int rank)
    {
        printMessage(SKIP_TEST, __func__, rank);
        return 0;
    }

    /*
     * this += alpha * diag
     */
    int matrixAddDiagonal(
            hiop::hiopMatrix& A,
            hiop::hiopVector& x,
            const int rank)
    {
        int fail = 0;
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type N = getNumLocCols(&A);
        assert(N == getLocalSize(&x));
        assert(M == getLocalSize(&x));
        assert(M == A.n());
        assert(A.n() == x.get_size());
        assert(A.m() == x.get_size());
        constexpr real_type alpha = two;

        A.setToConstant(one);
        x.setToConstant(two);
        A.addDiagonal(alpha, x);
        real_type expected = one + (alpha * two);
        for (local_ordinal_type i=0; i<M; i++)
            if (getLocalElement(&A, i, i) != expected)
                fail++;

        A.setToConstant(one);
        A.addDiagonal(alpha);
        expected = one + alpha;
        for (local_ordinal_type i=0; i<M; i++)
            if (getLocalElement(&A, i, i) != expected)
                fail++;

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    /*
     * this += alpha * subdiag
     */
    int matrixAddSubDiagonal(
            hiop::hiopMatrix& A,
            hiop::hiopVector& x,
            const int rank)
    {
        int fail = 0;
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type N = getNumLocCols(&A);
        assert(N == getLocalSize(&x));
        assert(M == getLocalSize(&x));
        assert(M == A.n());
        assert(A.n() == x.get_size());
        assert(A.m() == x.get_size());

        A.setToConstant(one);
        x.setToConstant(two);
        A.addSubDiagonal(two, 0, x);
        for (global_ordinal_type i=0; i<M; i++)
        {
            if (getLocalElement(&A, i, i) != (one + two * two))
                fail++;
        }

        A.setToConstant(one);
        x.setToConstant(two);
        A.addSubDiagonal(0, two, x, 0, N);
        for (global_ordinal_type i=0; i<M; i++)
        {
            if (getLocalElement(&A, i, i) != (one + two * two))
                fail++;
        }

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }
    
    /*
     * this += alpha * B
     */
    int matrixAddMatrix(
            hiop::hiopMatrix& A,
            hiop::hiopMatrix& B,
            const int rank)
    {
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type N = getNumLocCols(&A);
        assert(M == getNumLocRows(&B));
        assert(N == getNumLocCols(&B));

        A.setToConstant(one);
        B.setToConstant(one);
        A.addMatrix(zero, B);
        int fail = verifyAnswer(&A, one);

        A.setToConstant(one);
        B.setToConstant(one);
        A.addMatrix(one, B);
        fail += verifyAnswer(&A, two);

        A.setToConstant(one);
        B.setToConstant(one);
        A.addMatrix(two, B);
        fail += verifyAnswer(&A, one + two);

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    /*
     * Block of W += alpha*A
     *
     * Precondition: W is square
     */
    int matrixAddToSymDenseMatrixUpperTriangle(
            hiop::hiopMatrix& _W,
            hiop::hiopMatrix& A,
            const int rank)
    {
        // This method only takes hiopMatrixDense
        auto W = dynamic_cast<hiop::hiopMatrixDense*>(&_W);
        const local_ordinal_type M = getNumLocRows(W);
        const local_ordinal_type N_loc = getNumLocCols(W);
        const local_ordinal_type A_M = getNumLocRows(&A);
        const local_ordinal_type A_N_loc = getNumLocCols(&A);
        assert(W->m() == W->n());
        assert(M >= getNumLocRows(&A));
        assert(W->n() >= A.n());

        const local_ordinal_type start_idx_row = 0;
        const local_ordinal_type start_idx_col = W->n() - A.n();
        int fail = 0;

        // Check with alpha=1 (only the matrix addition)
        A.setToConstant(one);
        W->setToConstant(one);
        A.addToSymDenseMatrixUpperTriangle(start_idx_row, start_idx_col, one, *W);
        real_type expected = two;
        for (int i=0; i<A_M; i++)
            for (int j=0; j<A_N_loc; j++)
                if (!isEqual(getLocalElement(W, start_idx_row+i, start_idx_col+j), expected))
                    fail++;

        // Check with non-1 alpha
        A.setToConstant(one);
        W->setToConstant(one);
        A.addToSymDenseMatrixUpperTriangle(start_idx_row, start_idx_col, two, *W);
        expected = two * one + one;
        for (int i=0; i<A_M; i++)
            for (int j=0; j<A_N_loc; j++)
                if (!isEqual(getLocalElement(W, start_idx_row+i, start_idx_col+j), expected))
                    fail++;

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    int matrixAssertSymmetry(
            hiop::hiopMatrix& A,
            const int rank)
    {
        assert(A.m() == A.n());
        A.setToConstant(one);
        int fail = !A.assertSymmetry(eps);
        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

protected:
    virtual void setLocalElement(
            hiop::hiopMatrix* a,
            local_ordinal_type i,
            local_ordinal_type j,
            real_type val) = 0;
    virtual real_type getLocalElement(
            const hiop::hiopMatrix* a,
            local_ordinal_type i,
            local_ordinal_type j) = 0;
    virtual real_type getLocalElementVec(
            const hiop::hiopVector* x,
            local_ordinal_type i) = 0;
    virtual local_ordinal_type getNumLocRows(hiop::hiopMatrix* a) = 0;
    virtual local_ordinal_type getNumLocCols(hiop::hiopMatrix* a) = 0;
    virtual int getLocalSize(const hiop::hiopVector* x) = 0;
    virtual int verifyAnswer(hiop::hiopMatrix* A, double answer) = 0;
    virtual int verifyAnswerVec(hiop::hiopVector* x, double answer) = 0;
    virtual bool reduceReturn(int failures, hiop::hiopMatrix* A) = 0;
};

} // namespace hiopTest
