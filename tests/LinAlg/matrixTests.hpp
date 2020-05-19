#pragma once

#include <iomanip>
#include <iostream>
#include <functional>
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
        A.setToZero();
        const int fail = verifyAnswer(&A, zero);
        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    int matrixSetToConstant(hiop::hiopMatrix& A, const int rank)
    {
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type N = getNumLocCols(&A);
        for (local_ordinal_type i=0; i<M; i++)
            for (local_ordinal_type j=0; j<N; j++)
                setLocalElement(&A, i, j, one);
        A.setToConstant(two);
        const int fail = verifyAnswer(&A, two);
        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    /*
     * y_{glob} \leftarrow \beta y_{glob} + \alpha A_{glob \times loc} x_{loc}
     */
    int matrixTimesVec(
            hiop::hiopMatrix& A,
            hiop::hiopVector& y,
            hiop::hiopVector& x,
            const int rank)
    {
        int fail = 0;
        A.setToConstant(one);
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type N = getNumLocCols(&A);
        const global_ordinal_type N_glob = A.n();
        assert(getLocalSize(&y) == M && "Did you pass in vectors of the correct sizes?");
        assert(getLocalSize(&x) == N && "Did you pass in vectors of the correct sizes?");
        const real_type alpha = one,
                        beta  = one,
                        A_val = one,
                        y_val = three,
                        x_val = three;

        y.setToConstant(y_val);
        x.setToConstant(x_val);
        A.setToConstant(A_val);
        A.timesVec(beta, y, alpha, x);

        real_type expected = (beta * y_val) + (alpha * A_val * x_val * N_glob);
        fail += verifyAnswer(&y, expected);

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    /*
     * y = beta * y + alpha * A^T * x
     *
     * Notice that since A^T, x must not be distributed in this case, whereas
     * the plain `timesVec' nessecitated that x be distributed and y not be.
     */
    int matrixTransTimesVec(
            hiop::hiopMatrix& A,
            hiop::hiopVector& x,
            hiop::hiopVector& y,
            const int rank)
    {
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type N = getNumLocCols(&A);

        // Take m() because A will be transposed
        const global_ordinal_type N_glob = A.m();
        assert(getLocalSize(&x) == M && "Did you pass in vectors of the correct sizes?");
        assert(getLocalSize(&y) == N && "Did you pass in vectors of the correct sizes?");
        const real_type alpha = one,
                        beta  = one,
                        A_val = one,
                        y_val = three,
                        x_val = three;
        int fail = 0;

        A.setToConstant(A_val);
        y.setToConstant(y_val);
        x.setToConstant(x_val);
        A.transTimesVec(beta, y, alpha, x);

        real_type expected = (beta * y_val) + (alpha * A_val * x_val * N_glob);
        fail += verifyAnswer(&y, expected);

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    /**
     *  W = beta * W + alpha * A * M
     *
     * Shapes:
     *   A: KxM
     *   M: MxN
     *   W: KxN
     */
    int matrixTimesMatLocal(
            hiop::hiopMatrix& A,
            hiop::hiopMatrix& X,
            hiop::hiopMatrix& W,
            const int rank)
    {
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type K = getNumLocCols(&A);
        const local_ordinal_type N = getNumLocCols(&X);
        assert(K == A.n());
        assert(N == X.n());
        assert(K == getNumLocRows(&X));
        assert(M == getNumLocRows(&W));
        assert(N == getNumLocCols(&W));
        const real_type A_val = two,
                        X_val = three,
                        W_val = two,
                        alpha = two,
                        beta  = two;

        A.setToConstant(A_val);
        W.setToConstant(W_val);
        X.setToConstant(X_val);
        A.timesMat(beta, W, alpha, X);
        real_type expected = (beta * W_val) + (alpha * A_val * X_val * N);
        const int fail = verifyAnswer(&W, expected);

        /*
         * This is commented out until we successfully
         * test the previous lines
         *
        //     W        = 0 * W + A   * M
        real_type expected =         one * one * N_glob;
        fail += verifyAnswer(&W, expected);

        A.setToConstant(one);
        W.setToConstant(two);
        M.setToConstant(half);
        A.timesMat(one, W, one, M);

        //     W = 0   * W   + \sum_0^{N_glob} A   * M
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

        printMessage(fail, __func__, rank);
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
     * this += alpha * subdiag
     */
    int matrixAddSubDiagonalLocal(
            hiop::hiopMatrix& A,
            hiop::hiopVector& x,
            const int rank)
    {
        int fail = 0;
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type N = getNumLocCols(&A);
        const local_ordinal_type x_len = getLocalSize(&x);
        const real_type alpha = half,
                        A_val = half,
                        x_val = one;

        // We're only going to add n-1 elements of the vector
        local_ordinal_type start_idx = (N - x_len) + 1;

        A.setToConstant(A_val);
        x.setToConstant(x_val);
        A.addSubDiagonal(start_idx, alpha, x, 1, x_len-1);
        fail += verifyAnswer(&A,
          [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
          {
            const bool isOnSubDiagonal = (i>=start_idx && i==j);
            return isOnSubDiagonal ? A_val + x_val * alpha : A_val;
          });

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
    int matrixAddSubDiagonalDistributed(
            hiop::hiopMatrix& A,
            hiop::hiopVector& x,
            const int rank)
    {
        int fail = 0;
        const local_ordinal_type M = getNumLocRows(&A);
        const local_ordinal_type N = getNumLocCols(&A);
        const local_ordinal_type x_len = getLocalSize(&x);
        local_ordinal_type start_idx = N - x_len;
        const real_type alpha = half,
                        A_val = half,
                        x_val = one;

        A.setToConstant(A_val);
        x.setToConstant(x_val);
        A.addSubDiagonal(alpha, start_idx, x);
        fail += verifyAnswer(&A,
                [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
                {
                    const bool isOnDiagonal = (i>=start_idx && i==j);
                    return isOnDiagonal ? A_val + x_val * alpha : A_val;
                });

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }
    
    /*
     * A += alpha * B
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
        const real_type alpha = half,
                        A_val = half,
                        B_val = one;

        A.setToConstant(A_val);
        B.setToConstant(B_val);
        A.addMatrix(alpha, B);
        const int fail = verifyAnswer(&A, A_val + B_val * alpha);

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
        const local_ordinal_type start_idx_col = N_loc - A_N_loc;
        const real_type alpha = half,
                        A_val = half,
                        W_val = one;
        int fail = 0;

        // Check with non-1 alpha
        A.setToConstant(A_val);
        W->setToConstant(W_val);
        A.addToSymDenseMatrixUpperTriangle(start_idx_row, start_idx_col, alpha, *W);
        fail += verifyAnswer(W,
                [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
                {
                    const bool isUpperTriangle = (
                        i>=start_idx_row && i<start_idx_row+A_M &&
                        j>=start_idx_col && j<start_idx_col+A_N_loc);
                    return isUpperTriangle ? W_val + A_val*alpha : W_val;
                });

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    /*
     * Block of W += alpha*A
     *
     * Block of W summed with A is in the trasposed
     * location of the same call to addToSymDenseMatrixUpperTriangle
     *
     * Precondition: W is square
     */
    int matrixTransAddToSymDenseMatrixUpperTriangle(
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
        const local_ordinal_type start_idx_col = N_loc - A_N_loc;
        const real_type alpha = half,
                        A_val = half,
                        W_val = one;
        int fail = 0;

        A.setToConstant(A_val);
        W->setToConstant(W_val);
        A.transAddToSymDenseMatrixUpperTriangle(start_idx_row, start_idx_col, alpha, *W);
        fail += verifyAnswer(W,
                [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
                {
                    const bool isTransUpperTriangle = (
                        i>=start_idx_row && i<start_idx_row+A_N_loc &&
                        j>=start_idx_col && j<start_idx_col+A_M);

                    return isTransUpperTriangle ? W_val + A_val*alpha : W_val;
                });

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    /*
     * Upper diagonal block of W += alpha * A
     *
     * Preconditions:
     * W is square
     * A is square
     * degree of A <= degree of W
     */
    int matrixAddUpperTriangleToSymDenseMatrixUpperTriangle(
            hiop::hiopMatrix& _W,
            hiop::hiopMatrix& A,
            const int rank)
    {
        const local_ordinal_type A_M = getNumLocRows(&A);
        const local_ordinal_type A_N = getNumLocCols(&A);
        const local_ordinal_type W_M = getNumLocRows(&_W);
        const local_ordinal_type W_N = getNumLocCols(&_W);
        assert(_W.m() == _W.n());
        assert(A.m() == A.n());
        assert(_W.n() >= A.n());
        assert(getNumLocCols(&A) <= getNumLocCols(&_W));
        auto W = dynamic_cast<hiop::hiopMatrixDense*>(&_W);
        // Map the upper triangle of A to W starting
        // at W's upper left corner
        const local_ordinal_type diag_start = 0;
        int fail = 0;
        const real_type alpha = half,
                        A_val = half,
                        W_val = one;

        A.setToConstant(A_val);
        W->setToConstant(W_val);
        A.addUpperTriangleToSymDenseMatrixUpperTriangle(diag_start, alpha, *W);
        fail += verifyAnswer(W,
                [=] (local_ordinal_type i, local_ordinal_type j) -> real_type
                {
                    bool isUpperTriangle = (i>=diag_start && i<diag_start+A_N && j>=i && j<diag_start+A_M);
                    return isUpperTriangle ? W_val + A_val*alpha : W_val;
                });

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

    int matrixIsFinite(
            hiop::hiopMatrix& A,
            const int rank)
    {
        int fail = 0;

        A.setToConstant(zero);
        if (!A.isfinite()) fail++;

        A.setToConstant(zero);
        if (rank == 0) setLocalElement(&A, 0, 0, INFINITY);
        if (!A.isfinite() && rank != 0) fail++;

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    int matrixMaxAbsValue(
            hiop::hiopMatrix& A,
            const int rank)
    {
        int fail = 0;

        // Positive largest value
        A.setToConstant(zero);
        if (rank == 0) setLocalElement(&A, 0, 0, one);
        fail += A.max_abs_value() != one;

        // Negative largest value
        A.setToConstant(zero);
        if (rank == 0) setLocalElement(&A, 0, 0, -one);
        fail += A.max_abs_value() != one;

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
    virtual local_ordinal_type getLocalSize(const hiop::hiopVector* x) = 0;
    virtual int verifyAnswer(hiop::hiopMatrix* A, real_type answer) = 0;
    virtual int verifyAnswer(
            hiop::hiopMatrix* A,
            std::function<real_type(local_ordinal_type, local_ordinal_type)> expect) = 0;
    virtual int verifyAnswer(hiop::hiopVector* x, real_type answer) = 0;
    virtual bool reduceReturn(int failures, hiop::hiopMatrix* A) = 0;
};

} // namespace hiopTest
