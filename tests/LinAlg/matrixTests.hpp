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

    int matrixNumRows(const hiop::hiopMatrix& A, global_ordinal_type M, const int rank)
    {
        const bool fail = A.m() == M ? 0 : 1;
        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    int matrixNumCols(const hiop::hiopMatrix& A, global_ordinal_type N, const int rank)
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
                if(getElement(&A,i,j) != 0)
                {
                    std::cerr << "Element (" << i << "," << j << ") not set to zero\n";
                    fail++;
                }

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    int matrixSetToConstant(hiop::hiopMatrix& A, const int rank)
    {
        const int M = getNumLocRows(&A);
        const int N = getNumLocCols(&A);
        int fail = 0;
        for (int i=0; i<M; i++)
            for (int j=0; j<N; j++)
                setElement(&A, i, j, one);
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
        const int M = getNumLocRows(&A);
        const int N = getNumLocCols(&A);
        const int N_glob = n_vec.get_size();
        assert(getLocalSize(&m_vec) == M && "Did you pass in vectors of the correct sizes?");
        assert(getLocalSize(&n_vec) == N && "Did you pass in vectors of the correct sizes?");

        // First, check A_{MxN} \times x_N
        // beta = zero so y \leftarrow alpha * A * x
        m_vec.setToConstant(zero);
        n_vec.setToConstant(two);
        A.setToConstant(one);
        A.timesVec(zero, m_vec, one, n_vec);
        double expected = two * N_glob;
        fail += verifyAnswerVec(&m_vec, expected);

        // Now, check y \leftarrow beta * y + alpha * A * x
        m_vec.setToConstant(two);
        n_vec.setToConstant(two);
        A.setToConstant(one);
        constexpr double beta = one;
        constexpr double alpha = two;
        A.timesVec(beta, m_vec, alpha, n_vec);
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
        const int M = getNumLocRows(&A);
        const int N = getNumLocCols(&A);
        const int N_glob = n_vec.get_size();
        assert(getLocalSize(&m_vec) == M && "Did you pass in vectors of the correct sizes?");
        assert(getLocalSize(&n_vec) == N && "Did you pass in vectors of the correct sizes?");
        int fail = 0;

        // First, test with \beta = 0
        A.setToConstant(one);
        m_vec.setToConstant(one);
        n_vec.setToConstant(zero);
        A.transTimesVec(zero, n_vec, two, m_vec);
        //                 0 * y + alpha * A^T   * 1
        double expected =          two   * one   * one * M;
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

    /* 
     * W = beta*W + alpha*this*X
     * For A with shape M x N,
     * X must have shape N x L, and
     * W must have shape M x L
     */
    int matrixTimesMat(
            hiop::hiopMatrix& A,
            hiop::hiopMatrix& W,
            hiop::hiopMatrix& X,
            const int rank)
    {
        const int M = getNumLocRows(&A);
        const int N = getNumLocCols(&A);
        const int L = getNumLocCols(&X);
        // W must have same shape as A \times X
        /*
        assert(M == getNumLocRows(&W) && "Matrices have mismatched shapes");
        assert(L == getNumLocCols(&W) && "Matrices have mismatched shapes");
        assert(N == getNumLocRows(&X) && "Matrices have mismatched shapes");
        */
        int fail = 0;
        printMessage(SKIP_TEST, __func__, rank);
        return 0;
    }

    int matrixTransTimesMat(
            hiop::hiopMatrix& A,
            hiop::hiopMatrix& W,
            hiop::hiopMatrix& X,
            const int rank)
    {
        printMessage(SKIP_TEST, __func__, rank);
        return 0;
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

protected:
    virtual void setElement(
            hiop::hiopMatrix* a, local_ordinal_type i,
            local_ordinal_type j, real_type val) = 0;
    virtual real_type getElement(hiop::hiopMatrix* a,
            local_ordinal_type i, local_ordinal_type j) = 0;
    virtual real_type getElementVec(const hiop::hiopVector* x, local_ordinal_type i) = 0;
    virtual local_ordinal_type getNumLocRows(hiop::hiopMatrix* a) = 0;
    virtual local_ordinal_type getNumLocCols(hiop::hiopMatrix* a) = 0;
    virtual local_ordinal_type getLocalSize(const hiop::hiopVector* x) = 0;
    virtual int verifyAnswer(hiop::hiopMatrix* A, real_type answer) = 0;
    virtual int verifyAnswerVec(hiop::hiopVector* x, real_type answer) = 0;
    virtual bool reduceReturn(int failures, hiop::hiopMatrix* A) = 0;
};

} // namespace hiopTest
