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

    int matrixNumRows(hiop::hiopMatrix& A, long long M, const int rank)
    {
        const bool fail = A.m() == M ? 0 : 1;
        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    int matrixNumCols(hiop::hiopMatrix& A, long long N, const int rank)
    {
        const bool fail = A.n() == N ? 0 : 1;
        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    int matrixSetToZero(hiop::hiopMatrix& A, const int rank)
    {
        int M = getNumLocRows(&A);
        int N = getNumLocCols(&A);

        A.setToZero();

        int fail = 0;
        for(int i=0; i<M; ++i)
            for(int j=0; j<N; ++j)
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
        assert(getLocalSize(&m_vec) == M && "Did you correctly pass in vectors of the correct size?");
        assert(getLocalSize(&n_vec) == N && "Did you correctly pass in vectors of the correct size?");

        // First, check A_{MxN} \times x_N
        // beta = zero so y \leftarrow alpha * A * x
        m_vec.setToConstant(zero);
        n_vec.setToConstant(two);
        A.setToConstant(one);
        A.timesVec(zero, m_vec, one, n_vec);
        double expected = two * N_glob;
        for (int i=0; i<M; i++)
        {
            double actual = getElementVec(&m_vec, i);
            if (!isEqual(actual, expected))
            {
                fail++;
                std::cerr << RED << "---- Rank " << rank
                    << " got " << actual
                    << " expected " << expected
                    << CLEAR << "\n";
            }
        }

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
        // Sum over num local columns <--------------------|
        for (int i=0; i<M; i++)
        {
            double actual = getElementVec(&m_vec, i);
            if (!isEqual(actual, expected))
            {
                fail++;
                std::cerr << RED << "---- Rank " << rank
                    << " got " << actual
                    << " expected " << expected
                    << CLEAR << "\n";
            }
        }

        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

protected:
    virtual void setElement(hiop::hiopMatrix* a, int i, int j, double val) = 0;
    virtual double getElement(const hiop::hiopMatrix* a, int i, int j) = 0;
    virtual double getElementVec(const hiop::hiopVector* x, int i) = 0;
    virtual int getNumLocRows(hiop::hiopMatrix* a) = 0;
    virtual int getNumLocCols(hiop::hiopMatrix* a) = 0;
    virtual int getLocalSize(const hiop::hiopVector* x) = 0;
    virtual int verifyAnswer(hiop::hiopMatrix* A, double answer) = 0;
    virtual bool reduceReturn(int failures, hiop::hiopMatrix* A) = 0;
};

} // namespace hiopTest
