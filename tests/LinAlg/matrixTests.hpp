#pragma once

#include <iostream>
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

protected:
    virtual void setElement(hiop::hiopMatrix* a, int i, int j, double val) = 0;
    virtual double getElement(hiop::hiopMatrix* a, int i, int j) = 0;
    virtual int getNumLocRows(hiop::hiopMatrix* a) = 0;
    virtual int getNumLocCols(hiop::hiopMatrix* a) = 0;
    virtual int verifyAnswer(hiop::hiopMatrix* A, double answer) = 0;
    virtual bool reduceReturn(int failures, hiop::hiopMatrix* A) = 0;
};

} // namespace hiopTest
