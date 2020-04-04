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

    global_ordinal_type matrixNumRows(hiop::hiopMatrix& A, long long M)
    {
        const bool fail = A.m() == M ? 0 : 1;
        printMessage(fail, __func__, rank);
        return reduceReturn(fail, &A);
    }

    global_ordinal_type matrixNumCols(hiop::hiopMatrix& A, long long N)
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

protected:
    virtual void setElement(hiop::hiopMatrix* a, local_ordinal_type i, local_ordinal_type j, real_type val) = 0;
    virtual real_type getElement(hiop::hiopMatrix* a, local_ordinal_type i, local_ordinal_type j) = 0;
    virtual local_ordinal_type getNumLocRows(hiop::hiopMatrix* a) = 0;
    virtual local_ordinal_type getNumLocCols(hiop::hiopMatrix* a) = 0;
};

} // namespace hiopTest
