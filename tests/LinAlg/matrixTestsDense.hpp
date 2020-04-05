#pragma once

#include "matrixTests.hpp"

namespace hiop::tests {

class MatrixTestsDense : public MatrixTests
{
public:
    MatrixTestsDense() {}
    virtual ~MatrixTestsDense(){}

private:
    virtual void setElement(hiop::hiopMatrix* a, int i, int j, double val);
    virtual double getElement(const hiop::hiopMatrix* a, int i, int j);
    virtual double getElementVec(const hiop::hiopVector* x, int i);
    virtual int getNumLocRows(hiop::hiopMatrix* a);
    virtual int getNumLocCols(hiop::hiopMatrix* a);
    virtual int getLocalSize(const hiop::hiopVector* x);
    virtual int verifyAnswer(hiop::hiopMatrix* A, double answer);
    virtual int verifyAnswerVec(hiop::hiopVector* x, double answer);
    virtual bool reduceReturn(int failures, hiop::hiopMatrix* A);

#ifdef HIOP_USE_MPI
    MPI_Comm getMPIComm(hiop::hiopMatrix* A);
#endif
};

} // namespace hiop::tests
