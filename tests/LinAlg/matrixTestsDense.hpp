#pragma once

#include "matrixTests.hpp"

namespace hiop::tests {

class MatrixTestsDense : public MatrixTests
{
public:
    MatrixTestsDense() {}
    virtual ~MatrixTestsDense(){}

private:
    virtual void setElement(hiop::hiopMatrix* a, local_ordinal_type i, local_ordinal_type j, real_type val);
    virtual real_type getElement(hiop::hiopMatrix* a, local_ordinal_type i, local_ordinal_type j);
    virtual real_type getElementVec(const hiop::hiopVector* x, local_ordinal_type i);
    virtual local_ordinal_type getNumLocRows(hiop::hiopMatrix* a);
    virtual local_ordinal_type getNumLocCols(hiop::hiopMatrix* a);
    virtual local_ordinal_type getLocalSize(const hiop::hiopVector* x);
    virtual int verifyAnswer(hiop::hiopMatrix* A, real_type answer);
    virtual int verifyAnswerVec(hiop::hiopVector* x, real_type answer);
    virtual bool reduceReturn(int failures, hiop::hiopMatrix* A);

#ifdef HIOP_USE_MPI
    MPI_Comm getMPIComm(hiop::hiopMatrix* A);
#endif
};

} // namespace hiop::tests
