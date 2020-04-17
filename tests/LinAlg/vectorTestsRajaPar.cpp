#include <hiopVectorRajaPar.hpp>
#include "vectorTestsRajaPar.hpp"

namespace hiop::tests {

/// Method to set vector _x_ element _i_ to _value_.
/// First need to retrieve hiopVectorRajaPar from the abstract interface
void VectorTestsRajaPar::setElement(hiop::hiopVector* x, local_ordinal_type i, real_type value)
{
    hiop::hiopVectorRajaPar* xvec = dynamic_cast<hiop::hiopVectorRajaPar*>(x);
    xvec->copyFromDev();
    real_type* xdat = xvec->local_data();
    xdat[i] = value;
    xvec->copyToDev();
}

/// Returns element _i_ of vector _x_.
/// First need to retrieve hiopVectorRajaPar from the abstract interface
real_type VectorTestsRajaPar::getElement(const hiop::hiopVector* x, local_ordinal_type i)
{
    const hiop::hiopVectorRajaPar* xv = dynamic_cast<const hiop::hiopVectorRajaPar*>(x);
    hiop::hiopVectorRajaPar* xvec = const_cast<hiop::hiopVectorRajaPar*>(xv);
    xvec->copyFromDev();
    return xvec->local_data_const()[i];
}

/// Returns pointer to local ector data
real_type* VectorTestsRajaPar::getLocalData(hiop::hiopVector* x)
{
    hiop::hiopVectorRajaPar* xvec = dynamic_cast<hiop::hiopVectorRajaPar*>(x);
    return xvec->local_data();
}

/// Returns size of local data array for vector _x_
local_ordinal_type VectorTestsRajaPar::getLocalSize(const hiop::hiopVector* x)
{
    const hiop::hiopVectorRajaPar* xvec = dynamic_cast<const hiop::hiopVectorRajaPar*>(x);
    return static_cast<local_ordinal_type>(xvec->get_local_size());
}

#ifdef HIOP_USE_MPI
/// Get communicator
MPI_Comm VectorTestsRajaPar::getMPIComm(hiop::hiopVector* x)
{
    const hiop::hiopVectorRajaPar* xvec = dynamic_cast<const hiop::hiopVectorRajaPar*>(x);
    return xvec->get_mpi_comm();
}
#endif

/// If test fails on any rank set fail flag on all ranks
bool VectorTestsRajaPar::reduceReturn(int failures, hiop::hiopVector* x)
{
    int fail = 0;

#ifdef HIOP_USE_MPI
    MPI_Allreduce(&failures, &fail, 1, MPI_INT, MPI_SUM, getMPIComm(x));
#else
    fail = failures;
#endif

    return (fail != 0);
}


/// Checks if _local_ vector elements are set to `answer`.
int VectorTestsRajaPar::verifyAnswer(hiop::hiopVector* x, real_type answer)
{
  hiop::hiopVectorRajaPar* xvec = dynamic_cast<hiop::hiopVectorRajaPar*>(x);                            

  xvec->copyFromDev();
    const local_ordinal_type N = getLocalSize(x);
    const real_type* xdata = getLocalData(x);

    int local_fail = 0;
    for(local_ordinal_type i=0; i<N; ++i)
        if(!isEqual(xdata[i], answer))
	{
	    std::cout << xdata[i] << " ?= " << answer << "\n";
            ++local_fail;
	}

    return local_fail;
}



} // namespace hiop::tests
