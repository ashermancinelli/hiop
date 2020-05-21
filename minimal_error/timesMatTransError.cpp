#include <iostream>
#include <cassert>

#include <hiopVector.hpp>
#include <hiopMatrix.hpp>

int main(int argc, char** argv)
{
  int rank=0, numRanks=1;
  MPI_Comm comm = MPI_COMM_NULL;

#ifdef HIOP_USE_MPI
  int err;
  err = MPI_Init(&argc, &argv);        assert(MPI_SUCCESS==err);
  err = MPI_Comm_rank(comm,&rank);     assert(MPI_SUCCESS==err);
  err = MPI_Comm_size(comm,&numRanks); assert(MPI_SUCCESS==err);
  if(0 == rank)
    printf("Support for MPI is enabled\n");
#endif

  global_ordinal_type M_local = 5 * numRanks;
  global_ordinal_type N_local = 10 * M_local;
  global_ordinal_type M_global = M_local * numRanks;
  global_ordinal_type N_global = N_local * numRanks;

  auto n_partition = new global_ordinal_type[numRanks+1];
  auto m_partition = new global_ordinal_type[numRanks+1];
  n_partition[0] = 0;
  m_partition[0] = 0;

  for(int i = 1; i < numRanks + 1; ++i)
  {
    n_partition[i] = i*N_local;
    m_partition[i] = i*M_local;
  }

  hiop::hiopMatrixDense A(K_global, M_global, m_partition, comm);
  hiop::hiopMatrixDense W(K_global, N_global);
  hiop::hiopMatrixDense X(N_global, M_global, m_partition, comm);

  fail += test.matrixTimesMatTrans(A_kxm, A_kxn_local, A_nxm, rank);

  const real_type A_val = 2.,
        X_val = 3.,
        W_val = 2.,
        alpha = 2.,
        beta  = 2.;

  A.setToConstant(A_val);
  W_local.setToConstant(W_val);
  X.setToConstant(X_val);
  A.timesMatTrans(beta, W_local, alpha, X);

  real_type expected = (beta * W_val) + (alpha * A_val * X_val * M);

  const int fail = verifyAnswer(&W_local, expected);
  std::cout "-- Fail: " << fail << "\n";

#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif

  return fail;
}
