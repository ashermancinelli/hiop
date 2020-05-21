#include <iostream>
#include <cassert>

#include <hiopVector.hpp>
#include <hiopMatrix.hpp>
#include <hiop_defs.hpp>

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

  long long int M_local = 5 * numRanks;
  long long int K_local = 5 * M_local;
  long long int N_local = 10 * M_local;

  long long int M_global = M_local * numRanks;
  long long int K_global = K_local * numRanks;
  long long int N_global = N_local * numRanks;

  auto n_partition = new long long int[numRanks+1];
  auto m_partition = new long long int[numRanks+1];
  auto k_partition = new long long int[numRanks+1];
  n_partition[0] = 0;
  m_partition[0] = 0;
  k_partition[0] = 0;

  for(int i = 1; i < numRanks + 1; ++i)
  {
    n_partition[i] = i*N_local;
    k_partition[i] = i*K_local;
    m_partition[i] = i*M_local;
  }

  hiop::hiopMatrixDense A(K_global, M_global, m_partition, comm);
  hiop::hiopMatrixDense W(K_global, N_global);
  hiop::hiopMatrixDense X(N_global, M_global, m_partition, comm);

  const double A_val = 2.,
        X_val = 3.,
        W_val = 2.,
        alpha = 2.,
        beta  = 2.;

  A.setToConstant(A_val);
  W.setToConstant(W_val);
  X.setToConstant(X_val);
  A.timesMatTrans(beta, W, alpha, X);

  double expected = (beta * W_val) + (alpha * A_val * X_val * M_global);

#ifdef HIOP_USE_MPI
  MPI_Finalize();
#endif

  return 0;
}
