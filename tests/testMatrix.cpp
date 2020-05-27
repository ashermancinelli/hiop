// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory (LLNL).
// Written by Cosmin G. Petra, petra1@llnl.gov.
// LLNL-CODE-742473. All rights reserved.
//
// This file is part of HiOp. For details, see https://github.com/LLNL/hiop. HiOp 
// is released under the BSD 3-clause license (https://opensource.org/licenses/BSD-3-Clause). 
// Please also read “Additional BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice, this list 
// of conditions and the disclaimer below.
// ii. Redistributions in binary form must reproduce the above copyright notice, 
// this list of conditions and the disclaimer (as noted below) in the documentation and/or 
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may be used to 
// endorse or promote products derived from this software without specific prior written 
// permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE U.S. DEPARTMENT OF ENERGY OR 
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS 
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
// AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S. Department 
// of Energy (DOE). This work was produced at Lawrence Livermore National Laboratory under 
// Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National Security, LLC 
// nor any of their employees, makes any warranty, express or implied, or assumes any 
// liability or responsibility for the accuracy, completeness, or usefulness of any 
// information, apparatus, product, or process disclosed, or represents that its use would
// not infringe privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or services by 
// trade name, trademark, manufacturer or otherwise does not necessarily constitute or 
// imply its endorsement, recommendation, or favoring by the United States Government or 
// Lawrence Livermore National Security, LLC. The views and opinions of authors expressed 
// herein do not necessarily state or reflect those of the United States Government or 
// Lawrence Livermore National Security, LLC, and shall not be used for advertising or 
// product endorsement purposes.
#include <iostream>
#include <cassert>

#include <hiopVector.hpp>
#include <hiopMatrix.hpp>
#include "LinAlg/matrixTestsDense.hpp"

int main(int argc, char** argv)
{
    int rank=0, numRanks=1;
    MPI_Comm comm = MPI_COMM_NULL;

#ifdef HIOP_USE_MPI
    int err;
    err = MPI_Init(&argc, &argv);                  assert(MPI_SUCCESS==err);
    comm = MPI_COMM_WORLD;
    err = MPI_Comm_rank(comm,&rank);     assert(MPI_SUCCESS==err);
    err = MPI_Comm_size(comm,&numRanks); assert(MPI_SUCCESS==err);
    if(0 == rank)
        printf("Support for MPI is enabled\n");
#endif

    global_ordinal_type M_local = 5 * numRanks;
    global_ordinal_type K_local = 2 * M_local;
    global_ordinal_type N_local = 10 * M_local;

    // all distribution occurs column-wise, so any length 
    // that will be used as a column of a matrix will have
    // to be scaled up by numRanks
    global_ordinal_type M_global = M_local * numRanks;
    global_ordinal_type K_global = K_local * numRanks;
    global_ordinal_type N_global = N_local * numRanks;
    
    auto n_partition = new global_ordinal_type[numRanks+1];
    auto k_partition = new global_ordinal_type[numRanks+1];
    auto m_partition = new global_ordinal_type[numRanks+1];
    n_partition[0] = 0;
    k_partition[0] = 0;
    m_partition[0] = 0;

    for(int i = 1; i < numRanks + 1; ++i)
    {
        n_partition[i] = i*N_local;
        k_partition[i] = i*K_local;
        m_partition[i] = i*M_local;
    }

    int fail = 0;

    // Test dense matrix
    {
        // Matrix dimensions denoted by subscript
        // Distributed matrices:
        hiop::hiopMatrixDense A_mxn(M_global, N_global, n_partition, comm);
        hiop::hiopMatrixDense B_mxn(M_global, N_global, n_partition, comm);
        hiop::hiopMatrixDense A_nxm(N_global, M_global, m_partition, comm);
        hiop::hiopMatrixDense A_nxn(N_global, N_global, n_partition, comm);
        hiop::hiopMatrixDense B_nxn(N_global, N_global, n_partition, comm);
        hiop::hiopMatrixDense A_kxn(K_global, N_global, n_partition, comm);
        hiop::hiopMatrixDense A_kxk(K_global, K_global, k_partition, comm);
        hiop::hiopMatrixDense A_mxk(M_global, K_global, k_partition, comm);
        hiop::hiopMatrixDense A_mxm(M_global, M_global, m_partition, comm);
        hiop::hiopMatrixDense A_kxm(K_global, M_global, m_partition, comm);

        // Some matrices need to scale only in one dimension as
        // ranks scale. The subscripted size will denote which
        // dimension will not scale with ranks by suffixing an
        // 'l' to the dimension.
        hiop::hiopMatrixDense A_klxm(K_local, M_global, m_partition, comm);
        hiop::hiopMatrixDense A_mlxk(M_local, K_global, k_partition, comm);

        // Data is local even though sizes are global such that they can be
        // multiplied with the distributed matrices
        hiop::hiopMatrixDense A_mxk_local(M_global, K_global);
        hiop::hiopMatrixDense A_kxn_local(K_global, N_global);
        hiop::hiopMatrixDense A_mxn_local(M_global, N_global);

        // Vectors with shape of the form:
        // x_<size>_<distributed or local>
        //
        // Distributed vectors
        hiop::hiopVectorPar x_n_dist(N_global, n_partition, comm);
        hiop::hiopVectorPar x_m_dist(M_global, m_partition, comm);

        // Local vectors
        hiop::hiopVectorPar x_n_nodist(N_global);
        hiop::hiopVectorPar x_m_nodist(M_global);

        hiop::tests::MatrixTestsDense test;

        fail += test.matrixSetToZero(A_mxn, rank);
        fail += test.matrixSetToConstant(A_mxn, rank);
        fail += test.matrixTimesVec(A_mxn, x_m_nodist, x_n_dist, rank);
        fail += test.matrixTransTimesVec(A_mxn, x_m_nodist, x_n_dist, rank);

        if (numRanks == 1)
        {
            // These functions are only meant to be called locally
            fail += test.matrixTimesMat(A_mxk_local, A_kxn_local, A_mxn_local, rank);
            fail += test.matrixAddDiagonal(A_nxn, x_n_nodist, rank);
            fail += test.matrixAddSubDiagonal(A_nxn, x_m_nodist, rank);
            fail += test.matrixAddToSymDenseMatrixUpperTriangle(A_nxn, A_mlxk, rank);
            fail += test.matrixTransAddToSymDenseMatrixUpperTriangle(A_nxn, A_klxm, rank);
            fail += test.matrixAddUpperTriangleToSymDenseMatrixUpperTriangle(A_nxn, A_mxm, rank);
#ifdef HIOP_DEEPCHECKS
            fail += test.matrixAssertSymmetry(A_nxn, rank);
#endif
        }

        fail += test.matrixTransTimesMat(A_mxk_local, A_kxn, A_mxn, rank);
        fail += test.matrixTimesMatTrans(A_kxm, A_kxn_local, A_nxm, rank);
        fail += test.matrixAddMatrix(A_mxn, B_mxn, rank);
        fail += test.matrixMaxAbsValue(A_mxn, rank);
        fail += test.matrixIsFinite(A_mxn, rank);
        fail += test.matrixNumRows(A_mxn, M_global, rank);
        fail += test.matrixNumCols(A_mxn, N_global, rank);
    }

    // Test RAJA matrix
    {
      // Code here ...
    }

    if (rank == 0)
    {
      if(fail)
      {
        std::cout << "Matrix tests failed\n";
      }
      else
      {
        std::cout << "Matrix tests passed\n";
      }
    }

#ifdef HIOP_USE_MPI
    MPI_Finalize();
#endif

    return fail;
}
