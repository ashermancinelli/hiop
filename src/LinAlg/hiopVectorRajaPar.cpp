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

/**
 * @file hiopVectorRajaPar.cpp
 *
 * @author Asher Mancinelli <asher.mancinelli@pnnl.gov>, PNNL
 * @author Slaven Peles <slaven.peles@pnnl.gov>, PNNL
 * @author Jake K. Ryan <jake.ryan@pnnl.gov>, PNNL
 *
 */
#include "hiopVectorRajaPar.hpp"

#include <cmath>
#include <cstring> //for memcpy
#include <algorithm>
#include <cassert>

#include "hiop_blasdefs.hpp"

#include <limits>
#include <cstddef>

#include <umpire/Allocator.hpp>
#include <umpire/ResourceManager.hpp>

#include <RAJA/RAJA.hpp>


namespace hiop
{
// Define type aliases
using real_type = double;
using local_index_type = int;
using global_index_type = long long;

// Define constants
static constexpr real_type zero = 0.0;
static constexpr real_type one  = 1.0;


#ifdef HIOP_USE_CUDA
  #include "cuda.h"
  const std::string hiop_umpire_dev = "DEVICE";
  #define RAJA_CUDA_BLOCK_SIZE 128
  using hiop_raja_exec   = RAJA::cuda_exec<RAJA_CUDA_BLOCK_SIZE>;
  using hiop_raja_reduce = RAJA::cuda_reduce;
  using hiop_raja_atomic = RAJA::cuda_atomic;
  #define RAJA_LAMBDA [=] __device__
#else
  const std::string hiop_umpire_dev = "HOST"; 
  using hiop_raja_exec   = RAJA::omp_parallel_for_exec;
  using hiop_raja_reduce = RAJA::omp_reduce;
  using hiop_raja_atomic = RAJA::omp_atomic;
  #define RAJA_LAMBDA [=]
#endif

// helper function for the host<->device copy methods
// TODO: investigate potentially better places to put this
template<typename T, typename SizeType>
void registerWith(
  T* ptr,
  SizeType N,
  umpire::ResourceManager& resmgr,
  umpire::Allocator& allocator)
{
  umpire::util::AllocationRecord record{ptr, sizeof(T) * N, allocator.getAllocationStrategy()};
  resmgr.registerAllocation(ptr, record);
}

hiopVectorRajaPar::hiopVectorRajaPar(const long long& glob_n, long long* col_part/*=NULL*/, MPI_Comm comm/*=MPI_COMM_NULL*/)
  : hiopVector(),
    comm_(comm)
{
  n = glob_n; // n is member variable of hiopVector base class

#ifdef HIOP_USE_MPI
  // if this is a serial vector, make sure it has a valid comm in the mpi case
  if(comm_ == MPI_COMM_NULL) 
    comm_ = MPI_COMM_SELF;
#endif

  int P = 0; 
  if(col_part)
  {
#ifdef HIOP_USE_MPI
    int ierr=MPI_Comm_rank(comm_, &P);  assert(ierr==MPI_SUCCESS);
#endif
    glob_il_ = col_part[P];
    glob_iu_ = col_part[P+1];
  } 
  else
  {
    glob_il_ = 0;
    glob_iu_ = n;
  }
  n_local_ = glob_iu_ - glob_il_;

  auto& resmgr = umpire::ResourceManager::getInstance();
  hostalloc_ = resmgr.getAllocator("HOST");
  devalloc_  = resmgr.getAllocator(hiop_umpire_dev);

  data_ = static_cast<double*>(hostalloc_.allocate(n_local_*sizeof(double)));
  data_dev_ = static_cast<double*>(devalloc_.allocate(n_local_*sizeof(double)));
}

hiopVectorRajaPar::hiopVectorRajaPar(const hiopVectorRajaPar& v)
  : hiopVector()
{
  n_local_ = v.n_local_;
  n = v.n;
  glob_il_ = v.glob_il_;
  glob_iu_ = v.glob_iu_;
  comm_ = v.comm_;
  auto& resmgr = umpire::ResourceManager::getInstance();
  hostalloc_ = resmgr.getAllocator("HOST");
  devalloc_  = resmgr.getAllocator(hiop_umpire_dev);

  // data_ = new double[n_local_];
  data_ = static_cast<double*>(hostalloc_.allocate(n_local_*sizeof(double)));
  data_dev_ = static_cast<double*>(devalloc_.allocate(n_local_*sizeof(double)));
}
hiopVectorRajaPar::~hiopVectorRajaPar()
{
  hostalloc_.deallocate(data_);
  devalloc_.deallocate(data_dev_);
  //delete[] data_;

  data_=nullptr;
  data_dev_ = nullptr;
}

hiopVectorRajaPar* hiopVectorRajaPar::alloc_clone() const
{
  hiopVectorRajaPar* v = new hiopVectorRajaPar(*this); assert(v);
  return v;
}
hiopVectorRajaPar* hiopVectorRajaPar::new_copy () const
{
  hiopVectorRajaPar* v = new hiopVectorRajaPar(*this); assert(v);
  v->copyFrom(*this);
  return v;
}

//
// Compute kernels
//

/// Set all vector elements to zero
void hiopVectorRajaPar::setToZero()
{
  auto& rm = umpire::ResourceManager::getInstance();
  rm.memset(data_dev_, 0);
}

/// Set all vector elements to constant c
void hiopVectorRajaPar::setToConstant(double c)
{
  double* data = data_dev_;
  RAJA::forall< hiop_raja_exec >(RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      data[i] = c;
    });
}

/// Set selected elements to constant, zero otherwise
void hiopVectorRajaPar::setToConstant_w_patternSelect(double c, const hiopVector& select)
{
  const hiopVectorRajaPar& s = dynamic_cast<const hiopVectorRajaPar&>(select);
  const double* pattern = s.local_data_dev_const();
  double* data = data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      data[i] = pattern[i]*c;
    });
}

/// Copy data from vec to this vector
void hiopVectorRajaPar::copyFrom(const hiopVector& vec)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  assert(n_local_ == v.n_local_);
  assert(glob_il_ == v.glob_il_);
  assert(glob_iu_ == v.glob_iu_);

  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(data_dev_, v.data_dev_);
}

/// Copy data from local_array to this vector local data
/// TODO: Look into use cases to understand where this array comes from (host/device?)
void hiopVectorRajaPar::copyFrom(const double* local_array)
{
  // TODO: see why this function isn't being called in tests
  if(local_array)
  {
    memcpy(this->data_, local_array, n_local_*sizeof(double));
    //auto& rm = umpire::ResourceManager::getInstance();
    //double* vv = const_cast<double*>(v); // scary
    //registerWith(vv, nv, rm, hostalloc_);
    //rm.copy(this->data_dev_+start_index_in_this, vv, nv*sizeof(double));

  }
}

/// Copy `nv` elements from array `v` to this vector starting from `start_index_in_this`
void hiopVectorRajaPar::copyFromStarting(int start_index_in_this, const double* v, int nv)
{
  assert(start_index_in_this+nv <= n_local_);
  
  // assumes v is a host pointer! TODO: Find common use cases
  auto& rm = umpire::ResourceManager::getInstance();
  double* vv = const_cast<double*>(v); // scary
  registerWith(vv, nv, rm, hostalloc_);
  rm.copy(this->data_dev_+start_index_in_this, vv, nv*sizeof(double));
}

/// Copy data from `vec` starting from `start_index` into this vector
void hiopVectorRajaPar::copyFromStarting(int start_index/*_in_src*/,const hiopVector& vec)
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n && "are you sure you want to call this?");
#endif
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  assert(start_index + v.n_local_ <= n_local_);
  
  auto& rm = umpire::ResourceManager::getInstance();
  double* vv = const_cast<double*>(v.data_dev_); // scary
  rm.copy(this->data_dev_ + start_index, vv, v.n_local_*sizeof(double));
}

/// Copy from `vec` starting at `start_idx_src` into this vector starting at `start_idx_dest`.
void hiopVectorRajaPar::startingAtCopyFromStartingAt(
  int start_idx_src,
  const hiopVector& vec,
  int start_idx_dest)
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n && "are you sure you want to call this?");
#endif
  assert(start_idx_src >= 0 && start_idx_src < this->n_local_);
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  assert(start_idx_dest >=0 && start_idx_dest < v.n_local_);

  int howManyToCopy = this->n_local_ - start_idx_src;
  
  assert(howManyToCopy <= v.n_local_-start_idx_dest);
  howManyToCopy = howManyToCopy <= v.n_local_-start_idx_dest ? howManyToCopy : v.n_local_-start_idx_dest;
  
  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(this->data_dev_+start_idx_src, v.data_dev_+start_idx_dest, howManyToCopy*sizeof(double));
}

/// Copy from this vector starting from `start_index` into `vec`.
void hiopVectorRajaPar::copyToStarting(int start_index, hiopVector& vec)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == n && "are you sure you want to call this?");
#endif
  assert(start_index + v.n_local_ <= n_local_);

  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(v.data_dev_, this->data_dev_ + start_index, v.n_local_*sizeof(double));
}

/// Copy 'this' to `vec` starting at `start_index` in `vec`.
void hiopVectorRajaPar::copyToStarting(hiopVector& vec, int start_index/*_in_dest*/)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  assert(start_index+n_local_ <= v.n_local_);

  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(v.data_dev_ + start_index, this->data_dev_, this->n_local_*sizeof(double));
}

/* copy 'this' (source) starting at 'start_idx_in_src' to 'dest' starting at index 'int start_idx_dest' 
 * If num_elems>=0, 'num_elems' will be copied; if num_elems<0, elements will be copied till the end of
 * either source ('this') or destination ('dest') is reached */
void hiopVectorRajaPar::startingAtCopyToStartingAt(
  int start_idx_in_src, 
  hiopVector& destination, 
  int start_idx_dest, 
  int num_elems/*=-1*/) const
{
  const hiopVectorRajaPar& dest = dynamic_cast<hiopVectorRajaPar&>(destination);

  assert(start_idx_in_src >= 0 && start_idx_in_src < this->n_local_);
  assert(start_idx_dest   >= 0 && start_idx_dest   < dest.n_local_);

  if(num_elems<0)
  {
    num_elems = std::min(this->n_local_ - start_idx_in_src, dest.n_local_ - start_idx_dest);
  } 
  else
  {
    assert(num_elems+start_idx_in_src <= this->n_local_);
    assert(num_elems+start_idx_dest   <= dest.n_local_);
    //make sure everything stays within bounds (in release)
    num_elems = std::min(num_elems, (int)this->n_local_-start_idx_in_src);
    num_elems = std::min(num_elems, (int)dest.n_local_-start_idx_dest);
  }

  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(dest.data_dev_ + start_idx_dest, this->data_dev_ + start_idx_in_src, num_elems*sizeof(double));
}

/// Copy local vector data to a local array
void hiopVectorRajaPar::copyTo(double* dest) const
{
  // TODO: this function is untested
  //memcpy(dest, this->data_, n_local_*sizeof(double));
  
  // assumes dest is a host pointer with len==n_local_
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator allocator = rm.getAllocator("HOST");
  registerWith(dest, n_local_, rm, allocator);
  rm.copy(dest, this->data_dev_, n_local_*sizeof(double));
}

/// L2 Norm
/// TODO: implement with BLAS call (<D>NRM2)
double hiopVectorRajaPar::twonorm() const
{
  double* self_dev = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, double> sum(0.0);
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      sum += self_dev[i] * self_dev[i];
    });
  double nrm = sum.get();

#ifdef HIOP_USE_MPI
  double nrm_global;
  int ierr = MPI_Allreduce(&nrm, &nrm_global, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(MPI_SUCCESS == ierr);
  return std::sqrt(nrm_global);
#endif  
  return std::sqrt(nrm);
}

/// Scalar product of this vector with `vec`
/// TODO: consider implementing with BLAS call (<D>DOT)
double hiopVectorRajaPar::dotProductWith( const hiopVector& vec) const
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  assert(n_local_ == v.n_local_);

  double* dd = data_dev_;
  double* vd = v.data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, double> dot(0.0);
  RAJA::forall<hiop_raja_exec>( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      dot += dd[i] * vd[i];
    });
  double dotprod = dot.get();

#ifdef HIOP_USE_MPI
  double dotprodG;
  int ierr = MPI_Allreduce(&dotprod, &dotprodG, 1, MPI_DOUBLE, MPI_SUM, comm_);
  assert(MPI_SUCCESS==ierr);
  dotprod=dotprodG;
#endif

  return dotprod;
}

/// Infinity norm of `this` vector
double hiopVectorRajaPar::infnorm() const
{
  assert(n_local_ >= 0);
  double* data = data_dev_;
  RAJA::ReduceMax< hiop_raja_reduce, double > norm(0.0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      norm.max(std::abs(data[i]));
    });
  double nrm = norm.get();
#ifdef HIOP_USE_MPI
  double nrm_global;
  int ierr = MPI_Allreduce(&nrm, &nrm_global, 1, MPI_DOUBLE, MPI_MAX, comm_);
  assert(MPI_SUCCESS==ierr);
  return nrm_global;
#endif

  return nrm;
}

/// Infinity norm of local vector data
/// TODO: Not ported to RAJA
double hiopVectorRajaPar::infnorm_local() const
{
  assert(n_local_>=0);
  double nrm=0.;
  if(n_local_>0) {
    nrm = fabs(data_[0]); 
    double aux;
    
    for(int i=1; i<n_local_; i++) {
      aux=fabs(data_[i]);
      if(aux>nrm) nrm=aux;
    }
  }
  return nrm;
}

/// 1-norm of `this` vector
double hiopVectorRajaPar::onenorm() const
{
  double* data = data_dev_;
  RAJA::ReduceSum< hiop_raja_reduce, double > sum(0.0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      sum += std::abs(data[i]);
    });
  double norm1 = sum.get();
#ifdef HIOP_USE_MPI
  double nrm1_global;
  int ierr = MPI_Allreduce(&norm1, &nrm1_global, 1, MPI_DOUBLE, MPI_SUM, comm_); assert(MPI_SUCCESS==ierr);
  return nrm1_global;
#endif
  return norm1;
}

/// 1-norm of local vector data
/// TODO: Not ported to RAJA
double hiopVectorRajaPar::onenorm_local() const
{
  double nrm1=0.;
  for(int i=0; i<n_local_; i++)
    nrm1 += fabs(data_[i]);
  return nrm1;
}

/// Multiply `this` vector by `vec` elementwise and store result in `this`.
void hiopVectorRajaPar::componentMult(const hiopVector& vec)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  assert(n_local_ == v.n_local_);
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      dd[i] *= vd[i];
    });
}

/// @brief Divide `this` vector elemenwise in-place by `vec`. 
/// @pre vec[i] != 0 forall i
void hiopVectorRajaPar::componentDiv (const hiopVector& vec)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  assert(n_local_ == v.n_local_);
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      dd[i] /= vd[i];
    });
}

/// @brief Divide `this` vector elemenwise in-place by `vec` 
/// with pattern selection.
/// @pre vec[i] != 0 forall i
void hiopVectorRajaPar::componentDiv_w_selectPattern( const hiopVector& vec, const hiopVector& pattern)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(vec);
  const hiopVectorRajaPar& ix= dynamic_cast<const hiopVectorRajaPar&>(pattern);
#ifdef HIOP_DEEPCHECKS
  assert(v.n_local_ == n_local_);
  assert(n_local_ == ix.n_local_);
#endif
  double* dd = data_dev_;
  double* vd = v.data_dev_;
  double* id = ix.data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      dd[i] = id[i]*dd[i]/vd[i];
    });
}

/// Scale `this` vector by `c`
// TODO: Consider implementing with BLAS call (<D>SCAL)
void hiopVectorRajaPar::scale(double c)
{
  if(1.0==c)
    return;
  
  double* data = data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      data[i] *= c;
    });
}

/// AXPY kernel
/// TODO: Consider implementing with BLAS call (<D>AXPY)
void hiopVectorRajaPar::axpy(double alpha, const hiopVector& xvec)
{
  const hiopVectorRajaPar& x = dynamic_cast<const hiopVectorRajaPar&>(xvec);
  
  double* yd = data_dev_;
  double* xd = x.data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      // y := a * x + y
      yd[i] = alpha * xd[i] + yd[i];
    });
}

/// this[i] += alpha*x[i]*z[i] forall i
void hiopVectorRajaPar::axzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec)
{
  const hiopVectorRajaPar& x = dynamic_cast<const hiopVectorRajaPar&>(xvec);
  const hiopVectorRajaPar& z = dynamic_cast<const hiopVectorRajaPar&>(zvec);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_ == z.n_local_);
  assert(  n_local_ == z.n_local_);
#endif  
  double *dd       = data_dev_;
  const double *xd = x.local_data_dev_const();
  const double *zd = z.local_data_dev_const();
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      dd[i] += alpha*xd[i]*zd[i];
    });
}

/// this[i] += alpha*x[i]/z[i] forall i
void hiopVectorRajaPar::axdzpy(double alpha, const hiopVector& xvec, const hiopVector& zvec)
{
  const hiopVectorRajaPar& x = dynamic_cast<const hiopVectorRajaPar&>(xvec);
  const hiopVectorRajaPar& z = dynamic_cast<const hiopVectorRajaPar&>(zvec);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_==z.n_local_);
  assert(  n_local_==z.n_local_);
#endif  
  double *yd       = data_dev_;
  const double *xd = x.local_data_dev_const();
  const double *zd = z.local_data_dev_const();
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      yd[i] += alpha*xd[i]/zd[i];
    });
}

/// this[i] += alpha*x[i]/z[i] forall i with pattern selection
void hiopVectorRajaPar::axdzpy_w_pattern( 
  double alpha,
  const hiopVector& xvec, 
  const hiopVector& zvec,
  const hiopVector& select)
{
  const hiopVectorRajaPar& x = dynamic_cast<const hiopVectorRajaPar&>(xvec);
  const hiopVectorRajaPar& z = dynamic_cast<const hiopVectorRajaPar&>(zvec);
  const hiopVectorRajaPar& sel = dynamic_cast<const hiopVectorRajaPar&>(select);
#ifdef HIOP_DEEPCHECKS
  assert(x.n_local_==z.n_local_);
  assert(  n_local_==z.n_local_);
#endif  
  double* yd = data_dev_;
  const double* xd = x.local_data_dev_const();
  const double* zd = z.local_data_dev_const(); 
  const double* id = sel.local_data_dev_const();
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) 
    {
      assert(id[i] == one || id[i] == zero);
      yd[i] += id[i] * alpha * xd[i] / zd[i];
    });
}

/// Add constant elementwise
void hiopVectorRajaPar::addConstant(double c)
{
  double *yd = data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      yd[i] += c;
    });
}

/// Add constant elementwise with pattern selection
void  hiopVectorRajaPar::addConstant_w_patternSelect(double c, const hiopVector& select)
{
  const hiopVectorRajaPar& sel = dynamic_cast<const hiopVectorRajaPar&>(select);
  assert(this->n_local_ == sel.n_local_);
  double *data = data_dev_;
  const double *id = sel.local_data_dev_const();
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      data[i] += id[i]*c;
    });
}

/// Find minimum vector element
void hiopVectorRajaPar::min( double& /* m */, int& /* index */) const
{
  assert(false && "not implemented");
}

/// Negate vector elements
// TODO: Consider implementing with BLAS call (<D>SCAL)
void hiopVectorRajaPar::negate()
{
  double* data = data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      data[i] *= -1;
    });
}

/// Invert vector elements
void hiopVectorRajaPar::invert()
{
  const double small_real = 1e-35;
  double *data = data_dev_;
  RAJA::forall< hiop_raja_exec >(RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
#ifdef HIOP_DEEPCHECKS
      assert(std::abs(data[i]) > small_real);
#endif
      data[i] = one/data[i];
    });
}

/// Sum all select[i]*log(this[i]), select[i] = 0,1
double hiopVectorRajaPar::logBarrier(const hiopVector& select) const
{
  const hiopVectorRajaPar& sel = dynamic_cast<const hiopVectorRajaPar&>(select);
  assert(this->n_local_ == sel.n_local_);

  double* data = data_dev_;
  const double* id = sel.local_data_dev_const();
  RAJA::ReduceSum< hiop_raja_reduce, double > sum(0.0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
		RAJA_LAMBDA(RAJA::Index_type i)
    {
#ifdef HIOP_DEEPCHECKS
      assert(id[i] == one || id[i] == zero);
#endif
      sum += id[i] * std::log(data[i]);
		});
  double res = sum.get();

  // Comment out MPI code until clarified if it is needed.
  // #ifdef HIOP_USE_MPI
  //   double sum_global;
  //   int ierr = MPI_Allreduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, comm_); assert(MPI_SUCCESS==ierr);
  //   return sum_global;
  // #endif
  return res;
}

/* adds the gradient of the log barrier, namely this=this+alpha*1/select(x) */
void hiopVectorRajaPar::addLogBarrierGrad(
  double alpha,
  const hiopVector& xvec,
  const hiopVector& select)
{
  const hiopVectorRajaPar& x = dynamic_cast<const hiopVectorRajaPar&>(xvec);
  const hiopVectorRajaPar& sel = dynamic_cast<const hiopVectorRajaPar&>(select);  
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == x.n_local_);
  assert(n_local_ == sel.n_local_);
#endif
  double* data = data_dev_;
  const double* xd = x.local_data_dev_const();
  const double* id = sel.local_data_dev_const();
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) 
    {
      if (id[i] == 1.0) 
        data[i] += alpha/xd[i];
    });
}

/// Linear damping term (?)
double hiopVectorRajaPar::linearDampingTerm(
  const hiopVector& ixleft,
  const hiopVector& ixright,
	const double& mu,
  const double& kappa_d) const
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local_ == (dynamic_cast<const hiopVectorRajaPar&>(ixleft)).n_local_);
  assert(n_local_ == (dynamic_cast<const hiopVectorRajaPar&>(ixright)).n_local_);
#endif
  const double* ld = (dynamic_cast<const hiopVectorRajaPar&>(ixleft)).local_data_dev_const();
  const double* rd = (dynamic_cast<const hiopVectorRajaPar&>(ixright)).local_data_dev_const();
  double* data = data_dev_;
  RAJA::ReduceSum< hiop_raja_reduce, double > sum(zero);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
		RAJA_LAMBDA(RAJA::Index_type i)
    {
      if (ld[i] == one && rd[i] == zero)
        sum += data[i];
		});
  double term = static_cast<double>(sum.get());
  term *= mu; 
  term *= kappa_d;
  return term;
}

/// Check if all elements of the vector are positive
int hiopVectorRajaPar::allPositive()
{
  double* data = data_dev_;
  RAJA::ReduceMin< hiop_raja_reduce, double > minimum(one);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
		RAJA_LAMBDA(RAJA::Index_type i)
    {
      minimum.min(data[i]);
		});
  int allPos = minimum.get() > zero ? 1 : 0;

#ifdef HIOP_USE_MPI
  int allPosG;
  int ierr=MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm_); assert(MPI_SUCCESS==ierr);
  return allPosG;
#endif
  return allPos;
}

/// Project solution into bounds
bool hiopVectorRajaPar::projectIntoBounds(
  const hiopVector& xlo_, 
  const hiopVector& ixl_,
	const hiopVector& xup_,
  const hiopVector& ixu_,
	double kappa1,
  double kappa2)
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorRajaPar&>(xlo_)).n_local_ == n_local_);
  assert((dynamic_cast<const hiopVectorRajaPar&>(ixl_)).n_local_ == n_local_);
  assert((dynamic_cast<const hiopVectorRajaPar&>(xup_)).n_local_ == n_local_);
  assert((dynamic_cast<const hiopVectorRajaPar&>(ixu_)).n_local_ == n_local_);
#endif
  const double* xld = (dynamic_cast<const hiopVectorRajaPar&>(xlo_)).local_data_dev_const();
  const double* ild = (dynamic_cast<const hiopVectorRajaPar&>(ixl_)).local_data_dev_const();
  const double* xud = (dynamic_cast<const hiopVectorRajaPar&>(xup_)).local_data_dev_const();
  const double* iud = (dynamic_cast<const hiopVectorRajaPar&>(ixu_)).local_data_dev_const();
  double* xd = data_dev_; 
  // Perform preliminary check to see of all upper value
  RAJA::ReduceMin< hiop_raja_reduce, double > minimum(one);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i)
      {
        minimum.min(xud[i] - xld[i]);
      });
  if (minimum.get() < zero) 
    return false;

  const double small_real = std::numeric_limits<double>::min() * 100;

  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      double aux, aux2;
      if(ild[i] != zero && iud[i] != zero)
      {
        aux = kappa2*(xud[i] - xld[i]) - small_real;
        aux2 = xld[i] + std::min(kappa1*std::max(one, std::abs(xld[i])), aux);
        if(xd[i] < aux2)
        {
          xd[i] = aux2;
        }
        else
        {
          aux2=xud[i] - std::min(kappa1*fmax(one, std::abs(xud[i])), aux);
          if(xd[i] > aux2)
          {
            xd[i] = aux2;
          }
        }
#ifdef HIOP_DEEPCHECKS
      assert(xd[i] > xld[i] && xd[i] < xud[i] && "this should not happen -> HiOp bug");
#endif
      }
      else
      {
        if(ild[i] != zero)
          xd[i] = std::max(xd[i], xld[i] + kappa1*std::max(one, std::abs(xld[i])) - small_real);
        else 
          if(iud[i] != zero)
            xd[i] = std::min(xd[i], xud[i] - kappa1*std::max(one, std::abs(xud[i])) - small_real);
          else { /*nothing for free vars  */ }
      }
    });
  return true;
}

/// max{a\in(0,1]| x+ad >=(1-tau)x} 
/// TODO: It is unclear how this works at all and passes the test
double hiopVectorRajaPar::fractionToTheBdry(const hiopVector& dx, const double& tau) const
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorRajaPar&>(dx) ).n_local_==n_local_);
  assert(tau>0);
  assert(tau<1);
#endif

  const double* d = (dynamic_cast<const hiopVectorRajaPar&>(dx) ).local_data_dev_const();
  const double* x = data_dev_;

  RAJA::ReduceMin< hiop_raja_reduce, double > minimum(one);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      if(d[i]>=0)
        return;
#ifdef HIOP_DEEPCHECKS
      assert(x[i]>0);
#endif
      minimum.min(-tau*x[i]/d[i]);
    });
  return minimum.get();
}

/* max{a\in(0,1]| x+ad >=(1-tau)x} */
double hiopVectorRajaPar::fractionToTheBdry_w_pattern(const hiopVector& dx, const double& tau, const hiopVector& ix) const
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorRajaPar&>(dx) ).n_local_==n_local_);
  assert((dynamic_cast<const hiopVectorRajaPar&>(ix) ).n_local_==n_local_);
  assert(tau>0);
  assert(tau<1);
#endif
  const double* d = (dynamic_cast<const hiopVectorRajaPar&>(dx) ).local_data_dev_const();
  const double* x = data_dev_;
  const double* pat = (dynamic_cast<const hiopVectorRajaPar&>(ix) ).local_data_dev_const();

  RAJA::ReduceMin< hiop_raja_reduce, double > aux(one);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      if(d[i] >= 0 || pat[i] == 0)
        return;
#ifdef HIOP_DEEPCHECKS
      assert(x[i]>0);
#endif
      aux.min(-tau*x[i]/d[i]);
    });
  return aux.get();
}

void hiopVectorRajaPar::selectPattern(const hiopVector& ix_)
{
  const hiopVectorRajaPar& x = dynamic_cast<const hiopVectorRajaPar&>(ix_);
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorRajaPar&>(ix_) ).n_local_==n_local_);
#endif

  double* self_data = this->data_dev_;
  double* select_data = x.data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      self_data[i] *= select_data[i] == 1 ? 1 : 0;
    });
}

bool hiopVectorRajaPar::matchesPattern(const hiopVector& ix_)
{  
  const hiopVectorRajaPar& x = dynamic_cast<const hiopVectorRajaPar&>(ix_);
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorRajaPar&>(ix_) ).n_local_==n_local_);
#endif
  // const double* ix = (dynamic_cast<const hiopVectorRajaPar&>(ix_) ).local_data_const();
  // int bmatches=true;
  // double* x=data_;
  // for(int i=0; (i<n_local_) && bmatches; i++) 
  //   if(ix[i]==0.0 && x[i]!=0.0) bmatches=false;

  int bmatches = true;
  double* self_data = this->data_dev_;
  double* select_data = x.data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, int> ndiff(0);
  RAJA::forall<hiop_raja_exec>( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      int fail = self_data[i]!=0.0 && select_data[i]==0.0;
      ndiff += fail;
    });
  bmatches = (ndiff.get() == 0);

#ifdef HIOP_USE_MPI
  int bmatches_glob=bmatches;
  int ierr=MPI_Allreduce(&bmatches, &bmatches_glob, 1, MPI_INT, MPI_LAND, comm_); assert(MPI_SUCCESS==ierr);
  return bmatches_glob;
#endif
  return bmatches;
}

int hiopVectorRajaPar::allPositive_w_patternSelect(const hiopVector& w_)
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorRajaPar&>(w_)).n_local_ == n_local_);
#endif 
  const double* w = (dynamic_cast<const hiopVectorRajaPar&>(w_) ).local_data_dev_const();

  // Benchmark assignment vs increment for any
  //
  // Scan?
  // axpy with data_ & w and check reduce min for 0?
  const double* local_data_dev = data_dev_;
  RAJA::ReduceSum< hiop_raja_reduce, int > any(0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
                                  RAJA_LAMBDA(RAJA::Index_type i) {
                                    if(w[i]!=0.0 && local_data_dev[i]<=0.) any += 1;
                                  });
  int allPos = any.get() == 0;
  
#ifdef HIOP_USE_MPI
  int allPosG;
  int ierr = MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm_); assert(MPI_SUCCESS==ierr);
  return allPosG;
#endif  
  return allPos;
}

void hiopVectorRajaPar::adjustDuals_plh(const hiopVector& x_, const hiopVector& ix_, const double& mu, const double& kappa)
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorRajaPar&>(x_) ).n_local_==n_local_);
  assert((dynamic_cast<const hiopVectorRajaPar&>(ix_)).n_local_==n_local_);
#endif
  const double* x  = (dynamic_cast<const hiopVectorRajaPar&>(x_ )).local_data_dev_const();
  const double* ix = (dynamic_cast<const hiopVectorRajaPar&>(ix_)).local_data_dev_const();
  double* z=data_dev_; //the dual

  // **benchmark baseline sparsity of pattern vector**
  // How sparse is ix?
  //
  // 1. compressed copy of ix
  //    instead of range segmetn(0, nlocl)
  //    would haev list segment of populated indecices of ix
  //    that way, each thread will run the inner conditional
  //
  //    the same ix pattern will be called many other times. If 
  //    we create a compressed ix pattern vector, maybe keep it
  //    for other functiosn to use
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
          RAJA_LAMBDA(RAJA::Index_type i) {
              double a,b;
              // preemptive loop to reduce number of iterations?
              if(ix[i]==1.) {
                  // precompute a and b in another loop?
                  a=mu/x[i]; b=a/kappa; a=a*kappa;

                  // Necessary conditionals
                  if(z[i]<b) 
                      z[i]=b;
                  else //z[i]>=b
                      if(a<=b) 
                          z[i]=b;
                      else //a>b
                          if(a<z[i]) z[i]=a;
                  // - - - - 
                  //else a>=z[i] then *z=*z (z[i] does not need adjustment)
              }
          });
}

bool hiopVectorRajaPar::isnan() const
{
  // for(long long i=0; i<n_local_; i++) if(std::isnan(data_[i])) return true;
  // return false;
  double* local_data_dev = data_dev_;
  RAJA::ReduceSum< hiop_raja_reduce, int > any(0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
                                  RAJA_LAMBDA(RAJA::Index_type i) {
                                    if (0==std::isnan(local_data_dev[i])) any += 1;
                                  });
  return any.get() == 0;
}

bool hiopVectorRajaPar::isinf() const
{
  double* local_data_dev = data_dev_;
  RAJA::ReduceSum< hiop_raja_reduce, int > any(0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
                                  RAJA_LAMBDA(RAJA::Index_type i) {
                                    if (0==std::isinf(local_data_dev[i])) any += 1;
                                  });
  return any.get() == 0;
}

bool hiopVectorRajaPar::isfinite() const
{
  double* local_data_dev = data_dev_;
  RAJA::ReduceSum< hiop_raja_reduce, int > any(0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
                                  RAJA_LAMBDA(RAJA::Index_type i) {
                                    if (0==std::isfinite(local_data_dev[i])) any += 1;
                                  });
  return any.get() == 0;
}

void hiopVectorRajaPar::print(FILE* file, const char* msg/*=NULL*/, int max_elems/*=-1*/, int rank/*=-1*/) const
{
  int myrank=0, numranks=1; 
#ifdef HIOP_USE_MPI
  if(rank>=0) {
    int err = MPI_Comm_rank(comm_, &myrank); assert(err==MPI_SUCCESS);
    err = MPI_Comm_size(comm_, &numranks); assert(err==MPI_SUCCESS);
  }
#endif
  if(myrank==rank || rank==-1) {
    if(max_elems>n_local_) max_elems=n_local_;

    if(NULL==msg) {
      if(numranks>1)
	fprintf(file, "vector of size %lld, printing %d elems (on rank=%d)\n", n, max_elems, myrank);
      else
	fprintf(file, "vector of size %lld, printing %d elems (serial)\n", n, max_elems);
    } else {
      fprintf(file, "%s ", msg);
    }    
    fprintf(file, "=[");
    max_elems = max_elems>=0?max_elems:n_local_;
    for(int it=0; it<max_elems; it++)  fprintf(file, "%22.16e ; ", data_[it]);
    fprintf(file, "];\n");
  }
}

void hiopVectorRajaPar::copyToDev()
{
  auto& resmgr = umpire::ResourceManager::getInstance();
  resmgr.copy(data_dev_, data_);
}

void hiopVectorRajaPar::copyFromDev()
{
  auto& resmgr = umpire::ResourceManager::getInstance();
  resmgr.copy(data_, data_dev_);
}


} // namespace hiop
