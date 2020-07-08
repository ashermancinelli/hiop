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
  n = glob_n;

#ifdef HIOP_USE_MPI
  // if this is a serial vector, make sure it has a valid comm in the mpi case
  if(comm_==MPI_COMM_NULL) comm_=MPI_COMM_SELF;
#endif

  int P=0; 
  if(col_part) {
#ifdef HIOP_USE_MPI
    int ierr=MPI_Comm_rank(comm_, &P);  assert(ierr==MPI_SUCCESS);
#endif
    glob_il_=col_part[P]; glob_iu_=col_part[P+1];
  } else {
    glob_il_=0; glob_iu_=n;
  }
  n_local_=glob_iu_-glob_il_;

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

//hiopVector* hiopVectorRajaPar::new_alloc() const
//{ }
//hiopVector* hiopVectorRajaPar::new_copy() const
//{ }


void hiopVectorRajaPar::setToZero()
{
  auto& rm = umpire::ResourceManager::getInstance();
  rm.memset(data_dev_, 0);
}


void hiopVectorRajaPar::setToConstant(double c)
{
  double* local_data_dev = this->data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i)
    {
      local_data_dev[i] = c;
    });
}

void hiopVectorRajaPar::setToConstant_w_patternSelect(double c, const hiopVector& select)
{
  const hiopVectorRajaPar& s = dynamic_cast<const hiopVectorRajaPar&>(select);
  const double* svec = s.local_data_dev_const();
  double* local_data_dev = this->data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      if(svec[i]==1.)
        local_data_dev[i]=c;
      else
        local_data_dev[i]=0.;
    });
}
void hiopVectorRajaPar::copyFrom(const hiopVector& v_ )
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(v_);
  assert(n_local_==v.n_local_);
  assert(glob_il_==v.glob_il_); assert(glob_iu_==v.glob_iu_);

  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(data_dev_, v.data_dev_);
}

void hiopVectorRajaPar::copyFrom(const double* v_local_data )
{
  // TODO: see why this function isn't being called in tests
  if(v_local_data)
  {
    memcpy(this->data_, v_local_data, n_local_*sizeof(double));
    //auto& rm = umpire::ResourceManager::getInstance();
    //double* vv = const_cast<double*>(v); // scary
    //registerWith(vv, nv, rm, hostalloc_);
    //rm.copy(this->data_dev_+start_index_in_this, vv, nv*sizeof(double));

  }
}

void hiopVectorRajaPar::copyFromStarting(int start_index_in_this, const double* v, int nv)
{
  assert(start_index_in_this+nv <= n_local_);
  
  // assumes v is a host pointer!
  auto& rm = umpire::ResourceManager::getInstance();
  double* vv = const_cast<double*>(v); // scary
  registerWith(vv, nv, rm, hostalloc_);
  rm.copy(this->data_dev_+start_index_in_this, vv, nv*sizeof(double));
}

void hiopVectorRajaPar::copyFromStarting(int start_index/*_in_src*/,const hiopVector& v_)
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local_==n && "are you sure you want to call this?");
#endif
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(v_);
  assert(start_index+v.n_local_ <= n_local_);
  
  auto& rm = umpire::ResourceManager::getInstance();
  double* vv = const_cast<double*>(v.data_dev_); // scary
  rm.copy(this->data_dev_+start_index, vv, v.n_local_*sizeof(double));
}

void hiopVectorRajaPar::startingAtCopyFromStartingAt(
  int start_idx_src,
  const hiopVector& v_,
  int start_idx_dest)
{
#ifdef HIOP_DEEPCHECKS
  assert(n_local_==n && "are you sure you want to call this?");
#endif
  assert(start_idx_src>=0 && start_idx_src<this->n_local_);
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(v_);
  assert(start_idx_dest>=0 && start_idx_dest<v.n_local_);

  int howManyToCopy = this->n_local_ - start_idx_src;
  
  assert(howManyToCopy <= v.n_local_-start_idx_dest);
  howManyToCopy = howManyToCopy <= v.n_local_-start_idx_dest ? howManyToCopy : v.n_local_-start_idx_dest;
  
  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(this->data_dev_+start_idx_src, v.data_dev_+start_idx_dest, howManyToCopy*sizeof(double));
}

void hiopVectorRajaPar::copyToStarting(int start_index, hiopVector& v_)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(v_);
#ifdef HIOP_DEEPCHECKS
  assert(n_local_==n && "are you sure you want to call this?");
#endif
  assert(start_index+v.n_local_ <= n_local_);

  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(v.data_dev_, this->data_dev_+start_index, v.n_local_*sizeof(double));
}

/* Copy 'this' to v starting at start_index in 'v'. */
void hiopVectorRajaPar::copyToStarting(hiopVector& v_, int start_index/*_in_dest*/)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(v_);
  assert(start_index+n_local_ <= v.n_local_);

  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(v.data_dev_+start_index, this->data_dev_, this->n_local_*sizeof(double));
}

/* copy 'this' (source) starting at 'start_idx_in_src' to 'dest' starting at index 'int start_idx_dest' 
 * If num_elems>=0, 'num_elems' will be copied; if num_elems<0, elements will be copied till the end of
 * either source ('this') or destination ('dest') is reached */
void hiopVectorRajaPar::startingAtCopyToStartingAt(
  int start_idx_in_src, 
  hiopVector& dest_, 
  int start_idx_dest, 
  int num_elems/*=-1*/) const
{
  assert(start_idx_in_src>=0 && start_idx_in_src<this->n_local_);
  const hiopVectorRajaPar& dest = dynamic_cast<hiopVectorRajaPar&>(dest_);
  assert(start_idx_dest>=0 && start_idx_dest<dest.n_local_);
  if(num_elems<0) {
    num_elems = std::min(this->n_local_-start_idx_in_src, dest.n_local_-start_idx_dest);
  } else {
    assert(num_elems+start_idx_in_src <= this->n_local_);
    assert(num_elems+start_idx_dest   <= dest.n_local_);
    //make sure everything stays within bounds (in release)
    num_elems = std::min(num_elems, (int)this->n_local_-start_idx_in_src);
    num_elems = std::min(num_elems, (int)dest.n_local_-start_idx_dest);
  }

  auto& rm = umpire::ResourceManager::getInstance();
  rm.copy(dest.data_dev_ + start_idx_dest, this->data_dev_ + start_idx_in_src, num_elems*sizeof(double));
}

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

// TODO: implement with BLAS call (DNRM2)
double hiopVectorRajaPar::twonorm() const
{
  double* self_dev = data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, double> res(0.0);
  RAJA::forall<hiop_raja_exec>(RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      res += self_dev[i] * self_dev[i];
    });
  double nrm = std::sqrt(res.get());

#ifdef HIOP_USE_MPI
  nrm *= nrm;
  double nrmG;
  int ierr = MPI_Allreduce(&nrm, &nrmG, 1, MPI_DOUBLE, MPI_SUM, comm_); assert(MPI_SUCCESS==ierr);
  nrm=sqrt(nrmG);
#endif  
  return nrm;
}

// TODO: implement with BLAS call (DDOT)
double hiopVectorRajaPar::dotProductWith( const hiopVector& v_ ) const
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(v_);
  assert(this->n_local_==v.n_local_);

  /* 
   * blas ddot function
   *
   * TODO template function to take raja policy
   *
   * use blas *macros* to mimic template fucntionality to generate
   * functions that will run on the target device
   */

  //int one=1; int n=n_local_;
  //double dotprod=DDOT(&n, this->data_, &one, v.data_, &one);

  double* dd = this->data_dev_;
  double* vd = v.data_dev_;
  RAJA::ReduceSum<hiop_raja_reduce, double> dot(0.0);
  RAJA::forall<hiop_raja_exec>( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      dot += dd[i] * vd[i];
    });
  double dotprod = dot.get();

#ifdef HIOP_USE_MPI
  double dotprodG;
  int ierr = MPI_Allreduce(&dotprod, &dotprodG, 1, MPI_DOUBLE, MPI_SUM, comm_); assert(MPI_SUCCESS==ierr);
  dotprod=dotprodG;
#endif

  return dotprod;
}

double hiopVectorRajaPar::infnorm() const
{
  assert(n_local_>=0);
  //double nrm=0.;
  //if(n_local_!=0) {
  //  nrm=fabs(data_[0]);
  //  double aux;
  
  //  for(int i=1; i<n_local_; i++) {
  //    aux=fabs(data_[i]);
  //    if(aux>nrm) nrm=aux;
  //  }
  //}
  double* self_data = data_dev_;
  RAJA::ReduceMax< hiop_raja_reduce, double > norm(0.0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      norm.max(std::abs(self_data[i]));
    });
  double nrm = static_cast<double>(norm.get());
#ifdef HIOP_USE_MPI
  double nrm_glob;
  int ierr = MPI_Allreduce(&nrm, &nrm_glob, 1, MPI_DOUBLE, MPI_MAX, comm_); assert(MPI_SUCCESS==ierr);
  return nrm_glob;
#endif

  return nrm;
}

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


double hiopVectorRajaPar::onenorm() const
{
  double* local_data_dev = data_dev_;
  RAJA::ReduceSum< hiop_raja_reduce, double > norm(0.0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      norm += std::abs(local_data_dev[i]);
    });
  double nrm1 = static_cast<double>(norm.get());
#ifdef HIOP_USE_MPI
  double nrm1_global;
  int ierr = MPI_Allreduce(&nrm1, &nrm1_global, 1, MPI_DOUBLE, MPI_SUM, comm_); assert(MPI_SUCCESS==ierr);
  return nrm1_global;
#endif
  return nrm1;
}

double hiopVectorRajaPar::onenorm_local() const
{
  double nrm1=0.;
  for(int i=0; i<n_local_; i++)
    nrm1 += fabs(data_[i]);
  return nrm1;
}

// Multiply elements in this by corresponding elements in v_ and store result in
// this
void hiopVectorRajaPar::componentMult( const hiopVector& v_ )
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(v_);
  assert(n_local_==v.n_local_);
  double* self_data = data_dev_;
  double* v_data = v.data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      self_data[i] = self_data[i]*v_data[i];
    });
}

// Divide elements in this by corresponding elements in v_ and store result in
// this. Evidently assuming that no elements in v_ vanish
void hiopVectorRajaPar::componentDiv ( const hiopVector& v_ )
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(v_);
  assert(n_local_==v.n_local_);
  //for(int i=0; i<n_local_; i++) data_[i] /= v.data_[i];
  double* self_data = data_dev_;
  double* v_data = v.data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      self_data[i] = self_data[i]/v_data[i];
    });
}

void hiopVectorRajaPar::componentDiv_w_selectPattern( const hiopVector& v_, const hiopVector& ix_)
{
  const hiopVectorRajaPar& v = dynamic_cast<const hiopVectorRajaPar&>(v_);
  const hiopVectorRajaPar& ix= dynamic_cast<const hiopVectorRajaPar&>(ix_);
#ifdef HIOP_DEEPCHECKS
  assert(v.n_local_==n_local_);
  assert(n_local_==ix.n_local_);
#endif
  //double *s=this->data_, *x=v.data_, *pattern=ix.data_; 
  //for(int i=0; i<n_local_; i++)
  //  if(pattern[i]==0.0) s[i]=0.0;
  //  else                s[i]/=x[i];
  double* self_data = data_dev_;
  double* v_data = v.data_dev_;
  double* ix_data = ix.data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      self_data[i] = ix_data[i]*self_data[i]/v_data[i];
    });
}

// TODO: implement with BLAS call (DSCAL)
void hiopVectorRajaPar::scale(double num)
{
  if(1.0==num) return;
  
  double* self_data = data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      self_data[i] *= num;
    });
}

// TODO: implement with BLAS call (DAXPY)
void hiopVectorRajaPar::axpy(double alpha, const hiopVector& x_)
{
  const hiopVectorRajaPar& x = dynamic_cast<const hiopVectorRajaPar&>(x_);
  
  double* yd = this->data_dev_;
  double* xd = x.data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      // y := a * x + y
      yd[i] = alpha * xd[i] + yd[i];
    });
}

void hiopVectorRajaPar::axzpy(double alpha, const hiopVector& x_, const hiopVector& z_)
{
  const hiopVectorRajaPar& vx = dynamic_cast<const hiopVectorRajaPar&>(x_);
  const hiopVectorRajaPar& vz = dynamic_cast<const hiopVectorRajaPar&>(z_);
#ifdef HIOP_DEEPCHECKS
  assert(vx.n_local_==vz.n_local_);
  assert(   n_local_==vz.n_local_);
#endif  
  double *self_data = data_dev_;
  const double *x_data = vx.local_data_dev_const();
  const double *z_data = vz.local_data_dev_const();
  if(alpha== 1.0) { 
    RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
        RAJA_LAMBDA(RAJA::Index_type i) {
        self_data[i] += x_data[i]*z_data[i];
        });
  } else if(alpha==-1.0) { 
    RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
        RAJA_LAMBDA(RAJA::Index_type i) {
        self_data[i] -= x_data[i]*z_data[i];
        });
  } else { // alpha is neither 1.0 nor -1.0
    // Question: does compiler automatically move alpha to device (if necessary)
    // or is this something we should do manually?
    RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
        RAJA_LAMBDA(RAJA::Index_type i) {
        self_data[i] += alpha*x_data[i]*z_data[i];
        });
  }
}

void hiopVectorRajaPar::axdzpy( double alpha, const hiopVector& x_, const hiopVector& z_)
{
  const hiopVectorRajaPar& vx = dynamic_cast<const hiopVectorRajaPar&>(x_);
  const hiopVectorRajaPar& vz = dynamic_cast<const hiopVectorRajaPar&>(z_);
#ifdef HIOP_DEEPCHECKS
  assert(vx.n_local_==vz.n_local_);
  assert(   n_local_==vz.n_local_);
#endif  
  double *self_data = data_dev_;
  const double *x_data = vx.local_data_dev_const();
  const double *z_data = vz.local_data_dev_const();
  if(alpha== 1.0) { 
    RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
        RAJA_LAMBDA(RAJA::Index_type i) {
        self_data[i] += x_data[i]/z_data[i];
        });
  } else if(alpha==-1.0) { 
    RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
        RAJA_LAMBDA(RAJA::Index_type i) {
        self_data[i] -= x_data[i]/z_data[i];
        });
  } else { // alpha is neither 1.0 nor -1.0
    // Question: does compiler automatically move alpha to device (if necessary)
    // or is this something we should do manually?
    RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
        RAJA_LAMBDA(RAJA::Index_type i) {
        self_data[i] += alpha*x_data[i]/z_data[i];
        });
  }
}

void hiopVectorRajaPar::axdzpy_w_pattern( double alpha, const hiopVector& x_, const hiopVector& z_, const hiopVector& select)
{
  const hiopVectorRajaPar& vx = dynamic_cast<const hiopVectorRajaPar&>(x_);
  const hiopVectorRajaPar& vz = dynamic_cast<const hiopVectorRajaPar&>(z_);
  const hiopVectorRajaPar& sel= dynamic_cast<const hiopVectorRajaPar&>(select);
#ifdef HIOP_DEEPCHECKS
  assert(vx.n_local_==vz.n_local_);
  assert(   n_local_==vz.n_local_);
#endif  
  // this += alpha * x / z   (y+=alpha*x/z)
  double*y = data_dev_;
  const double *x = vx.local_data_dev_const(),
        *z=vz.local_data_dev_const(), 
        *s=sel.local_data_dev_const();
  // for saving some muls
  //
  // Heavily dependent on pattern as well
  /*
  int it;
  if(alpha==1.0) {
      for(it=0;it<n_local_;it++)
          if(s[it]==1.0) y[it] += x[it]/z[it];
  } else 
      if(alpha==-1.0) {
          for(it=0; it<n_local_;it++)
              if(s[it]==1.0) y[it] -= x[it]/z[it];
      } else
          for(it=0; it<n_local_; it++)
              if(s[it]==1.0) y[it] += alpha*x[it]/z[it];
              */
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
          RAJA_LAMBDA(RAJA::Index_type it) {
            y[it] += s[it] * alpha * x[it] / z[it];
          });
}


void hiopVectorRajaPar::addConstant( double c )
{
  double *y = data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      y[i] += c;
    });

}

void  hiopVectorRajaPar::addConstant_w_patternSelect(double c, const hiopVector& ix_)
{
  const hiopVectorRajaPar& ix = dynamic_cast<const hiopVectorRajaPar&>(ix_);
  assert(this->n_local_ == ix.n_local_);
  double *s = data_dev_;
  const double *sel = ix.local_data_dev_const();
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      s[i] += sel[i]*c;
    });
}

void hiopVectorRajaPar::min( double& /* m */, int& /* index */) const
{
  assert(false && "not implemented");
}

// TODO: implement with BLAS call (DSCAL)
void hiopVectorRajaPar::negate()
{
  double* s = this->data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      s[i] *= -1;
    });
}

void hiopVectorRajaPar::invert()
{
  //for(int i=0; i<n_local_; i++) {
#ifdef HIOP_DEEPCHECKS
  //  if(fabs(data_[i])<1e-35) assert(false);
#endif
  //  data_[i]=1./data_[i];
  //}
  double *s = data_dev_;
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
#ifdef HIOP_DEEPCHECKS
      if (std::abs(s[i])<1e-35) assert(false);
#endif
      s[i] = 1.0/s[i];
      });
}

/// Sum all select[i]*log(this[i]), select[i] = 0,1
double hiopVectorRajaPar::logBarrier(const hiopVector& select) const
{
  //const double* ix_vec = ix.data_;
  //double res = 0.0;
  //for(int i=0; i<n_local_; i++) 
  //  if(ix_vec[i]==1.) 
  //    res += log(data_[i]);
  //return res;
  const hiopVectorRajaPar& ix = dynamic_cast<const hiopVectorRajaPar&>(select);
  assert(this->n_local_ == ix.n_local_);
  double* self_data = data_dev_;
  const double* ix_data = ix.local_data_dev_const();
  RAJA::ReduceSum< hiop_raja_reduce, double > sum(0.0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
		RAJA_LAMBDA(RAJA::Index_type i) {
#ifdef HIOP_DEEPCHECKS
      assert(ix_data[i] == one || ix_data[i] == zero);
#endif
      sum += ix_data[i] * std::log(self_data[i]);
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
void  hiopVectorRajaPar::addLogBarrierGrad(double alpha, const hiopVector& x, const hiopVector& ix)
{
#ifdef HIOP_DEEPCHECKS
  assert(this->n_local_ == dynamic_cast<const hiopVectorRajaPar&>(ix).n_local_);
  assert(this->n_local_ == dynamic_cast<const hiopVectorRajaPar&>( x).n_local_);
#endif
  //const double* ix_vec = dynamic_cast<const hiopVectorRajaPar&>(ix).data_;
  //const double*  x_vec = dynamic_cast<const hiopVectorRajaPar&>( x).data_;

  //for(int i=0; i<n_local_; i++) 
  //  if(ix_vec[i]==1.) 
  //    data_[i] += alpha/x_vec[i];
  double* self_data = data_dev_;
  const double* ix_vec = dynamic_cast<const hiopVectorRajaPar&>(ix).local_data_dev_const();
  const double*  x_vec = dynamic_cast<const hiopVectorRajaPar&>( x).local_data_dev_const();
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
				  RAJA_LAMBDA(RAJA::Index_type i) {
               if (ix_vec[i] == 1.0) self_data[i] += alpha/x_vec[i];
				  });
}


double hiopVectorRajaPar::linearDampingTerm(const hiopVector& ixleft, const hiopVector& ixright,
				   const double& mu, const double& kappa_d) const
{
  //const double* ixl= (dynamic_cast<const hiopVectorRajaPar&>(ixleft)).local_data_const();
  //const double* ixr= (dynamic_cast<const hiopVectorRajaPar&>(ixright)).local_data_const();
#ifdef HIOP_DEEPCHECKS
  assert(n_local_==(dynamic_cast<const hiopVectorRajaPar&>(ixleft) ).n_local_);
  assert(n_local_==(dynamic_cast<const hiopVectorRajaPar&>(ixright) ).n_local_);
#endif
  //double term=0.0;
  //for(long long i=0; i<n_local_; i++) {
  //  if(ixl[i]==1. && ixr[i]==0.) term += data_[i];
  //}
  // TODO: Can conditional be improved?
  const double* ixl= (dynamic_cast<const hiopVectorRajaPar&>(ixleft)).local_data_dev_const();
  const double* ixr= (dynamic_cast<const hiopVectorRajaPar&>(ixright)).local_data_dev_const();
  double* self_data = data_dev_;
  RAJA::ReduceSum< hiop_raja_reduce, double > sum(0.0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
				  RAJA_LAMBDA(RAJA::Index_type i) {
               if (ixl[i] == 1.0 && ixr[i] == 0.0) sum += self_data[i];
				  });
  double term = static_cast<double>(sum.get());
  term *= mu; 
  term *= kappa_d;
  return term;
}

int hiopVectorRajaPar::allPositive()
{
  //int allPos=true, i=0;
  //while(i<n_local_ && allPos) if(data_[i++]<=0) allPos=false;
  double* self_data = data_dev_;
  RAJA::ReduceMin< hiop_raja_reduce, double > min(1.0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
				  RAJA_LAMBDA(RAJA::Index_type i) {
               min.min(self_data[i]);
				  });
  int allPos=1;
  if (min.get() <= 0.0) allPos = 0;

#ifdef HIOP_USE_MPI
  int allPosG;
  int ierr=MPI_Allreduce(&allPos, &allPosG, 1, MPI_INT, MPI_MIN, comm_); assert(MPI_SUCCESS==ierr);
  return allPosG;
#endif
  return allPos;
}

bool hiopVectorRajaPar::projectIntoBounds(const hiopVector& xl_, const hiopVector& ixl_,
				      const hiopVector& xu_, const hiopVector& ixu_,
				      double kappa1, double kappa2)
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorRajaPar&>(xl_) ).n_local_==n_local_);
  assert((dynamic_cast<const hiopVectorRajaPar&>(ixl_)).n_local_==n_local_);
  assert((dynamic_cast<const hiopVectorRajaPar&>(xu_) ).n_local_==n_local_);
  assert((dynamic_cast<const hiopVectorRajaPar&>(ixu_)).n_local_==n_local_);
#endif
  //const double* xl = (dynamic_cast<const hiopVectorRajaPar&>(xl_) ).local_data_const();
  //const double* ixl= (dynamic_cast<const hiopVectorRajaPar&>(ixl_)).local_data_const();
  //const double* xu = (dynamic_cast<const hiopVectorRajaPar&>(xu_) ).local_data_const();
  //const double* ixu= (dynamic_cast<const hiopVectorRajaPar&>(ixu_)).local_data_const();
  //double* x0=data_; 

  //const double small_double = std::numeric_limits<double>::min() * 100;

  //double aux, aux2;
  //for(long long i=0; i<n_local_; i++) {
  //  if(ixl[i]!=0 && ixu[i]!=0) {
  //    if(xl[i]>xu[i]) return false;
  //    aux=kappa2*(xu[i]-xl[i])-small_double;
  //    aux2=xl[i]+fmin(kappa1*fmax(1., fabs(xl[i])),aux);
  //    if(x0[i]<aux2) {
  //      x0[i]=aux2;
  //    } else {
  //      aux2=xu[i]-fmin(kappa1*fmax(1., fabs(xu[i])),aux);
  //      if(x0[i]>aux2) {
  //        x0[i]=aux2;
  //      }
  //    }
#ifdef HIOP_DEEPCHECKS
      //if(x0[i]>xl[i] && x0[i]<xu[i]) {
      //} else {
      //printf("i=%d  x0=%g xl=%g xu=%g\n", i, x0[i], xl[i], xu[i]);
      //}
  //    assert(x0[i]>xl[i] && x0[i]<xu[i] && "this should not happen -> HiOp bug");

#endif
  //  } else {
  //    if(ixl[i]!=0.)
  //      x0[i] = fmax(x0[i], xl[i]+kappa1*fmax(1, fabs(xl[i]))-small_double);
  //    else 
  //      if(ixu[i]!=0)
  //        x0[i] = fmin(x0[i], xu[i]-kappa1*fmax(1, fabs(xu[i]))-small_double);
  //      else { /*nothing for free vars  */ }
  //  }
  //}
  const double* xl = (dynamic_cast<const hiopVectorRajaPar&>(xl_) ).local_data_dev_const();
  const double* ixl= (dynamic_cast<const hiopVectorRajaPar&>(ixl_)).local_data_dev_const();
  const double* xu = (dynamic_cast<const hiopVectorRajaPar&>(xu_) ).local_data_dev_const();
  const double* ixu= (dynamic_cast<const hiopVectorRajaPar&>(ixu_)).local_data_dev_const();
  double* x0=data_dev_; 
  // Perform preliminary check to see of all upper value
  RAJA::ReduceMin< hiop_raja_reduce, double > min(1.0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
      min.min(xu[i]-xl[i]);
      });
  if (min.get() < 0.0) return false;
  const double small_double = std::numeric_limits<double>::min() * 100;

  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
      RAJA_LAMBDA(RAJA::Index_type i) {
      double aux, aux2;
      if(ixl[i]!=0 && ixu[i]!=0) {
        aux=kappa2*(xu[i]-xl[i])-small_double;
        aux2=xl[i]+fmin(kappa1*fmax(1., fabs(xl[i])),aux);
        if(x0[i]<aux2) {
          x0[i]=aux2;
        } else {
          aux2=xu[i]-fmin(kappa1*fmax(1., fabs(xu[i])),aux);
          if(x0[i]>aux2) {
            x0[i]=aux2;
          }
        }
#ifdef HIOP_DEEPCHECKS
        assert(x0[i]>xl[i] && x0[i]<xu[i] && "this should not happen -> HiOp bug");
#endif
      } else {
        if(ixl[i]!=0.)
          x0[i] = fmax(x0[i], xl[i]+kappa1*fmax(1, fabs(xl[i]))-small_double);
        else 
          if(ixu[i]!=0)
            x0[i] = fmin(x0[i], xu[i]-kappa1*fmax(1, fabs(xu[i]))-small_double);
          else { /*nothing for free vars  */ }
      }
      });
  return true;
}

/* max{a\in(0,1]| x+ad >=(1-tau)x} */
double hiopVectorRajaPar::fractionToTheBdry(const hiopVector& dx, const double& tau) const
{
#ifdef HIOP_DEEPCHECKS
  assert((dynamic_cast<const hiopVectorRajaPar&>(dx) ).n_local_==n_local_);
  assert(tau>0);
  assert(tau<1);
#endif

  const double* d = (dynamic_cast<const hiopVectorRajaPar&>(dx) ).local_data_dev_const();
  const double* x = data_dev_;

  RAJA::ReduceMin< hiop_raja_reduce, double > aux(1.0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
    RAJA_LAMBDA(RAJA::Index_type i) {
      if(d[i]>=0)
        return;
#ifdef HIOP_DEEPCHECKS
      assert(x[i]>0);
#endif
      aux.min(-tau*x[i]/d[i]);
    });
  return aux.get();
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

  /*
   * TODO
   *
   * what is sparsity of x, pat
   */
  RAJA::ReduceMin< hiop_raja_reduce, double > aux(1.0);
  RAJA::forall< hiop_raja_exec >( RAJA::RangeSegment(0, n_local_),
          RAJA_LAMBDA(RAJA::Index_type i) {
              if(d[i]>=0) return;
              if(pat[i]==0) return;
#ifdef HIOP_DEEPCHECKS
              assert(x[i]>0);
#endif
              aux.min(-tau*x[i]/d[i]);
          });
  /*
  for(int i=0; i<n_local_; i++) {
    if(d[i]>=0) continue;
    if(pat[i]==0) continue;
#ifdef HIOP_DEEPCHECKS
    assert(x[i]>0);
#endif
    aux = -tau*x[i]/d[i];
    if(aux<alpha) alpha=aux;
  }
  */
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
