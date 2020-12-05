
# Number of failures encountered
export failures=0

# We don't want verbose output when running all possible builds, the output
# becomes very difficult to read.
set +x

# If any build configurations fail to build, they will be written here
# and reported after all build configurations have run
export logFile="$BASE_PATH/buildmatrix.log"

# Error codes for specific cases
export success=0
export buildError=1
export testError=2
export strerr=('Success' 'BuildError' 'TestError')

# Options used to configure CMake (set in buildMatrix function)
export cmakeOptions=""

# Logs the output of a given run to the logfile
logRun()
{
  echo "${strerr[$1]}:$cmakeOptions" | tee --append $logFile
  if [[ $1 -ne $success ]]; then
    ((failures++))
  fi
}

reportRuns()
{
  if [[ $(wc -l $logFile | cut -f1 -d' ') -eq 0 ]] && [[ $failures -eq 0 ]]; then
    echo
    echo No failures detected.
    echo
    return 0
  else
    echo
    echo "$failures failures detected"
    echo
    echo Logfile:
    return 1
  fi
  cat $logFile
}

# Iterates through every configuration of CMake variables and call the 
# _buildAndTest_ function to ensure the configuration is functional
buildMatrix()
{
  [[ -f $logFile ]] && rm $logFile
  touch $logFile

  baseCmakeOptions=" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_TESTS=ON \
    -DHIOP_DEEPCHECKS=ON \
    "

  rajaOps=('-DHIOP_USE_RAJA=OFF' "-DHIOP_USE_RAJA=ON \
    -DHIOP_RAJA_DIR=$MY_RAJA_DIR")
  gpuOps=('-DHIOP_USE_GPU=OFF' "-DHIOP_USE_GPU=ON \
    -DHIOP_NVCC_ARCH=$MY_NVCC_ARCH \
    -DHIOP_MAGMA_DIR=$MY_HIOP_MAGMA_DIR")
  kronRedOps=('-DHIOP_WITH_KRON_REDUCTION=OFF' "-DHIOP_WITH_KRON_REDUCTION=ON \
    -DHIOP_METIS_DIR=$MY_METIS_DIR \
    -DHIOP_UMFPACK_DIR=$MY_UMFPACK_DIR")
  mpiOps=('-DHIOP_USE_MPI=ON' '-DHIOP_USE_MPI=OFF')

  # STRUMPACK is not yet installed on our target platforms, so this will not
  # be a part of the build matrix yet.
  sparseOps=('-DHIOP_SPARSE=OFF' "-DHIOP_SPARSE=ON \
    -DHIOP_USE_STRUMPACK=ON \
    -DHIOP_STRUMPACK_DIR=$MY_STRUMPACK_DIR \
    -DHIOP_METIS_DIR=$MY_METIS_DIR \
    -DHIOP_COINHSL_DIR=$MY_COINHSL_DIR")

  for rajaOp in ${rajaOps[@]}; do
    for gpuOp in ${gpuOps[@]}; do
      for kronRedOp in ${kronRedOps[@]}; do
        for mpiOp in ${mpiOps[@]}; do
          export cmakeOptions="$baseCmakeOptions $rajaOp $gpuOp $kronRedOp $mpiOp"
          buildAndTest 1 0
          logRun $?
        done
      done
    done
  done

  reportRuns
}

buildAndTest()
{
  echo
  echo CMake Options:
  echo
  echo $cmakeOptions
  echo
  doBuild=${1:-1}
  doTest=${2:-1}

  echo
  echo Configuring
  echo

  rm -rf $BUILDDIR
  mkdir -p $BUILDDIR
  pushd $BUILDDIR
  cmake $cmakeOptions ..
  popd

  if [[ $doBuild -eq 1 ]]; then
    echo
    echo Building
    echo
    pushd $BUILDDIR
    $MAKE_CMD || {
      return $buildError
    }
    popd
  fi

  if [[ $doTest -eq 1 ]]; then
    echo
    echo Testing
    echo
    pushd $BUILDDIR
    $CTEST_CMD || {
      return $testError
    }
    popd
  fi
  return $success
}
