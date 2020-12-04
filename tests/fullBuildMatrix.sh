
export logFile="$BUILDDIR/buildmatrix.log"

logFail()
{
  echo "Build matrix failed in build stage: \
    $1 with cmake arguments: \
    $cmakeOptions" >> $logFile
}

buildMatrix()
{
  [[ -f $logFile ]] && rm $logFile
  touch $logFile

  baseCmakeOptions=" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_TESTS=ON \
    -DHIOP_DEEPCHECKS=ON \
    -DHIOP_NVCC_ARCH=$MY_NVCC_ARCH \
    "

  for rajaOp in '-DHIOP_USE_RAJA='{OFF,"ON -DHIOP_RAJA_DIR=$MY_RAJA_DIR"}; do
    for gpuOp in '-DHIOP_USE_GPU='{OFF,"ON \
      -DHIOP_MAGMA_DIR=$MY_HIOP_MAGMA_DIR"}; do
      for kronRedOp in "-DHIOP_WITH_KRON_REDUCTION="{OFF,"ON \
        -DHIOP_UMFPACK_DIR=$MY_UMFPACK_DIR \
        -DHIOP_METIS_DIR=$MY_METIS_DIR \
        -DHIOP_UMFPACK_DIR=$MY_UMFPACK_DIR"}; do
        for mpiOp in "-DHIOP_USE_MPI="{OFF,ON}; do
          export cmakeOptions="$rajaOp $gpuOp $kronRedOp $mpiOp"
          buildAndTest 1 1
        done
      done
    done
  done

  if [[ $(wc -l $logFile | cut -f1 -d' ') -eq 0 ]]; then
    exit 0
  fi
}

buildAndTest()
{
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
    make -j 8 || logFail buildstage
  fi

  if [[ $doTest -eq 1 ]]; then
    echo
    echo Testing
    echo
    ctest -VV || logFail testStage
  fi
}
