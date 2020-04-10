set(LinAlg_SRC 
  hiopVector.cpp
  hiopMatrix.cpp
  hiopMatrixComplexDense.cpp
  hiopMatrixSparseTripletStorage.cpp
  hiopMatrixSparseTriplet.cpp 
  hiopMatrixComplexSparseTriplet.cpp
)

if(HIOP_USE_GPU)
  set_source_files_properties(hiopVectorRajaPar.cpp PROPERTIES LANGUAGE CUDA)
endif()

if(HIOP_USE_UMPIRE)
  set(LinAlg_SRC ${LinAlg_SRC} hiopVectorRajaPar.cpp)
  include_directories(${UMPIRE_INCLUDE_DIRS} ${camp_INSTALL_PREFIX}/include)
endif()

add_library(hiopLinAlg OBJECT ${LinAlg_SRC})
target_link_libraries(hiopLinAlg PUBLIC hiopOptimization hiop_math)

if(HIOP_USE_UMPIRE)
  target_link_libraries(hiopLinAlg PUBLIC umpire RAJA OpenMP::OpenMP_CXX)
endif()


if(HIOP_WITH_KRON_REDUCTION)
  add_library(hiopLinAlgZ OBJECT hiopLinSolverUMFPACKZ.cpp)
  target_link_libraries(hiopLinAlgZ PUBLIC hiop_math)
  target_sources(hiopLinAlg PUBLIC $<TARGET_OBJECTS:hiopLinAlgZ>)

  add_executable(test_hiopLinAlgComplex.exe test_hiopLinalgComplex.cpp)
  target_link_libraries(test_hiopLinAlgComplex.exe PRIVATE hiopLinAlg hiop_math)
endif(HIOP_WITH_KRON_REDUCTION)
