set(LinAlg_SRC 
  hiopVector.cpp
  hiopMatrix.cpp
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

# Figure out how to make these play nicely with nvcc
#target_link_libraries(hiopLinAlg
#    PUBLIC
#        hiop_warnings_1
#        hiop_options)

if(HIOP_USE_UMPIRE)
    target_link_libraries(hiopLinAlg PUBLIC umpire RAJA OpenMP::OpenMP_CXX)
endif()



add_executable(test_hiopLinAlgComplex.exe test_hiopLinalgComplex.cpp)

target_link_libraries(test_hiopLinAlgComplex.exe
    PRIVATE
        hiopLinAlg
        ${HIOP_MATH_LIBRARIES})

target_link_libraries(test_hiopLinAlgComplex.exe
    PRIVATE
        hiop_warnings_3
        hiop_options)

#if(HIOP_USE_UMPIRE)
#  target_link_libraries(test_hiopLinAlgComplex.exe)
#endif()

