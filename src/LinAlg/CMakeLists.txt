set(LinAlg_SRC 
  hiopVector.cpp
  hiopMatrix.cpp
  hiopMatrixSparseTripletStorage.cpp
  hiopMatrixSparseTriplet.cpp 
  hiopMatrixComplexSparseTriplet.cpp
)


if(HIOP_USE_UMPIRE)
  set(LinAlg_SRC ${LinAlg_SRC} hiopVectorRajaPar.cpp)
  include_directories(${UMPIRE_INCLUDE_DIRS} ${camp_INSTALL_PREFIX}/include)
endif()
message("Include dirs: ${UMPIRE_INCLUDE_DIRS} ${camp_INSTALL_PREFIX}/include")

add_library(hiopLinAlg OBJECT ${LinAlg_SRC})

target_compile_options(hiopLinAlg PRIVATE "-fopenmp") 

target_link_libraries(hiopLinAlg
    PUBLIC
        hiop_warnings_1
        hiop_options)

add_executable(test_hiopLinAlgComplex.exe test_hiopLinalgComplex.cpp)

target_compile_options(test_hiopLinAlgComplex.exe PRIVATE "-fopenmp") 

target_link_libraries(test_hiopLinAlgComplex.exe
    PRIVATE
        hiopLinAlg
        ${HIOP_MATH_LIBRARIES})

target_link_libraries(test_hiopLinAlgComplex.exe
    PRIVATE
        hiop_warnings_3
        hiop_options)

if(HIOP_USE_UMPIRE)
  target_link_libraries(test_hiopLinAlgComplex.exe PRIVATE umpire RAJA)
endif()

