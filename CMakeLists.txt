
cmake_minimum_required(VERSION 3.15)
project(hybridynamics)
include(CTest)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-no-as-needed")


find_package (Eigen3 3.3.7 REQUIRED)
find_package(pybind11 REQUIRED) # for pybind11

message(STATUS "Eigen3 : " ${EIGEN3_INCLUDE_DIR}/Eigen)
message(STATUS "current : " ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(-DMJSIMULATE_STATIC) # use static =,solve the "error LNK2001/LNK2019" in windows


include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
)


set(CONTROLLER_SRC
  simulate/robotmodel.cpp
  simulate/trajectory.cpp
)

set_source_files_properties(${CONTROLLER_SRC} PROPERTIES COMPILE_FLAGS "-fPIC")

add_library(franka_emika_lib_py
    ${CONTROLLER_SRC}
)

pybind11_add_module(controller simulate/controller.cpp)
target_link_libraries(controller
    PUBLIC
    Eigen3::Eigen
    franka_emika_lib_py
    -lrbdl
    -lrbdl_urdfreader    
)



