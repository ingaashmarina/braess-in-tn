cmake_minimum_required(VERSION 3.5...3.26)
project(graph_flow LANGUAGES CXX)

add_subdirectory(extern/pybind11)

set(LPSOLVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/lpsolve")
set(LPSOLVE_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/extern/lpsolve/liblpsolve55.so")
pybind11_add_module(graph_flow graph_flow.cpp)
target_include_directories(graph_flow PRIVATE "${LPSOLVE_DIR}")
target_link_libraries(graph_flow PRIVATE -L"${LPSOLVE_DIR}" -Wl,-rpath="${LPSOLVE_DIR}" lpsolve55)