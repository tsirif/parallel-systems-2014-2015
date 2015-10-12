cmake_minimum_required(VERSION 2.8.3)
project(pagerank_pthreads C)

find_package(Threads REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME}_utils_float
  src/utils.c
  )

add_library(${PROJECT_NAME}_utils_double
  src/utils.c
  )
set_target_properties(${PROJECT_NAME}_utils_double PROPERTIES
  COMPILE_DEFINITIONS "DOUBLE"
  )

add_library(${PROJECT_NAME}_parser
  src/parser.c
  )

add_library(${PROJECT_NAME}_pagerank_single_float
  src/pagerank_single.c
  )

add_library(${PROJECT_NAME}_pagerank_single_double
  src/pagerank_single.c
  )
set_target_properties(${PROJECT_NAME}_pagerank_single_double PROPERTIES
  COMPILE_DEFINITIONS "DOUBLE"
  )

add_library(${PROJECT_NAME}_pagerank_pthreads_float
  src/pagerank_pthreads.c
  )
target_link_libraries(${PROJECT_NAME}_pagerank_pthreads_float
  ${CMAKE_THREAD_LIBS_INIT}
  ${PROJECT_NAME}_utils_float
  )

add_library(${PROJECT_NAME}_pagerank_pthreads_double
  src/pagerank_pthreads.c
  )
set_target_properties(${PROJECT_NAME}_pagerank_pthreads_double PROPERTIES
  COMPILE_DEFINITIONS "DOUBLE"
  )
target_link_libraries(${PROJECT_NAME}_pagerank_pthreads_double
  ${CMAKE_THREAD_LIBS_INIT}
  ${PROJECT_NAME}_utils_double
  )

add_executable(pagerank_single_float
  src/main.c
  )
target_link_libraries(pagerank_single_float
  ${PROJECT_NAME}_utils_float
  ${PROJECT_NAME}_parser
  ${PROJECT_NAME}_pagerank_single_float
  )

add_executable(pagerank_single_double
  src/main.c
  )
target_link_libraries(pagerank_single_double
  ${PROJECT_NAME}_utils_double
  ${PROJECT_NAME}_parser
  ${PROJECT_NAME}_pagerank_single_double
  )
set_target_properties(pagerank_single_double PROPERTIES
  COMPILE_DEFINITIONS "DOUBLE"
  )

add_executable(pagerank_pthreads_float
  src/main.c
  )
target_link_libraries(pagerank_pthreads_float
  ${PROJECT_NAME}_utils_float
  ${PROJECT_NAME}_parser
  ${PROJECT_NAME}_pagerank_pthreads_float
  )
set_target_properties(pagerank_pthreads_float PROPERTIES
  COMPILE_DEFINITIONS "PTHREADS"
  )

add_executable(pagerank_pthreads_double
  src/main.c
  )
target_link_libraries(pagerank_pthreads_double
  ${PROJECT_NAME}_utils_double
  ${PROJECT_NAME}_parser
  ${PROJECT_NAME}_pagerank_pthreads_double
  )
set_target_properties(pagerank_pthreads_double PROPERTIES
  COMPILE_DEFINITIONS "DOUBLE"
  COMPILE_DEFINITIONS "PTHREADS"
  )

add_subdirectory(test)
