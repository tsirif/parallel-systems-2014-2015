cmake_minimum_required(VERSION 2.8.3)
project(pagerank_pthreads C)

find_package(Threads REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME}_utils
  src/utils.c
  )
add_library(${PROJECT_NAME}_parser
  src/parser.c
  )

add_executable(pagerank_single_float
  src/single-pr.c
  )
target_link_libraries(pagerank_single_float
  ${PROJECT_NAME}_utils
  ${PROJECT_NAME}_parser
  )

add_executable(pagerank_single_double
  src/single-pr.c
  )
target_link_libraries(pagerank_single_double
  ${PROJECT_NAME}_utils
  ${PROJECT_NAME}_parser
  )
set_target_properties(pagerank_single_double PROPERTIES
  COMPILE_DEFINITIONS
  "DOUBLE"
  )

add_subdirectory(test)
