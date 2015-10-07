cmake_minimum_required(VERSION 2.8.3)
project(pagerank-pthreads C)

find_package(Threads REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME}_utils
  src/utils.c
  )
add_library(${PROJECT_NAME}_parser
  src/parser.c
  )
