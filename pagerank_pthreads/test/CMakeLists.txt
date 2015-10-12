add_executable(${PROJECT_NAME}_parser-test
  parser_test.c
  )
target_link_libraries(${PROJECT_NAME}_parser-test
  ${PROJECT_NAME}_parser
  )
add_custom_command(TARGET ${PROJECT_NAME}_parser-test
  POST_BUILD
  COMMAND ${PROJECT_NAME}_parser-test
  DEPENDS example-matrix.txt
  COMMENT "Executing Unit Test ${PROJECT_NAME}_parser-test"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
  )

add_executable(${PROJECT_NAME}_pagerank_single_float-test
  pagerank_single_test.c
  )
target_link_libraries(${PROJECT_NAME}_pagerank_single_float-test
  ${PROJECT_NAME}_parser
  ${PROJECT_NAME}_pagerank_single_float
  )
add_custom_command(TARGET ${PROJECT_NAME}_parser-test
  POST_BUILD
  COMMAND ${PROJECT_NAME}_pagerank_single_float-test
  DEPENDS example-matrix.txt
  COMMENT "Executing Unit Test ${PROJECT_NAME}_pagerank_single_float-test"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
  )

add_executable(${PROJECT_NAME}_pagerank_single_double-test
  pagerank_single_test.c
  )
target_link_libraries(${PROJECT_NAME}_pagerank_single_double-test
  ${PROJECT_NAME}_parser
  ${PROJECT_NAME}_pagerank_single_double
  )
add_custom_command(TARGET ${PROJECT_NAME}_parser-test
  POST_BUILD
  COMMAND ${PROJECT_NAME}_pagerank_single_double-test
  DEPENDS example-matrix.txt
  COMMENT "Executing Unit Test ${PROJECT_NAME}_pagerank_single_double-test"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
  )
set_target_properties(${PROJECT_NAME}_pagerank_single_double-test PROPERTIES
  COMPILE_DEFINITIONS "DOUBLE"
  )
