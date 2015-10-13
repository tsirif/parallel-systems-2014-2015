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
add_test(NAME test_parser
  COMMAND ${PROJECT_NAME}_parser-test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

add_executable(${PROJECT_NAME}_pagerank_single_float-test
  pagerank_single_test.c
  )
target_link_libraries(${PROJECT_NAME}_pagerank_single_float-test
  ${PROJECT_NAME}_parser
  ${PROJECT_NAME}_pagerank_single_float
  )
add_custom_command(TARGET ${PROJECT_NAME}_pagerank_single_float-test
  POST_BUILD
  COMMAND ${PROJECT_NAME}_pagerank_single_float-test
  DEPENDS example-matrix.txt example-matrix2.txt
  COMMENT "Executing Unit Test ${PROJECT_NAME}_pagerank_single_float-test"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
  )
add_test(NAME test_pagerank_single_float
  COMMAND ${PROJECT_NAME}_pagerank_single_float-test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

add_executable(${PROJECT_NAME}_pagerank_single_double-test
  pagerank_single_test.c
  )
target_link_libraries(${PROJECT_NAME}_pagerank_single_double-test
  ${PROJECT_NAME}_parser
  ${PROJECT_NAME}_pagerank_single_double
  )
set_target_properties(${PROJECT_NAME}_pagerank_single_double-test PROPERTIES
  COMPILE_DEFINITIONS "DOUBLE"
  )
add_custom_command(TARGET ${PROJECT_NAME}_pagerank_single_double-test
  POST_BUILD
  COMMAND ${PROJECT_NAME}_pagerank_single_double-test
  DEPENDS example-matrix.txt example-matrix2.txt
  COMMENT "Executing Unit Test ${PROJECT_NAME}_pagerank_single_double-test"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  VERBATIM
  )
add_test(NAME test_pagerank_single_double
  COMMAND ${PROJECT_NAME}_pagerank_single_double-test
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
