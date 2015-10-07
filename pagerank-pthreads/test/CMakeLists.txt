add_executable(${PROJECT_NAME}_parser-test
  parser_test.c
  )
target_link_libraries(${PROJECT_NAME}_parser-test
  ${PROJECT_NAME}_utils
  ${PROJECT_NAME}_parser
  )
