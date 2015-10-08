add_executable(${PROJECT_NAME}_parser-test
  parser_test.c
  )
target_link_libraries(${PROJECT_NAME}_parser-test
  ${PROJECT_NAME}_utils
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
