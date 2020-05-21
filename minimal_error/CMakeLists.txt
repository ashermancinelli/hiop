
# Build vector test
add_executable(timesMatTransError timesMatTransError.cpp)
target_link_libraries(timesMatTransError PRIVATE hiop hiop_warnings hiop_options)
target_include_directories(timesMatTransError PRIVATE ${PROJECT_BINARY_DIR})
