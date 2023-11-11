cmake_minimum_required(VERSION 3.8)
project(unitree_neural_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(unitree_a1_legged_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

set(dependencies
    unitree_a1_legged_msgs
    rclcpp
    Eigen3
)

add_library(${PROJECT_NAME} SHARED
  src/unitree_neural_control.cpp
  src/unitree_neural_control_node.cpp
)

ament_target_dependencies(${PROJECT_NAME} ${dependencies})

add_executable(control src/main.cpp)
ament_target_dependencies(control ${dependencies})
target_link_libraries(control ${PROJECT_NAME})

install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(
  TARGETS control
  DESTINATION lib/${PROJECT_NAME}
)
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${dependencies})
ament_package()
