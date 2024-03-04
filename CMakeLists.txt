cmake_minimum_required(VERSION 3.14)
project(unitree_a1_neural_control)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies(REQUIRED
  ${${PROJECT_NAME}_BUILD_DEPENDS}
  ${${PROJECT_NAME}_BUILDTOOL_DEPENDS}
)

add_compile_options(-Wall -Wextra -pedantic -Werror)
add_compile_options(-Wno-missing-field-initializers)
find_package(Eigen3 REQUIRED)
find_path(TORCH_INCLUDE_DIR torch/torch.h
    PATHS $ENV{HOME}/.local/lib/python3.10/site-packages/torch/include
    NO_DEFAULT_PATH
)
find_library(TORCH_LIBRARY torch
    PATHS $ENV{HOME}/.local/lib/python3.10/site-packages/torch/lib
    NO_DEFAULT_PATH
)
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)

set(UNITREE_A1_NEURAL_CONTROL_LIB_SRC
  src/unitree_a1_neural_control.cpp
)

set(UNITREE_A1_NEURAL_CONTROL_LIB_HEADERS
  include/unitree_a1_neural_control/unitree_a1_neural_control.hpp
  include/unitree_a1_neural_control/visibility_control.hpp
)

ament_auto_add_library(${PROJECT_NAME} SHARED
  ${UNITREE_A1_NEURAL_CONTROL_LIB_SRC}
  ${UNITREE_A1_NEURAL_CONTROL_LIB_HEADERS}
)
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
set(UNITREE_A1_NEURAL_CONTROL_NODE_SRC
  src/unitree_a1_neural_control_node.cpp
)

set(UNITREE_A1_NEURAL_CONTROL_NODE_HEADERS
  include/unitree_a1_neural_control/unitree_a1_neural_control_node.hpp
)

ament_auto_add_library(${PROJECT_NAME}_node SHARED
  ${UNITREE_A1_NEURAL_CONTROL_NODE_SRC}
  ${UNITREE_A1_NEURAL_CONTROL_NODE_HEADERS}
)
rclcpp_components_register_node(${PROJECT_NAME}_node
  PLUGIN "unitree_a1_neural_control::UnitreeNeuralControlNode"
  EXECUTABLE ${PROJECT_NAME}_node_exe
)

ament_auto_package(INSTALL_TO_SHARE
  launch
  config
)
