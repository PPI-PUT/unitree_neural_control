#include "unitree_neural_control/unitree_neural_control_node.hpp"

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  rclcpp::spin(std::make_shared<unitree_neural_control::UnitreeNeuralControlNode>(options));
  rclcpp::shutdown();

  return 0;
}