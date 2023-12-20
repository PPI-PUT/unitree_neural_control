// Copyright 2023 Maciej Krupka
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "unitree_a1_neural_control/unitree_a1_neural_control_node.hpp"

namespace unitree_a1_neural_control
{

UnitreeNeuralControlNode::UnitreeNeuralControlNode(const rclcpp::NodeOptions & options)
: Node("unitree_neural_control", options)
{
  // Parameters
  nominal_joint_position_ = {
    -0.1f, 0.8f, -1.5f, 0.1f, 0.8f, -1.5f,
    -0.1f, 1.0f, -1.5f, 0.1f, 1.0f, -1.5f};
  int16_t foot_contact_threshold = this->declare_parameter<int16_t>("foot_contact_threshold", 20);
  this->declare_parameter(
    "model_path",
    std::string("/home/mackop/inttention_ws/policy_network_trained.pt"));
  std::string model_path;
  this->get_parameter("model_path", model_path);
  // Controller
  controller_ = std::make_unique<UnitreeNeuralControl>(foot_contact_threshold, nominal_joint_position_);
  RCLCPP_INFO(this->get_logger(), "Loading model: '%s'", model_path.c_str());
  controller_->loadModel(model_path);

  msg_goal_ = std::make_shared<geometry_msgs::msg::TwistStamped>();
  msg_state_ = std::make_shared<unitree_a1_legged_msgs::msg::LowState>();
  // Subscribers and publishers
  state_ = this->create_subscription<unitree_a1_legged_msgs::msg::LowState>(
    "~/input/state", 1,
    std::bind(&UnitreeNeuralControlNode::stateCallback, this, std::placeholders::_1));
  cmd_vel_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
    "~/input/cmd_vel", 1,
    std::bind(&UnitreeNeuralControlNode::cmdVelCallback, this, std::placeholders::_1));
  cmd_ = this->create_publisher<unitree_a1_legged_msgs::msg::LowCmd>("~/output/command", 1);
  control_loop_ = this->create_wall_timer(std::chrono::milliseconds(20), std::bind(&UnitreeNeuralControlNode::controlLoop, this));
}

void UnitreeNeuralControlNode::controlLoop()
{
  auto cmd = controller_->modelForward(msg_goal_, msg_state_);
  cmd.header.stamp = this->now();
  cmd_->publish(cmd);  
}

void UnitreeNeuralControlNode::stateCallback(unitree_a1_legged_msgs::msg::LowState::SharedPtr msg)
{
  msg_state_ = msg;
}

void UnitreeNeuralControlNode::cmdVelCallback(geometry_msgs::msg::TwistStamped::SharedPtr msg)
{
  msg_goal_ = msg;
}


}  // namespace unitree_a1_neural_control

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(unitree_a1_neural_control::UnitreeNeuralControlNode)
