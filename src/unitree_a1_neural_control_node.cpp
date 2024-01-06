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
  this->declare_parameter(
    "model_path",
    std::string("/home/mackop/inttention_ws/policy_network_trained.pt"));
  std::string model_path;
  this->get_parameter("model_path", model_path);
  // Controller
  RCLCPP_INFO(this->get_logger(), "Loading model: '%s'", model_path.c_str());
  controller_ = std::make_unique<UnitreeNeuralControl>(
    model_path,
    nominal_joint_position_);

  msg_goal_ = std::make_shared<TwistStamped>();
  msg_state_ = std::make_shared<LowState>();
  // Subscribers and publishers
  state_ = this->create_subscription<LowState>(
    "~/input/state", 1,
    std::bind(&UnitreeNeuralControlNode::stateCallback, this, _1));
  cmd_vel_ = this->create_subscription<TwistStamped>(
    "~/input/cmd_vel", 1,
    std::bind(&UnitreeNeuralControlNode::cmdVelCallback, this, _1));
  cmd_ = this->create_publisher<LowCmd>("~/output/command", 1);
  control_loop_ =
    this->create_wall_timer(
    std::chrono::milliseconds(20),
    std::bind(&UnitreeNeuralControlNode::controlLoop, this));
  // Service
  reset_ = this->create_service<Trigger>(
    "~/service/reset",
    std::bind(
      &UnitreeNeuralControlNode::resetCallback, this, _1, _2));
  // Debug
  debug_ = false;
  debug_tensor_ = this->create_publisher<DebugMsg>("~/debug/tensor", 1);
  debug_action_ = this->create_publisher<DebugMsg>("~/debug/action", 1);
}

void UnitreeNeuralControlNode::controlLoop()
{
  auto cmd = controller_->modelForward(msg_goal_, msg_state_);
  auto timestamp = this->now();
  cmd.header.stamp = timestamp;
  cmd_->publish(cmd);
  // Debug
  RCLCPP_INFO(this->get_logger(), "Debug: %d", debug_);
  if (!debug_) {
    return;
  }
  std::vector<float> input, output;
  controller_->getInputAndOutput(input, output);
  auto tensor_msg = DebugMsg();
  tensor_msg.header.stamp = timestamp;
  tensor_msg.dim = {1, static_cast<uint8_t>(input.size())};
  tensor_msg.data = input;
  debug_tensor_->publish(tensor_msg);
  tensor_msg.header.stamp = timestamp;
  tensor_msg.dim = {1, static_cast<uint8_t>(output.size())};
  tensor_msg.data = output;
  debug_action_->publish(tensor_msg);

}

void UnitreeNeuralControlNode::stateCallback(LowState::SharedPtr msg)
{
  msg_state_ = msg;
}

void UnitreeNeuralControlNode::cmdVelCallback(TwistStamped::SharedPtr msg)
{
  msg_goal_ = msg;
}

void UnitreeNeuralControlNode::resetCallback(
  const std::shared_ptr<Trigger::Request> request,
  std::shared_ptr<Trigger::Response> response)
{
  (void) request; // unused
  controller_->resetController();
  response->success = true;
  debug_ = true;
}


}  // namespace unitree_a1_neural_control

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(unitree_a1_neural_control::UnitreeNeuralControlNode)
