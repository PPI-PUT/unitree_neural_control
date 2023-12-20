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

#ifndef UNITREE_A1_NEURAL_CONTROL__UNITREE_A1_NEURAL_CONTROL_NODE_HPP_
#define UNITREE_A1_NEURAL_CONTROL__UNITREE_A1_NEURAL_CONTROL_NODE_HPP_

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include "unitree_a1_neural_control/unitree_a1_neural_control.hpp"

namespace unitree_a1_neural_control
{
using UnitreeNeuralControlPtr = std::unique_ptr<unitree_a1_neural_control::UnitreeNeuralControl>;

class UNITREE_A1_NEURAL_CONTROL_PUBLIC UnitreeNeuralControlNode : public rclcpp::Node
{
public:
  explicit UnitreeNeuralControlNode(const rclcpp::NodeOptions & options);

private:
  UnitreeNeuralControlPtr controller_{nullptr};
  geometry_msgs::msg::TwistStamped::SharedPtr msg_goal_;
  rclcpp::Subscription<unitree_a1_legged_msgs::msg::LowState>::SharedPtr state_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_;
  rclcpp::Publisher<unitree_a1_legged_msgs::msg::LowCmd>::SharedPtr cmd_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr tensor_;
  void stateCallback(unitree_a1_legged_msgs::msg::LowState::SharedPtr msg);
  void cmdVelCallback(geometry_msgs::msg::TwistStamped::SharedPtr msg);
  std::array<float, 12> nominal_joint_position_;
};
}  // namespace unitree_a1_neural_control

#endif  // UNITREE_A1_NEURAL_CONTROL__UNITREE_A1_NEURAL_CONTROL_NODE_HPP_
