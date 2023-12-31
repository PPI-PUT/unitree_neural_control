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
#include <std_srvs/srv/trigger.hpp>

namespace unitree_a1_neural_control
{
using UnitreeNeuralControlPtr = std::unique_ptr<unitree_a1_neural_control::UnitreeNeuralControl>;
using LowState = unitree_a1_legged_msgs::msg::LowState;
using LowCmd = unitree_a1_legged_msgs::msg::LowCmd;
using TwistStamped = geometry_msgs::msg::TwistStamped;
using Trigger = std_srvs::srv::Trigger;
using DebugMsg = unitree_a1_legged_msgs::msg::DebugDoubleArray;
using namespace std::placeholders;

class UNITREE_A1_NEURAL_CONTROL_PUBLIC UnitreeNeuralControlNode : public rclcpp::Node
{
public:
  explicit UnitreeNeuralControlNode(const rclcpp::NodeOptions & options);

private:
  UnitreeNeuralControlPtr controller_{nullptr};
  std::array<float, 12> nominal_joint_position_;
  TwistStamped::SharedPtr msg_goal_;
  LowState::SharedPtr msg_state_;
  rclcpp::TimerBase::SharedPtr control_loop_;
  rclcpp::Publisher<LowCmd>::SharedPtr cmd_;
  // Debug
  bool debug_;
  rclcpp::Publisher<DebugMsg>::SharedPtr debug_tensor_;
  rclcpp::Publisher<DebugMsg>::SharedPtr debug_action_;
  rclcpp::Subscription<LowState>::SharedPtr state_;
  rclcpp::Subscription<TwistStamped>::SharedPtr cmd_vel_;
  rclcpp::Service<Trigger>::SharedPtr reset_;

  void stateCallback(LowState::SharedPtr msg);
  void cmdVelCallback(TwistStamped::SharedPtr msg);
  void resetCallback(
    const std::shared_ptr<Trigger::Request> request,
    std::shared_ptr<Trigger::Response> response);
  void controlLoop();
};
}  // namespace unitree_a1_neural_control

#endif  // UNITREE_A1_NEURAL_CONTROL__UNITREE_A1_NEURAL_CONTROL_NODE_HPP_
