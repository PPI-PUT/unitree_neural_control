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
#include "unitree_a1_neural_control/unitree_a1_neural_control.hpp"
#include <std_srvs/srv/trigger.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

namespace unitree_a1_neural_control
{
using UnitreeNeuralControlPtr = std::unique_ptr<unitree_a1_neural_control::UnitreeNeuralControl>;
using LowState = unitree_a1_legged_msgs::msg::LowState;
using Imu = sensor_msgs::msg::Imu;
using LowCmd = unitree_a1_legged_msgs::msg::LowCmd;
using TwistStamped = unitree_a1_legged_msgs::msg::TwistStamped;
using Trigger = std_srvs::srv::Trigger;
using DebugMsg = unitree_a1_legged_msgs::msg::DebugDoubleArray;
using SyncPolicy = message_filters::sync_policies::ApproximateTime<Imu, LowState>;
using Synchronizer = message_filters::Synchronizer<SyncPolicy>;
using SubscriberLowState = message_filters::Subscriber<LowState>;
using SubscriberImu = message_filters::Subscriber<Imu>;
using WrenchStamped = geometry_msgs::msg::WrenchStamped;
using ControllerType = unitree_a1_legged_msgs::msg::ControllerType;

using namespace std::placeholders;

class UNITREE_A1_NEURAL_CONTROL_PUBLIC UnitreeNeuralControlNode : public rclcpp::Node
{
public:
  explicit UnitreeNeuralControlNode(const rclcpp::NodeOptions & options);

private:
  UnitreeNeuralControlPtr controller_{nullptr};
  double max_age_cmd_vel_;
  std::array<float, 12> nominal_joint_position_;
  TwistStamped::SharedPtr msg_goal_;
  LowState::SharedPtr msg_state_;
  Imu::SharedPtr msg_imu_;
  std::mutex state_mutex_;
  // Subscribers and publishers
  rclcpp::Subscription<TwistStamped>::SharedPtr cmd_vel_;
  std::shared_ptr<SubscriberImu> imu_sub_;
  std::shared_ptr<SubscriberLowState> state_sub_;
  std::shared_ptr<Synchronizer> sync_;
  rclcpp::TimerBase::SharedPtr control_loop_;
  rclcpp::Publisher<LowCmd>::SharedPtr cmd_;
  rclcpp::Service<Trigger>::SharedPtr reset_;
  void imuStateCallback(Imu::SharedPtr imu, LowState::SharedPtr state);
  void cmdVelCallback(TwistStamped::SharedPtr msg);
  void controlLoop();
  void resetCallback(
    const std::shared_ptr<Trigger::Request> request,
    std::shared_ptr<Trigger::Response> response);
  // Debug
  bool debug_;
  bool publish_debug_;
  rclcpp::Publisher<DebugMsg>::SharedPtr debug_tensor_;
  rclcpp::Publisher<DebugMsg>::SharedPtr debug_action_;
  rclcpp::Publisher<WrenchStamped>::SharedPtr debug_wrench_;
  rclcpp::Publisher<WrenchStamped>::SharedPtr debug_foot_contact_rl_;
  rclcpp::Publisher<WrenchStamped>::SharedPtr debug_foot_contact_rr_;
  rclcpp::Publisher<WrenchStamped>::SharedPtr debug_foot_contact_fl_;
  rclcpp::Publisher<WrenchStamped>::SharedPtr debug_foot_contact_fr_;
  void publishDebugMsg();
};
}  // namespace unitree_a1_neural_control

#endif  // UNITREE_A1_NEURAL_CONTROL__UNITREE_A1_NEURAL_CONTROL_NODE_HPP_
