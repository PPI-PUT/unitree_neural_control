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
  double kp = this->declare_parameter<double>("kp", 50.0);
  double kd = this->declare_parameter<double>("kd", 4.0);
  int16_t foot_contact_threshold = this->declare_parameter<int16_t>("foot_contact_threshold", 20);
  publish_debug_ = this->declare_parameter<bool>("publish_debug", false);
  // Controller
  RCLCPP_INFO(this->get_logger(), "Loading model: '%s'", model_path.c_str());
  controller_ = std::make_unique<UnitreeNeuralControl>(
    model_path,
    foot_contact_threshold,
    nominal_joint_position_);
  controller_->setGains(kp, kd);
  msg_goal_ = std::make_shared<TwistStamped>();
  msg_state_ = std::make_shared<LowState>();
  msg_imu_ = std::make_shared<Imu>();
  // Subscribers and publishers
  rmw_qos_profile_t qos_filter = rmw_qos_profile_default;
  qos_filter.depth = 1;
  qos_filter.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
  qos_filter.durability = RMW_QOS_POLICY_DURABILITY_VOLATILE;
  imu_sub_.reset(new SubscriberImu(this, "~/input/imu", qos_filter));
  state_sub_.reset(new SubscriberLowState(this, "~/input/state", qos_filter));
  sync_.reset(
    new Synchronizer(
      SyncPolicy(2), *imu_sub_, *state_sub_));
  sync_->registerCallback(&UnitreeNeuralControlNode::imuStateCallback, this);
  cmd_vel_ = this->create_subscription<TwistStamped>(
    "~/input/cmd_vel", 1,
    std::bind(&UnitreeNeuralControlNode::cmdVelCallback, this, _1));
  auto qos = rclcpp::QoS(1);
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
  qos.durability_volatile();
  cmd_ = this->create_publisher<LowCmd>("~/output/command", qos);
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
  if (publish_debug_) {
    debug_ = false;
    debug_tensor_ = this->create_publisher<DebugMsg>("~/debug/tensor", 1);
    debug_action_ = this->create_publisher<DebugMsg>("~/debug/action", 1);
    debug_wrench_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>("~/debug/wrench", 1);
    debug_foot_contact_fl_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>(
      "~/debug/foot_contact_fl", 1);
    debug_foot_contact_fr_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>(
      "~/debug/foot_contact_fr", 1);
    debug_foot_contact_rl_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>(
      "~/debug/foot_contact_rl", 1);
    debug_foot_contact_rr_ = this->create_publisher<geometry_msgs::msg::WrenchStamped>(
      "~/debug/foot_contact_rr", 1);
  }

}

void UnitreeNeuralControlNode::controlLoop()
{
  // LowState::SharedPtr local_state;
  // {
  //   std::lock_guard<std::mutex> lock(state_mutex_);
  //   local_state = msg_state_;
  // }
  auto cmd = controller_->modelForward(msg_goal_, msg_imu_, msg_state_);
  cmd.header.stamp = this->now();
  cmd_->publish(cmd);
  if(publish_debug_) {
    publishDebugMsg();
  }
}

void UnitreeNeuralControlNode::imuStateCallback(Imu::SharedPtr imu, LowState::SharedPtr msg)
{
  // std::lock_guard<std::mutex> lock(state_mutex_);
  msg_state_ = msg;
  msg_imu_ = imu;
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
  if (publish_debug_) {
    debug_ = true;
  }

}

void UnitreeNeuralControlNode::publishDebugMsg()
{
  auto timestamp = this->now();
  std::vector<float> input, output;
  controller_->getInputAndOutput(input, output);
  auto wrench_msg = geometry_msgs::msg::WrenchStamped();
  wrench_msg.header.frame_id = "imu_link";
  wrench_msg.wrench.force.z = input[30];
  debug_foot_contact_fl_->publish(wrench_msg);
  wrench_msg.header.frame_id = "FR_foot";
  wrench_msg.wrench.force.z = input[31];
  debug_foot_contact_fr_->publish(wrench_msg);
  wrench_msg.header.frame_id = "RL_foot";
  wrench_msg.wrench.force.z = input[32];
  debug_foot_contact_rl_->publish(wrench_msg);
  wrench_msg.header.frame_id = "RR_foot";
  wrench_msg.wrench.force.z = input[33];
  debug_foot_contact_rr_->publish(wrench_msg);
  // Debug
  if (!debug_) {
    return;
  }
  auto tensor_msg = DebugMsg();
  tensor_msg.header.stamp = timestamp;
  tensor_msg.dim = {1, static_cast<uint8_t>(input.size())};
  tensor_msg.data = input;
  debug_tensor_->publish(tensor_msg);
  tensor_msg.header.stamp = timestamp;
  tensor_msg.dim = {1, static_cast<uint8_t>(output.size())};
  tensor_msg.data = output;
  debug_action_->publish(tensor_msg);
  wrench_msg.header.stamp = timestamp;
  wrench_msg.header.frame_id = "imu_link";
  wrench_msg.wrench.force.x = input[34];
  wrench_msg.wrench.force.y = input[35];
  wrench_msg.wrench.force.z = input[36];
  debug_wrench_->publish(wrench_msg);
}


}  // namespace unitree_a1_neural_control

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(unitree_a1_neural_control::UnitreeNeuralControlNode)
