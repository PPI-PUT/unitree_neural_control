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

#include "unitree_a1_neural_control/unitree_a1_neural_control.hpp"

#include <iostream>

namespace unitree_a1_neural_control
{

UnitreeNeuralControl::UnitreeNeuralControl(
  const std::string & filepath,
  std::array<float, 12> nominal_joint_position)
{
  model_path_ = filepath;
  nominal_ = nominal_joint_position;
  last_state_.resize(52);
  last_action_.resize(12);

  this->resetController();
}

void UnitreeNeuralControl::resetController()
{
  this->initValues();
  this->loadModel();
}

void UnitreeNeuralControl::loadModel()
{
  module_ = torch::jit::load(model_path_);
}

void UnitreeNeuralControl::getInputAndOutput(
  std::vector<float> & input,
  std::vector<float> & output)
{
  input = last_state_;
  output = last_action_;
}
void UnitreeNeuralControl::initValues()
{
  std::fill(last_state_.begin(), last_state_.end(), 0.0f);
  std::fill(last_action_.begin(), last_action_.end(), 0.0f);
  std::copy(nominal_.begin(), nominal_.end(), last_action_.begin());
}

unitree_a1_legged_msgs::msg::LowCmd UnitreeNeuralControl::modelForward(
  const geometry_msgs::msg::TwistStamped::SharedPtr goal,
  const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg)
{
  (void)msg;
  auto debug_msg = std::make_shared<unitree_a1_legged_msgs::msg::LowState>();
  debug_msg->motor_state.front_right.hip.q = last_action_[0];
  debug_msg->motor_state.front_right.thigh.q = last_action_[1];
  debug_msg->motor_state.front_right.calf.q = last_action_[2];
  debug_msg->motor_state.front_left.hip.q = last_action_[3];
  debug_msg->motor_state.front_left.thigh.q = last_action_[4];
  debug_msg->motor_state.front_left.calf.q = last_action_[5];
  debug_msg->motor_state.rear_right.hip.q = last_action_[6];
  debug_msg->motor_state.rear_right.thigh.q = last_action_[7];
  debug_msg->motor_state.rear_right.calf.q = last_action_[8];
  debug_msg->motor_state.rear_left.hip.q = last_action_[9];
  debug_msg->motor_state.rear_left.thigh.q = last_action_[10];
  debug_msg->motor_state.rear_left.calf.q = last_action_[11];
  // Convert msg to states
  auto state = this->msgToTensor(goal, debug_msg);
  // Copy state to last state for debug purposes
  last_state_ = state;
  // Convert vector to tensor
  auto stateTensor = torch::from_blob(state.data(), {1, static_cast<long>(state.size())});
  // Forward pass
  at::Tensor action = module_.forward({stateTensor}).toTensor();
  // Convert tensor to vector
  std::vector<float> action_vec(action.data_ptr<float>(),
    action.data_ptr<float>() + action.numel());
  // Take nominal position and add action
  std::transform(
    nominal_.begin(), nominal_.end(),
    action_vec.begin(), action_vec.begin(),
    [&](double a, double b)
    {return a + (b * scaled_factor_);});

  // Debug
  std::transform(
    action_vec.begin(), action_vec.end(),
    last_action_.begin(), last_action_.end(),
    [&](double a, double b)
    {return (1.0f - alpha_) * b + alpha_ * a;});

  // Update last action
  std::copy(action_vec.begin(), action_vec.end(), last_action_.begin());

  // Convert to message
  return this->actionToMsg(action_vec);
}

std::vector<float> UnitreeNeuralControl::msgToTensor(
  const geometry_msgs::msg::TwistStamped::SharedPtr goal,
  const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg)
{
  std::vector<float> tensor;
  // Joint positions
  auto position = this->pushJointPositions(msg->motor_state);
  tensor.insert(tensor.end(), position.begin(), position.end());
  // Joint velocities
  this->pushJointVelocities(tensor, msg->motor_state.front_right);
  this->pushJointVelocities(tensor, msg->motor_state.front_left);
  this->pushJointVelocities(tensor, msg->motor_state.rear_right);
  this->pushJointVelocities(tensor, msg->motor_state.rear_left);
  // Goal velocity
  tensor.push_back(goal->twist.linear.x);
  tensor.push_back(goal->twist.linear.y);
  tensor.push_back(goal->twist.angular.z);
  // Cycles since last contact
  return tensor;
}

unitree_a1_legged_msgs::msg::LowCmd UnitreeNeuralControl::actionToMsg(
  const std::vector<float> & action)
{
  unitree_a1_legged_msgs::msg::LowCmd cmd;
  cmd.motor_cmd.front_right.hip.q = action[0];
  cmd.motor_cmd.front_right.thigh.q = action[1];
  cmd.motor_cmd.front_right.calf.q = action[2];
  cmd.motor_cmd.front_left.hip.q = action[3];
  cmd.motor_cmd.front_left.thigh.q = action[4];
  cmd.motor_cmd.front_left.calf.q = action[5];
  cmd.motor_cmd.rear_right.hip.q = action[6];
  cmd.motor_cmd.rear_right.thigh.q = action[7];
  cmd.motor_cmd.rear_right.calf.q = action[8];
  cmd.motor_cmd.rear_left.hip.q = action[9];
  cmd.motor_cmd.rear_left.thigh.q = action[10];
  cmd.motor_cmd.rear_left.calf.q = action[11];
  this->initControlParams(cmd);
  return cmd;
}
std::array<float, 12> UnitreeNeuralControl::pushJointPositions(
  const unitree_a1_legged_msgs::msg::QuadrupedState & leg)
{
  std::array<float, 12> pose;
  pose[0] = leg.front_right.hip.q - nominal_[0];
  pose[1] = leg.front_right.thigh.q - nominal_[1];
  pose[2] = leg.front_right.calf.q - nominal_[2];

  pose[3] = leg.front_left.hip.q - nominal_[3];
  pose[4] = leg.front_left.thigh.q - nominal_[4];
  pose[5] = leg.front_left.calf.q - nominal_[5];

  pose[6] = leg.rear_right.hip.q - nominal_[6];
  pose[7] = leg.rear_right.thigh.q - nominal_[7];
  pose[8] = leg.rear_right.calf.q - nominal_[8];

  pose[9] = leg.rear_left.hip.q - nominal_[9];
  pose[10] = leg.rear_left.thigh.q - nominal_[10];
  pose[11] = leg.rear_left.calf.q - nominal_[11];
  return pose;
}

void UnitreeNeuralControl::pushJointVelocities(
  std::vector<float> & tensor,
  const unitree_a1_legged_msgs::msg::LegState & joint)
{
  tensor.push_back(joint.hip.dq);
  tensor.push_back(joint.thigh.dq);
  tensor.push_back(joint.calf.dq);
}

void UnitreeNeuralControl::initControlParams(unitree_a1_legged_msgs::msg::LowCmd & cmd)
{
  cmd.common.mode = 0x0A;
  cmd.common.kp = 20.0;
  cmd.common.kd = 0.5;
}
}  // namespace unitree_a1_neural_control
