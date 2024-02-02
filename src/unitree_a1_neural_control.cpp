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
  int16_t foot_threshold,
  std::array<float, 12> nominal_joint_position)
{
  model_path_ = filepath;
  nominal_ = nominal_joint_position;
  foot_contact_threshold_ = foot_threshold;
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
}

unitree_a1_legged_msgs::msg::LowCmd UnitreeNeuralControl::modelForward(
  const geometry_msgs::msg::TwistStamped::SharedPtr goal,
  const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg)
{
  // Convert msg to states
  auto state = this->msgToTensor(goal, msg);
  // Copy state to last state for debug purposes
  last_state_ = state;
  // Convert vector to tensor
  auto stateTensor = torch::from_blob(state.data(), {1, static_cast<long>(state.size())});
  // Forward pass
  at::Tensor action = module_.forward({stateTensor}).toTensor();
  // Convert tensor to vector
  std::vector<float> action_vec(action.data_ptr<float>(),
    action.data_ptr<float>() + action.numel());
  // Update last action
  last_action_ = action_vec;
  // Take nominal position and add action
  std::transform(
    nominal_.begin(), nominal_.end(),
    action_vec.begin(), action_vec.begin(),
    [&](double a, double b)
    {return a + (b * scaled_factor_);});
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
  // Imu angular velocity
  tensor.push_back(msg->imu.angular_velocity.x);
  tensor.push_back(msg->imu.angular_velocity.y);
  tensor.push_back(msg->imu.angular_velocity.z);
  // Joint velocities
  this->pushJointVelocities(tensor, msg->motor_state.front_right);
  this->pushJointVelocities(tensor, msg->motor_state.front_left);
  this->pushJointVelocities(tensor, msg->motor_state.rear_right);
  this->pushJointVelocities(tensor, msg->motor_state.rear_left);
  // Goal velocity
  tensor.push_back(goal->twist.linear.x);
  tensor.push_back(goal->twist.linear.y);
  tensor.push_back(goal->twist.angular.z);
  // Convert foot force to contact
  this->convertFootForceToContact(msg->foot_force);
  // Foot contact
  tensor.insert(tensor.end(), foot_contact_.begin(), foot_contact_.end());
  // Gravity vector
  auto gravity_vec =
    this->convertToGravityVector(msg->imu.orientation);
  tensor.insert(tensor.end(), gravity_vec.begin(), gravity_vec.end());
  // Last action
  tensor.insert(tensor.end(), last_action_.begin(), last_action_.end());
  // Cycles since last contact
  this->updateCyclesSinceLastContact();
  tensor.insert(tensor.end(), cycles_since_last_contact_.begin(), cycles_since_last_contact_.end());
  return tensor;
}

unitree_a1_legged_msgs::msg::LowCmd UnitreeNeuralControl::actionToMsg(
  const std::vector<float> & action)
{
  (void)action;
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
std::vector<float> UnitreeNeuralControl::convertToGravityVector(
  const geometry_msgs::msg::Quaternion & orientation)
{
  Quaternionf imu_orientation(orientation.w, orientation.x, orientation.y, orientation.z);
  // Define the gravity vector in world frame (assuming it's along -z)
  Vector3f gravity_world(0.0, 0.0, -1.0);
  // Rotate the gravity vector to the sensor frame
  Vector3f gravity_sensor = imu_orientation * gravity_world;
  gravity_sensor.normalize();

  return {static_cast<float>(gravity_sensor.x()),
    static_cast<float>(gravity_sensor.y()),
    static_cast<float>(gravity_sensor.z())};
}
void UnitreeNeuralControl::convertFootForceToContact(
  const unitree_a1_legged_msgs::msg::FootForceState & foot)
{
  auto convertFootForce = [&](const int16_t force)
    {
      return (force < foot_contact_threshold_) ? 0.0f : 1.0f;
    };
  foot_contact_[FL] = convertFootForce(foot.front_left);
  foot_contact_[FR] = convertFootForce(foot.front_right);
  foot_contact_[RL] = convertFootForce(foot.rear_right);
  foot_contact_[RR] = convertFootForce(foot.rear_left);
}

void UnitreeNeuralControl::updateCyclesSinceLastContact()
{
  for (size_t i = 0; i < foot_contact_.size(); i++) {
    if (foot_contact_[i] == 1.0f) {
      cycles_since_last_contact_[i] = 0.0f;
    } else {
      cycles_since_last_contact_[i] += 1.0f;
    }
  }
}

void UnitreeNeuralControl::initControlParams(unitree_a1_legged_msgs::msg::LowCmd & cmd)
{
  cmd.common.mode = 0x0A;
  cmd.common.kp = 50.0;
  cmd.common.kd = 4.0;
  // todo add common msg for different joints
}
}  // namespace unitree_a1_neural_control
