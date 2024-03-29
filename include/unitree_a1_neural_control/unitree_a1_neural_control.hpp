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

#ifndef UNITREE_A1_NEURAL_CONTROL__UNITREE_A1_NEURAL_CONTROL_HPP_
#define UNITREE_A1_NEURAL_CONTROL__UNITREE_A1_NEURAL_CONTROL_HPP_

#include <cstdint>
#include <Eigen/Dense>
#include <torch/script.h>
#include <vector>
#include <string>
#include <algorithm>

#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <unitree_a1_legged_msgs/msg/low_state.hpp>
#include <unitree_a1_legged_msgs/msg/low_cmd.hpp>
#include <unitree_a1_legged_msgs/msg/leg_state.hpp>
#include <unitree_a1_legged_msgs/msg/foot_force_state.hpp>
#include <unitree_a1_legged_msgs/msg/debug_double_array.hpp>
#include "unitree_a1_neural_control/visibility_control.hpp"

using Vector3f = Eigen::Vector3f;
using Quaternionf = Eigen::Quaternionf;

namespace unitree_a1_neural_control
{
constexpr size_t FL = 0;
constexpr size_t FR = 1;
constexpr size_t RL = 2 ;
constexpr size_t RR = 3;
constexpr size_t FL_cycle = 1;
constexpr size_t FR_cycle = 0;
constexpr size_t RL_cycle = 3;
constexpr size_t RR_cycle = 2;
constexpr uint8_t PMSM_SERVO_MODE = 0x0A;

class UNITREE_A1_NEURAL_CONTROL_PUBLIC UnitreeNeuralControl
{
public:
  UnitreeNeuralControl(
    const std::string & filepath, int16_t foot_threshold, std::array<float,
    12> nominal_joint_position);
  unitree_a1_legged_msgs::msg::LowCmd modelForward(
    const geometry_msgs::msg::TwistStamped::SharedPtr goal,
    const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg);
  unitree_a1_legged_msgs::msg::LowCmd modelForward(
    const geometry_msgs::msg::TwistStamped::SharedPtr goal,
    const sensor_msgs::msg::Imu::SharedPtr imu,
    const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg);
  void setFootContactThreshold(int16_t threshold);
  int16_t getFootContactThreshold() const;
  void getInputAndOutput(std::vector<float> & input, std::vector<float> & output);
  void resetController();
  void setGains(double kp, double kd);

private:
  std::string model_path_;
  torch::jit::script::Module module_;
  double scaled_factor_ = 0.25;
  double kp_ = 50.0;
  double kd_ = 4.0;
  int16_t foot_contact_threshold_;
  std::array<float, 12> nominal_;
  std::array<float, 4> foot_contact_;
  std::array<float, 4> cycles_since_last_contact_;
  std::vector<float> last_action_;
  std::vector<float> last_state_;
  std::vector<float> msgToTensor(
    const geometry_msgs::msg::TwistStamped::SharedPtr goal,
    const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg);
  std::vector<float> msgToTensor(
    const geometry_msgs::msg::TwistStamped::SharedPtr goal,
    const sensor_msgs::msg::Imu::SharedPtr imu,
    const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg);
  unitree_a1_legged_msgs::msg::LowCmd actionToMsg(const std::vector<float> & action);
  std::vector<float> convertToGravityVector(
    const geometry_msgs::msg::Quaternion & orientation);
  unitree_a1_legged_msgs::msg::QuadrupedState normalizeState(
    const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg);
  std::array<float, 12> pushJointPositions(
    const unitree_a1_legged_msgs::msg::QuadrupedState & joint);
  void pushJointVelocities(
    std::vector<float> & tensor,
    const unitree_a1_legged_msgs::msg::LegState & joint);
  void loadModel();
  void convertFootForceToContact(const unitree_a1_legged_msgs::msg::FootForceState & foot);
  void updateCyclesSinceLastContact();
  void initValues();
  void initControlParams(unitree_a1_legged_msgs::msg::LowCmd & cmd_msg);
};

}  // namespace unitree_a1_neural_control

#endif  // UNITREE_A1_NEURAL_CONTROL__UNITREE_A1_NEURAL_CONTROL_HPP_
