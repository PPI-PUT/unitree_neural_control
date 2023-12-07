#include "unitree_neural_control/unitree_neural_control.hpp"

namespace unitree_neural_control
{
    UnitreeNeuralControl::UnitreeNeuralControl()
    {
        foot_contact_threshold_ = 20;
        last_action_.resize(12);
        foot_contact_.resize(4);
        last_tick_.resize(4);
        cycles_since_last_contact_.resize(4);
        std::fill(last_action_.begin(), last_action_.end(), 0.0f);
        std::fill(last_tick_.begin(), last_tick_.end(), 0.0f);
        std::fill(foot_contact_.begin(), foot_contact_.end(), 0.0f);
        std::fill(cycles_since_last_contact_.begin(), cycles_since_last_contact_.end(), 0.0f);
    }

    void UnitreeNeuralControl::loadModel(const std::string &filename)
    {
        module_ = torch::jit::load(filename);
    }

    unitree_a1_legged_msgs::msg::LowCmd UnitreeNeuralControl::modelForward(const geometry_msgs::msg::TwistStamped::SharedPtr goal,
                                                                           const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg,
                                                                           const unitree_a1_legged_msgs::msg::QuadrupedState &nominal)
    {
        // Convert msg to states
        auto state = this->msgToTensor(goal, msg, nominal);
        // Convert vector to tensor
        auto stateTensor = torch::from_blob(state.data(), {1, static_cast<long>(state.size())});
        // Forward pass
        at::Tensor action = module_.forward({stateTensor}).toTensor();
        // Convert tensor to vector
        std::vector<float> action_vec(action.data_ptr<float>(), action.data_ptr<float>() + action.numel());
        // Apply to target
        // Take nominal position and add action
        std::transform(state.begin(), state.begin() + action_vec.size(),
                       action_vec.begin(), action_vec.begin(),
                       [&](double a, double b)
                       { return a + (b * scaled_factor_); });
        // Update last action
        std::copy(action_vec.begin(), action_vec.end(), last_action_.begin());

        return this->actionToMsg(nominal, action_vec);
    }

    void UnitreeNeuralControl::modelForward(const geometry_msgs::msg::TwistStamped::SharedPtr goal,
                                            const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg,
                                            const unitree_a1_legged_msgs::msg::QuadrupedState &nominal,
                                            unitree_a1_legged_msgs::msg::LowCmd &cmd,
                                            std_msgs::msg::Float32MultiArray &tensor)
    {
        // Convert msg to states
        auto state = this->msgToTensor(goal, msg, nominal);
        // Convert vector to tensor
        auto stateTensor = torch::from_blob(state.data(), {1, static_cast<long>(state.size())});
        // Forward pass
        at::Tensor action = module_.forward({stateTensor}).toTensor();
        // Convert tensor to vector
        std::vector<float> action_vec(action.data_ptr<float>(), action.data_ptr<float>() + action.numel());
        // Apply to target
        // Take nominal position and add action
        std::transform(state.begin(), state.begin() + action_vec.size(),
                       action_vec.begin(), action_vec.begin(),
                       [&](double a, double b)
                       { return a + (b * scaled_factor_); });
        // Update last action
        std::copy(action_vec.begin(), action_vec.end(), last_action_.begin());
        // Convert vector to tensor
        tensor.data.resize(state.size());
        std::copy(state.begin(), state.end(), tensor.data.begin());
        // Convert to message
        cmd = this->actionToMsg(nominal, action_vec);
    }

    void UnitreeNeuralControl::setFootContactThreshold(const int16_t threshold)
    {
        foot_contact_threshold_ = threshold;
    }

    int16_t UnitreeNeuralControl::getFootContactThreshold()
    {
        return foot_contact_threshold_;
    }

    std::vector<float> UnitreeNeuralControl::msgToTensor(const geometry_msgs::msg::TwistStamped::SharedPtr goal,
                                                         const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg,
                                                         const unitree_a1_legged_msgs::msg::QuadrupedState &nominal)
    {
        std::vector<float> tensor;
        // Joint positions
        this->pushJointPositions(tensor, nominal.front_right, msg->motor_state.front_right);
        this->pushJointPositions(tensor, nominal.front_left, msg->motor_state.front_left);
        this->pushJointPositions(tensor, nominal.rear_right, msg->motor_state.rear_right);
        this->pushJointPositions(tensor, nominal.rear_left, msg->motor_state.rear_left);
        // Imu orientation
        tensor.push_back(msg->imu.orientation.x);
        tensor.push_back(msg->imu.orientation.y);
        tensor.push_back(msg->imu.orientation.z);
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
        auto gravity_vec = this->convertToGravityVector(msg->imu.orientation, msg->imu.linear_acceleration);
        tensor.insert(tensor.end(), gravity_vec.begin(), gravity_vec.end());
        // Last action
        tensor.insert(tensor.end(), last_action_.begin(), last_action_.end());
        // Cycles since last contact
        this->updateCyclesSinceLastContact();
        tensor.insert(tensor.end(), cycles_since_last_contact_.begin(), cycles_since_last_contact_.end());
        return tensor;
    }

    unitree_a1_legged_msgs::msg::LowCmd UnitreeNeuralControl::actionToMsg(const unitree_a1_legged_msgs::msg::QuadrupedState &nominal,
                                                                          const std::vector<float> &action)
    {
        unitree_a1_legged_msgs::msg::LowCmd cmd;
        cmd.motor_cmd.front_right.hip.q = nominal.front_right.hip.q + action[0];
        cmd.motor_cmd.front_right.thigh.q = nominal.front_right.thigh.q + action[1];
        cmd.motor_cmd.front_right.calf.q = nominal.front_right.calf.q + action[2];
        cmd.motor_cmd.front_left.hip.q = nominal.front_left.hip.q + action[3];
        cmd.motor_cmd.front_left.thigh.q = nominal.front_left.thigh.q + action[4];
        cmd.motor_cmd.front_left.calf.q = nominal.front_left.calf.q + action[5];
        cmd.motor_cmd.rear_right.hip.q = nominal.rear_right.hip.q + action[6];
        cmd.motor_cmd.rear_right.thigh.q = nominal.rear_right.thigh.q + action[7];
        cmd.motor_cmd.rear_right.calf.q = nominal.rear_right.calf.q + action[8];
        cmd.motor_cmd.rear_left.hip.q = nominal.rear_left.hip.q + action[9];
        cmd.motor_cmd.rear_left.thigh.q = nominal.rear_left.thigh.q + action[10];
        cmd.motor_cmd.rear_left.calf.q = nominal.rear_left.calf.q + action[11];
        this->initControlParams(cmd);
        return cmd;
    }
    void UnitreeNeuralControl::pushJointPositions(std::vector<float> &tensor,
                                                  const unitree_a1_legged_msgs::msg::LegState &nominal,
                                                  const unitree_a1_legged_msgs::msg::LegState &joint)
    {
        tensor.push_back(nominal.hip.q - joint.hip.q);
        tensor.push_back(nominal.thigh.q - joint.thigh.q);
        tensor.push_back(nominal.calf.q - joint.calf.q);
    }

    void UnitreeNeuralControl::pushJointVelocities(std::vector<float> &tensor,
                                                   const unitree_a1_legged_msgs::msg::LegState &joint)
    {
        tensor.push_back(joint.hip.dq);
        tensor.push_back(joint.thigh.dq);
        tensor.push_back(joint.calf.dq);
    }

    void UnitreeNeuralControl::convertFootForceToContact(unitree_a1_legged_msgs::msg::FootForceState &foot)
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
        for (size_t i = 0; i < foot_contact_.size(); i++)
        {
            if (foot_contact_[i] == 1.0f)
            {
                cycles_since_last_contact_[i] = 0.0f;
            }
            else
            {
                cycles_since_last_contact_[i] += 1.0f;
            }
        }
    }

    void UnitreeNeuralControl::updateCyclesSinceLastContact(uint32_t tick)
    {
        auto tick_ms = static_cast<float>(tick) / 1000.0f;
        for (size_t i = 0; i < foot_contact_.size(); i++)
        {
            if (foot_contact_[i] == 1.0f)
            {
                last_tick_[i] = tick_ms;
                cycles_since_last_contact_[i] = 0.0f;
            }
            else
            {
                cycles_since_last_contact_[i] = tick_ms - last_tick_[i];
            }
        }
    }
    std::vector<float> UnitreeNeuralControl::convertToGravityVector(const geometry_msgs::msg::Quaternion &orientation,
                                                                    const geometry_msgs::msg::Vector3 &linear_acceleration)
    {
        Quaternionf imu_orientation(orientation.w, orientation.x, orientation.y, orientation.z);
        // Define the gravity vector in world frame (assuming it's along -z)
        Vector3f gravity_world(0.0, 0.0, -1.0);
        // Rotate the gravity vector to the sensor frame
        Vector3f gravity_sensor = imu_orientation * gravity_world;
        return {static_cast<float>(gravity_sensor.x()),
                static_cast<float>(gravity_sensor.y()),
                static_cast<float>(gravity_sensor.z())};
    }

    void UnitreeNeuralControl::initStandParams(unitree_a1_legged_msgs::msg::LowCmd &cmd_msg)
    {
        cmd_msg.mode = 0x0A;
        cmd_msg.motor_cmd.front_right.hip.mode = 0x0A;
        cmd_msg.motor_cmd.front_right.hip.kp = 70.0;
        cmd_msg.motor_cmd.front_right.hip.kd = 3.0;
        cmd_msg.motor_cmd.front_left.hip.mode = 0x0A;
        cmd_msg.motor_cmd.front_left.hip.kp = 70.0;
        cmd_msg.motor_cmd.front_left.hip.kd = 3.0;
        cmd_msg.motor_cmd.rear_right.hip.mode = 0x0A;
        cmd_msg.motor_cmd.rear_right.hip.kp = 70.0;
        cmd_msg.motor_cmd.rear_right.hip.kd = 3.0;
        cmd_msg.motor_cmd.rear_left.hip.mode = 0x0A;
        cmd_msg.motor_cmd.rear_left.hip.kp = 70.0;
        cmd_msg.motor_cmd.rear_left.hip.kd = 3.0;

        cmd_msg.motor_cmd.front_right.thigh.mode = 0x0A;
        cmd_msg.motor_cmd.front_right.thigh.kp = 180.0;
        cmd_msg.motor_cmd.front_right.thigh.kd = 8.0;
        cmd_msg.motor_cmd.front_left.thigh.mode = 0x0A;
        cmd_msg.motor_cmd.front_left.thigh.kp = 180.0;
        cmd_msg.motor_cmd.front_left.thigh.kd = 8.0;
        cmd_msg.motor_cmd.rear_right.thigh.mode = 0x0A;
        cmd_msg.motor_cmd.rear_right.thigh.kp = 180.0;
        cmd_msg.motor_cmd.rear_right.thigh.kd = 8.0;
        cmd_msg.motor_cmd.rear_left.thigh.mode = 0x0A;
        cmd_msg.motor_cmd.rear_left.thigh.kp = 180.0;
        cmd_msg.motor_cmd.rear_left.thigh.kd = 8.0;

        cmd_msg.motor_cmd.front_right.calf.mode = 0x0A;
        cmd_msg.motor_cmd.front_right.calf.kp = 300.0;
        cmd_msg.motor_cmd.front_right.calf.kd = 15.0;
        cmd_msg.motor_cmd.front_left.calf.mode = 0x0A;
        cmd_msg.motor_cmd.front_left.calf.kp = 300.0;
        cmd_msg.motor_cmd.front_left.calf.kd = 15.0;
        cmd_msg.motor_cmd.rear_right.calf.mode = 0x0A;
        cmd_msg.motor_cmd.rear_right.calf.kp = 300.0;
        cmd_msg.motor_cmd.rear_right.calf.kd = 15.0;
        cmd_msg.motor_cmd.rear_left.calf.mode = 0x0A;
        cmd_msg.motor_cmd.rear_left.calf.kp = 300.0;
        cmd_msg.motor_cmd.rear_left.calf.kd = 15.0;
    }
    void UnitreeNeuralControl::initControlParams(unitree_a1_legged_msgs::msg::LowCmd &cmd)
    {
        cmd.mode = 0x0A;
        cmd.motor_cmd.front_right.hip.mode = 0x0A;
        cmd.motor_cmd.front_right.thigh.mode = 0x0A;
        cmd.motor_cmd.front_right.calf.mode = 0x0A;
        cmd.motor_cmd.front_left.hip.mode = 0x0A;
        cmd.motor_cmd.front_left.thigh.mode = 0x0A;
        cmd.motor_cmd.front_left.calf.mode = 0x0A;
        cmd.motor_cmd.rear_right.hip.mode = 0x0A;
        cmd.motor_cmd.rear_right.thigh.mode = 0x0A;
        cmd.motor_cmd.rear_right.calf.mode = 0x0A;
        cmd.motor_cmd.rear_left.hip.mode = 0x0A;
        cmd.motor_cmd.rear_left.thigh.mode = 0x0A;
        cmd.motor_cmd.rear_left.calf.mode = 0x0A;
        cmd.motor_cmd.front_right.hip.kp = 5.0;
        cmd.motor_cmd.front_right.thigh.kp = 5.0;
        cmd.motor_cmd.front_right.calf.kp = 5.0;
        cmd.motor_cmd.front_left.hip.kp = 5.0;
        cmd.motor_cmd.front_left.thigh.kp = 5.0;
        cmd.motor_cmd.front_left.calf.kp = 5.0;
        cmd.motor_cmd.rear_right.hip.kp = 5.0;
        cmd.motor_cmd.rear_right.thigh.kp = 5.0;
        cmd.motor_cmd.rear_right.calf.kp = 5.0;
        cmd.motor_cmd.rear_left.hip.kp = 5.0;
        cmd.motor_cmd.rear_left.thigh.kp = 5.0;
        cmd.motor_cmd.rear_left.calf.kp = 5.0;
        cmd.motor_cmd.front_right.hip.kd = 1.0;
        cmd.motor_cmd.front_right.thigh.kd = 1.0;
        cmd.motor_cmd.front_right.calf.kd = 1.0;
        cmd.motor_cmd.front_left.hip.kd = 1.0;
        cmd.motor_cmd.front_left.thigh.kd = 1.0;
        cmd.motor_cmd.front_left.calf.kd = 1.0;
        cmd.motor_cmd.rear_right.hip.kd = 1.0;
        cmd.motor_cmd.rear_right.thigh.kd = 1.0;
        cmd.motor_cmd.rear_right.calf.kd = 1.0;
        cmd.motor_cmd.rear_left.hip.kd = 1.0;
        cmd.motor_cmd.rear_left.thigh.kd = 1.0;
        cmd.motor_cmd.rear_left.calf.kd = 1.0;
    }
    bool UnitreeNeuralControl::dummyStand(unitree_a1_legged_msgs::msg::LowState::SharedPtr msg, unitree_a1_legged_msgs::msg::LowCmd &cmd_msg)
    {
        if (init_pose_)
        {
            lastPos_[0] = msg->motor_state.front_right.hip.q;
            lastPos_[1] = msg->motor_state.front_right.thigh.q;
            lastPos_[2] = msg->motor_state.front_right.calf.q;
            lastPos_[3] = msg->motor_state.front_left.hip.q;
            lastPos_[4] = msg->motor_state.front_left.thigh.q;
            lastPos_[5] = msg->motor_state.front_left.calf.q;
            lastPos_[6] = msg->motor_state.rear_right.hip.q;
            lastPos_[7] = msg->motor_state.rear_right.thigh.q;
            lastPos_[8] = msg->motor_state.rear_right.calf.q;
            lastPos_[9] = msg->motor_state.rear_left.hip.q;
            lastPos_[10] = msg->motor_state.rear_left.thigh.q;
            lastPos_[11] = msg->motor_state.rear_left.calf.q;
            init_pose_ = false;
        }
        this->initStandParams(cmd_msg);
        if (motion_time_ < steps_)
        {
            percent_ = static_cast<float>(motion_time_) / steps_;
            auto jointLinearInterpolation = [](double initPos, double targetPos, double rate)
            {
                rate = std::min(std::max(rate, 0.0), 1.0);
                return initPos * (1 - rate) + targetPos * rate;
            };
            for (size_t i = 0; i < targetPos_.size(); i++)
            {
                targetPos_[i] = jointLinearInterpolation(lastPos_[i], standPos_[i], percent_);
            }
            motion_time_++;

            cmd_msg.motor_cmd.front_right.hip.q = targetPos_[0];
            cmd_msg.motor_cmd.front_right.thigh.q = targetPos_[1];
            cmd_msg.motor_cmd.front_right.calf.q = targetPos_[2];
            cmd_msg.motor_cmd.front_left.hip.q = targetPos_[3];
            cmd_msg.motor_cmd.front_left.thigh.q = targetPos_[4];
            cmd_msg.motor_cmd.front_left.calf.q = targetPos_[5];
            cmd_msg.motor_cmd.rear_right.hip.q = targetPos_[6];
            cmd_msg.motor_cmd.rear_right.thigh.q = targetPos_[7];
            cmd_msg.motor_cmd.rear_right.calf.q = targetPos_[8];
            cmd_msg.motor_cmd.rear_left.hip.q = targetPos_[9];
            cmd_msg.motor_cmd.rear_left.thigh.q = targetPos_[10];
            cmd_msg.motor_cmd.rear_left.calf.q = targetPos_[11];
            return true;
        }
        return false;
    }
} // namespace unitree_neural_control
