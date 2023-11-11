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
                                                                           const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg)
    {
        // Convert msg to states
        auto state = this->msgToTensor(goal, msg);
        // Convert vector to tensor
        auto stateTensor = torch::from_blob(state.data(), {1, static_cast<long>(state.size())});
        // Forward pass
        at::Tensor action = module_.forward({stateTensor}).toTensor();
        // Convert tensor to vector
        std::vector<float> action_vec(action.data_ptr<float>(), action.data_ptr<float>() + action.numel());
        // Update last action
        std::copy(action_vec.begin(), action_vec.end(), last_action_.begin());
        return this->actionToMsg(action_vec);
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
                                                         const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg)
    {
        std::vector<float> tensor;
        // Joint positions
        this->pushJointPositions(tensor, msg->motor_state.front_right);
        this->pushJointPositions(tensor, msg->motor_state.front_left);
        this->pushJointPositions(tensor, msg->motor_state.rear_right);
        this->pushJointPositions(tensor, msg->motor_state.rear_left);
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
        tensor.push_back(goal->twist.linear.z);
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

    unitree_a1_legged_msgs::msg::LowCmd UnitreeNeuralControl::actionToMsg(const std::vector<float> &action)
    {
        unitree_a1_legged_msgs::msg::LowCmd cmd;
        cmd.mode = PMSM_SERVO_MODE;
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
        return cmd;
    }

    void UnitreeNeuralControl::pushJointPositions(std::vector<float> &tensor,
                                                  const unitree_a1_legged_msgs::msg::LegState &joint)
    {
        tensor.push_back(joint.hip.q);
        tensor.push_back(joint.thigh.q);
        tensor.push_back(joint.calf.q);
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
        Vector3f gravity_sensor = imu_orientation.conjugate() * gravity_world;
        // Subtract the linear acceleration to get the gravity vector
        Vector3f linear_acc_vec(linear_acceleration.x, linear_acceleration.y, linear_acceleration.z);
        linear_acc_vec.normalize();
        Vector3f gravity_vec = gravity_sensor - linear_acc_vec;
        return {static_cast<float>(gravity_vec.x()),
                static_cast<float>(gravity_vec.y()),
                static_cast<float>(gravity_vec.z())};
    }
} // namespace unitree_neural_control


