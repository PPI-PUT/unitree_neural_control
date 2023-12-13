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

using Vector3f = Eigen::Vector3f;
using Quaternionf = Eigen::Quaternionf;

namespace unitree_neural_control
{
    constexpr size_t FL = 0;
    constexpr size_t FR = 1;
    constexpr size_t RL = 2;
    constexpr size_t RR = 3;
    constexpr uint8_t PMSM_SERVO_MODE = 0x0A;

    class UnitreeNeuralControl
    {
    public:
        explicit UnitreeNeuralControl();
        void loadModel(const std::string &filename);
        unitree_a1_legged_msgs::msg::LowCmd modelForward(const geometry_msgs::msg::TwistStamped::SharedPtr goal,
                                                         const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg,
                                                         const unitree_a1_legged_msgs::msg::QuadrupedState &nominal);
        void modelForward(const geometry_msgs::msg::TwistStamped::SharedPtr goal,
                          const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg,
                          const unitree_a1_legged_msgs::msg::QuadrupedState &nominal,
                          unitree_a1_legged_msgs::msg::LowCmd &cmd,
                          std_msgs::msg::Float32MultiArray &tensor);
        void setFootContactThreshold(int16_t threshold);
        int16_t getFootContactThreshold();
        bool dummyStand(unitree_a1_legged_msgs::msg::LowState::SharedPtr msg, unitree_a1_legged_msgs::msg::LowCmd &cmd_msg);

    private:
        torch::jit::script::Module module_;
        int16_t foot_contact_threshold_;
        double scaled_factor_ = 0.25;
        std::vector<float> last_action_;
        std::vector<float> foot_contact_;
        std::vector<float> cycles_since_last_contact_;
        std::vector<float> last_tick_;
        unitree_a1_legged_msgs::msg::LegState nominal_joint_positions_;
        std::vector<float> targetPos_ = {0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
                                         0.0, 0.67, -1.3, 0.0, 0.67, -1.3};
        std::vector<float> standPos_ = {0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
                                        0.0, 0.67, -1.3, 0.0, 0.67, -1.3};
        std::vector<float> lastPos_ = std::vector<float>(targetPos_.size(), 0.0);
        float steps_ = 3000.0;
        float percent_ = 0.0;
        int motion_time_ = 1;
        bool init_pose_ = true;
        std::vector<float> msgToTensor(const geometry_msgs::msg::TwistStamped::SharedPtr goal,
                                       const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg,
                                       const unitree_a1_legged_msgs::msg::QuadrupedState &nominal);
        std::vector<float> convertToGravityVector(const geometry_msgs::msg::Quaternion &orientation,
                                                  const geometry_msgs::msg::Vector3 &linear_acceleration);
        unitree_a1_legged_msgs::msg::LowCmd actionToMsg(const unitree_a1_legged_msgs::msg::QuadrupedState &nominal,
                                                        const std::vector<float> &action);
        unitree_a1_legged_msgs::msg::QuadrupedState normalizeState(const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg);
        void pushJointPositions(std::vector<float> &tensor,
                                const unitree_a1_legged_msgs::msg::LegState &nominal,
                                const unitree_a1_legged_msgs::msg::LegState &joint);
        void pushJointVelocities(std::vector<float> &tensor,
                                 const unitree_a1_legged_msgs::msg::LegState &joint);
        void convertFootForceToContact(unitree_a1_legged_msgs::msg::FootForceState &foot);
        void updateCyclesSinceLastContact();
        void updateCyclesSinceLastContact(uint32_t tick);
        void initStandParams(unitree_a1_legged_msgs::msg::LowCmd &cmd_msg);
        void initControlParams(unitree_a1_legged_msgs::msg::LowCmd &cmd_msg);
    };
}