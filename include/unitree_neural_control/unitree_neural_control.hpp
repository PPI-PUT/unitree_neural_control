#include "Eigen/Dense"
#include <vector>
#include <string>
#include <algorithm>

#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
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
                                                         const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg);
        void setFootContactThreshold(int16_t threshold);
        int16_t getFootContactThreshold();

    private:
        int16_t foot_contact_threshold_;
        std::vector<float> last_action_;
        std::vector<float> foot_contact_;
        std::vector<float> cycles_since_last_contact_;
        std::vector<float> last_tick_;
        std::vector<float> msgToTensor(const geometry_msgs::msg::TwistStamped::SharedPtr goal,
                                       const unitree_a1_legged_msgs::msg::LowState::SharedPtr msg);
        std::vector<float> convertToGravityVector(const geometry_msgs::msg::Quaternion &orientation,
                                                  const geometry_msgs::msg::Vector3 &linear_acceleration);
        unitree_a1_legged_msgs::msg::LowCmd actionToMsg(const std::vector<float> &action);
        void pushJointPositions(std::vector<float> &tensor,
                                const unitree_a1_legged_msgs::msg::LegState &joint);
        void pushJointVelocities(std::vector<float> &tensor,
                                 const unitree_a1_legged_msgs::msg::LegState &joint);
        void convertFootForceToContact(unitree_a1_legged_msgs::msg::FootForceState &foot);
        void updateCyclesSinceLastContact();
        void updateCyclesSinceLastContact(uint32_t tick);
    };
}