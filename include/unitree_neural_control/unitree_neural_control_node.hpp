#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>

#include "unitree_neural_control/unitree_neural_control.hpp"

namespace unitree_neural_control
{
    class UnitreeNeuralControlNode : public rclcpp::Node
    {
    public:
        explicit UnitreeNeuralControlNode(const rclcpp::NodeOptions &options);

    private:
        std::unique_ptr<UnitreeNeuralControl> controller_;
        geometry_msgs::msg::TwistStamped::SharedPtr msg_goal_;
        rclcpp::Subscription<unitree_a1_legged_msgs::msg::LowState>::SharedPtr state_;
        rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_;
        rclcpp::Publisher<unitree_a1_legged_msgs::msg::LowCmd>::SharedPtr cmd_;
        void stateCallback(unitree_a1_legged_msgs::msg::LowState::SharedPtr msg);
        void cmdVelCallback(geometry_msgs::msg::TwistStamped::SharedPtr msg);
    };
}