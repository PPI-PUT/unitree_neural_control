#include "unitree_neural_control/unitree_neural_control_node.hpp"

namespace unitree_neural_control
{
    UnitreeNeuralControlNode::UnitreeNeuralControlNode(const rclcpp::NodeOptions &options)
        : Node("unitree_neural_control", options)
    {
        // Parameters
        this->declare_parameter("foot_contact_threshold", 20);
        int16_t foot_contact_threshold = 0;
        this->get_parameter("foot_contact_threshold", foot_contact_threshold);
        this->declare_parameter("model_path", std::string("/root/ros2_ws/policy-app/policy_network.pt"));
        std::string model_path;
        this->get_parameter("model_path", model_path);
        // Controller
        controller_ = std::make_unique<UnitreeNeuralControl>();
        controller_->setFootContactThreshold(foot_contact_threshold);
        RCLCPP_INFO(this->get_logger(), "Loading model: '%s'", model_path.c_str());
        controller_->loadModel(model_path);

        msg_goal_ = std::make_shared<geometry_msgs::msg::TwistStamped>();
        // Subscribers and publishers
        state_ = this->create_subscription<unitree_a1_legged_msgs::msg::LowState>("~/state", 1, std::bind(&UnitreeNeuralControlNode::stateCallback, this, std::placeholders::_1));
        cmd_vel_ = this->create_subscription<geometry_msgs::msg::TwistStamped>("~/cmd_vel", 1, std::bind(&UnitreeNeuralControlNode::cmdVelCallback, this, std::placeholders::_1));
        cmd_ = this->create_publisher<unitree_a1_legged_msgs::msg::LowCmd>("~/command", 1);
    }
    void UnitreeNeuralControlNode::stateCallback(unitree_a1_legged_msgs::msg::LowState::SharedPtr msg)
    {
        auto cmd = controller_->modelForward(msg_goal_, msg);
        cmd.header.stamp = this->now();
        cmd_->publish(cmd);
    }
    void UnitreeNeuralControlNode::cmdVelCallback(geometry_msgs::msg::TwistStamped::SharedPtr msg)
    {
        msg_goal_ = msg;
    }
} // namespace unitree_neural_control

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  rclcpp::spin(std::make_shared<unitree_neural_control::UnitreeNeuralControlNode>(options));
  rclcpp::shutdown();

  return 0;
}