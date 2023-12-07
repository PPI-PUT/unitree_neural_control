#include "unitree_neural_control/unitree_neural_control_node.hpp"

namespace unitree_neural_control
{
    UnitreeNeuralControlNode::UnitreeNeuralControlNode(const rclcpp::NodeOptions &options)
        : Node("unitree_neural_control", options)
    {
        // Parameters
        this->declare_parameter("foot_contact_threshold", 1);
        int16_t foot_contact_threshold = 0;
        this->get_parameter("foot_contact_threshold", foot_contact_threshold);
        this->declare_parameter("model_path", std::string("/root/ros2_ws/policy_network_trained.pt"));
        std::string model_path;
        this->get_parameter("model_path", model_path);
        // Controller
        controller_ = std::make_unique<UnitreeNeuralControl>();
        controller_->setFootContactThreshold(foot_contact_threshold);
        RCLCPP_INFO(this->get_logger(), "Loading model: '%s'", model_path.c_str());
        controller_->loadModel(model_path);

        msg_goal_ = std::make_shared<geometry_msgs::msg::TwistStamped>();
        // Subscribers and publishers
        state_ = this->create_subscription<unitree_a1_legged_msgs::msg::LowState>("unitree_lowlevel/state", 1, std::bind(&UnitreeNeuralControlNode::stateCallback, this, std::placeholders::_1));
        cmd_vel_ = this->create_subscription<geometry_msgs::msg::TwistStamped>("unitree_a1_joystick/cmd_vel", 1, std::bind(&UnitreeNeuralControlNode::cmdVelCallback, this, std::placeholders::_1));
        cmd_ = this->create_publisher<unitree_a1_legged_msgs::msg::LowCmd>("unitree_lowlevel/command", 1);
        tensor_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("~/tensor", 1);
    }
    void UnitreeNeuralControlNode::stateCallback(unitree_a1_legged_msgs::msg::LowState::SharedPtr msg)
    {
        //calc time consume of modelForward
        // auto start = std::chrono::system_clock::now();
        // auto cmd = controller_->modelForward(msg_goal_, msg);
        // auto end = std::chrono::system_clock::now();
        // std::chrono::duration<double> elapsed_seconds = end - start;
        // RCLCPP_INFO(this->get_logger(), "time consume: %f", elapsed_seconds.count());
        // cmd.header.stamp = this->now();
        // cmd_->publish(cmd);
        // // Publish tensor
        // std_msgs::msg::Float32MultiArray tensor_msg;
        // tensor_msg.data.resize(12);
        unitree_a1_legged_msgs::msg::LowCmd cmd;
        std_msgs::msg::Float32MultiArray tensor_msg;
        controller_->modelForward(msg_goal_, msg, nominal_joint_position_, cmd, tensor_msg);
        cmd.header.stamp = this->now();
        cmd_->publish(cmd);
        tensor_->publish(tensor_msg);

    }
    void UnitreeNeuralControlNode::cmdVelCallback(geometry_msgs::msg::TwistStamped::SharedPtr msg)
    {
        msg_goal_ = msg;
        RCLCPP_INFO(this->get_logger(), "Received cmd_vel");
        RCLCPP_INFO(this->get_logger(), "Linear: %f %f %f", msg->twist.linear.x, msg->twist.linear.y, msg->twist.angular.z);
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