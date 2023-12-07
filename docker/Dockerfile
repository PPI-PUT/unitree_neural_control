FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Set the locale to UTF-8
RUN apt-get update \
    && apt-get install -y locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Set the environment variable for the locale
ENV LANG=en_US.UTF-8

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository universe \
    && apt-get update \
    && apt-get install -y curl

# Add ROS repository and key
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update and upgrade packages
RUN apt-get update \
    && apt-get upgrade -y


RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libssl-dev \
    libusb-1.0-0-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    curl \
    python3 \
    python3-dev \
    python3-vcstool \ 
    ca-certificates \
    net-tools \
    iputils-ping \ 
    nano \
    libboost-all-dev \
    unzip

RUN apt remove -y default-jdk
RUN apt autoremove -y

WORKDIR /root
# Torch
RUN curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip \
    && unzip libtorch.zip \
    && rm libtorch.zip

ENV Torch_DIR=/root/libtorch
ENV CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
# LCM
RUN curl -L -o lcm-v1.4.0.tar.gz https://github.com/lcm-proj/lcm/archive/refs/tags/v1.4.0.tar.gz \
    && tar -xzvf lcm-v1.4.0.tar.gz \
    && rm lcm-v1.4.0.tar.gz \
    && cd lcm-1.4.0 \
    && mkdir build \
    && cd build \
    && cmake .. \ 
    && make install

WORKDIR /root/ros2_ws/src
# 
RUN git clone https://github.com/PPI-PUT/unitree_legged.git
RUN git clone https://github.com/PPI-PUT/unitree_neural_control.git

WORKDIR /root/ros2_ws/src/unitree_legged
RUN mkdir -p a1_description/test \
    && vcs import < deps.repos \ 
    && cd unitree_a1_legged/unitree_legged_sdk \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make

# Install ROS 2
RUN apt-get install -y \
    ros-humble-desktop \
    ros-dev-tools
    
ENV UNITREE_SDK_VERSION=3_3
ENV UNITREE_LEGGED_SDK_PATH=/root/ros2_ws/src/unitree_legged/unitree_a1_legged/unitree_legged_sdk



WORKDIR /root/ros2_ws
RUN /bin/bash -c "source /opt/ros/humble/setup.bash; colcon build --packages-select unitree_a1_legged_msgs --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=On -Wall -Wextra -Wpedantic"
RUN /bin/bash -c "source install/setup.bash; colcon build --packages-up-to --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=On -Wall -Wextra -Wpedantic"
RUN /bin/bash -c "echo 'export LD_LIBRARY_PATH=/root/libtorch/lib/:$LD_LIBRARY_PATH' >> ~/.bashrc"

CMD ["bash"]