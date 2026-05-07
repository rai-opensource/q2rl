#!/bin/bash
set -e

mkdir -p ~/.mujoco
cd ~/.mujoco

if [ ! -d "mujoco210" ]; then
    echo "Downloading MuJoCo 2.1.0..."
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
    echo "Extracting..."
    tar -xzf mujoco210-linux-x86_64.tar.gz
    rm mujoco210-linux-x86_64.tar.gz
    echo "MuJoCo 2.1.0 installed successfully at ~/.mujoco/mujoco210"
else
    echo "MuJoCo 2.1.0 already installed at ~/.mujoco/mujoco210"
fi

echo ""
echo "To use MuJoCo, add these to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin"
echo "export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210"
