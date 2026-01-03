#!/bin/bash
set -euo pipefail

# --- 1. Basic System Setup & Tmux ---
echo ">>> Updating system and installing tmux..."
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y tmux git wget
else
  apt-get update && apt-get install -y tmux git wget
fi

# --- 2. Configure Hugging Face Cache to /workspace ---
# This prevents filling up the small root directory
echo ">>> Moving Hugging Face cache to /workspace/hf_cache..."
mkdir -p /workspace/hf_cache
export HF_HOME="/workspace/hf_cache"
# Make it persistent for future sessions
echo 'export HF_HOME="/workspace/hf_cache"' >> ~/.bashrc
echo 'export HF_HOME="/workspace/hf_cache"' >> ~/.profile

# --- 3. Install RoboticAttack Repo ---
echo ">>> Cloning RoboticAttack repository..."
cd /workspace
if [ ! -d "roboticAttack" ]; then
  git clone https://github.com/William-wAng618/roboticAttack.git
else
  echo ">>> /workspace/roboticAttack already exists, skipping clone."
fi
cd /workspace/roboticAttack

# Install dependencies (Trying common research repo patterns)
echo ">>> Installing python dependencies..."
# Upgrade pip first
pip install --upgrade pip
# Check for requirements file and install
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  echo "Warning: requirements.txt not found. You may need to install dependencies manually."
fi
# If there is a setup.py, install the repo itself
if [ -f "setup.py" ]; then
  pip install -e .
fi

# --- 4. Install LIBERO (The "Libero Part") ---
echo ">>> Installing LIBERO..."
cd /workspace
if [ ! -d "LIBERO" ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
else
  echo ">>> /workspace/LIBERO already exists, skipping clone."
fi
cd /workspace/LIBERO
pip install -e .
cd /workspace

# --- 5. Download Dataset ---
echo ">>> Downloading dataset to /dataset..."
# Create the target directory
mkdir -p /dataset

# Install HF CLI if not present
pip install -U "huggingface_hub[cli]"

# Download the dataset specified
# Note: This downloads to the local folder /dataset.
# We use --repo-type dataset to ensure it looks for data.
huggingface-cli download openvla/modified_libero_rlds \
  --repo-type dataset \
  --local-dir /dataset \
  --local-dir-use-symlinks False

pip install "git+https://github.com/kvablack/dlimp@main" tensorflow-graphics seaborn

echo ">>> Setup Complete!"
echo "You can now start tmux with: tmux"
