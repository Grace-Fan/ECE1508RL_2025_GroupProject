# ECE1508RL_2025_GroupProject

This project trains a DQN agent to safely navigate in the intersection‑v1 intersection environment using kinematic state information. The agent learns driving behavior through reinforcement learning, improving its decision-making across different traffic conditions

## Project Structure

- **`baseline/`**  
  Contains the rule based model.

- **`results/`**  
  Stores different training results and trained models.

- **`main.py/`**  
  Contains all code for the DQN training

- **`requirements.txt`**  
  Lists the Python dependencies required for the project.

## Setup

### 1. Create and Activate a Virtual Environment

Run the following commands to set up a virtual environment and install the required dependencies:

```bash
!apt-get install -y xvfb ffmpeg
!git clone https://github.com/Farama-Foundation/HighwayEnv.git 2> /dev/null
pip install -r requirements.txt
```

### 2. Train the Model

Train the DQN model using the following command:

```bash
python main.py
```
