# Snake Game with Reinforcement Learning

This project applies Reinforcement Learning to the classic Snake game using Python, PyTorch, and Pygame.

The snake acts as an agent. It learns how to move toward the food while avoiding collisions with the walls and its own body.

## Demo

[Watch the Snake RL demo](demo.webm)

## Reinforcement Learning Structure

- **Agent:** The snake
- **Environment:** The Snake game
- **State:** Nearby dangers, current direction, and food location
- **Actions:** Move straight, turn right, or turn left
- **Reward:** Feedback returned after each movement

The state contains 11 values:

- Danger straight, right, and left
- Current direction: left, right, up, and down
- Food position: left, right, up, and down

The neural network receives these 11 state values and predicts Q-values for three possible actions:

```text
[1, 0, 0] → Move straight
[0, 1, 0] → Turn right
[0, 0, 1] → Turn left
```

## Reward System

```text
Eat food:        +10
Collision:       -10
Normal movement:   0
```

## Learning Method

The agent is trained using Deep Q-Learning with:

- Epsilon-greedy exploration
- Experience replay
- Bellman Q-value updates
- Mean Squared Error loss
- Adam optimizer
- Training checkpoint saving

At the beginning, the snake performs more random actions to explore the environment. After training for more games, it gradually relies on the neural network and selects actions with higher predicted Q-values.

## Technologies

- Python
- PyTorch
- Pygame
- NumPy
- Matplotlib

## Project Structure

```text
snake_game/
├── agent.py
├── snake_game.py
├── model.py
├── helper.py
├── demo.webm
├── best_model/
└── README.md
```

## Installation

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install the required libraries:

```bash
pip install torch pygame numpy matplotlib
```

## Run the Project

Start the training process:

```bash
python agent.py
```

The program will display the Snake game, training score, record, and mean score.

## Purpose

The purpose of this project is to understand how the main components of Reinforcement Learning work together:

```text
State
→ Action
→ Reward
→ Next State
→ Model Training
```

This project provides a basic foundation for applying Reinforcement Learning to more complex autonomous control problems in the future.