import os
import random
from collections import deque

import numpy as np
import torch

from snake_game import SnakeGameRL, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

CHECKPOINT_PATH = "./best_model/training_checkpoint.pth"
BEST_MODEL_PATH = "./best_model/best_model.pth"


class Agent:

    def __init__(self):

        # init variables
        self.num_games = 0
        self.epsilon = 0
        self.gamma = 0.9

        # if we exceed the amount of memory,
        # this will automatically remove element from left
        self.memory = deque(maxlen=MAX_MEMORY)

        # model, train
        self.model = Linear_QNet(11, 256, 3)

        self.trainer = QTrainer(
            self.model,
            lr=LR,
            gamma=self.gamma
        )

    def get_state(self, game):
        head = game.snake[0]

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]

        return np.array(state, dtype=int)

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def train_replay_batch(self):
        if len(self.memory) == 0:
            return

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory,
                BATCH_SIZE
            )
        else:
            mini_sample = list(self.memory)

        states, actions, rewards, next_states, dones = zip(
            *mini_sample
        )

        self.trainer.train_step(
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def train_current_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        self.trainer.train_step(
            state,
            action,
            reward,
            next_state,
            done
        )

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(
            0,
            80 - self.num_games
        )

        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1

        else:
            state0 = torch.tensor(
                state,
                dtype=torch.float
            )

            with torch.no_grad():
                prediction = self.model(state0)

            move = torch.argmax(
                prediction
            ).item()

            final_move[move] = 1

        return final_move

    def save_checkpoint(
        self,
        record,
        total_score,
        plot_scores,
        plot_mean_scores
    ):
        os.makedirs(
            os.path.dirname(CHECKPOINT_PATH),
            exist_ok=True
        )

        checkpoint = {
            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.trainer.optimizer.state_dict(),

            "num_games":
                self.num_games,

            "record":
                record,

            "total_score":
                total_score,

            "plot_scores":
                plot_scores,

            "plot_mean_scores":
                plot_mean_scores
        }

        torch.save(
            checkpoint,
            CHECKPOINT_PATH
        )

    def load_checkpoint(self):
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location=torch.device("cpu")
            )

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            self.trainer.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

            self.num_games = checkpoint.get(
                "num_games",
                0
            )

            record = checkpoint.get(
                "record",
                0
            )

            total_score = checkpoint.get(
                "total_score",
                0
            )

            plot_scores = checkpoint.get(
                "plot_scores",
                []
            )

            plot_mean_scores = checkpoint.get(
                "plot_mean_scores",
                []
            )

            self.model.train()

            print(
                "Continued training from game:",
                self.num_games
            )

            print(
                "Current record:",
                record
            )

            return (
                record,
                total_score,
                plot_scores,
                plot_mean_scores
            )

        if os.path.exists(BEST_MODEL_PATH):
            self.model.load_state_dict(
                torch.load(
                    BEST_MODEL_PATH,
                    map_location=torch.device("cpu")
                )
            )

            self.model.train()

            print(
                "Loaded best model:",
                BEST_MODEL_PATH
            )

        else:
            print(
                "No checkpoint found. Starting new training."
            )

        return 0, 0, [], []


def train():
    agent = Agent()

    (
        record,
        total_score,
        plot_scores,
        plot_mean_scores
    ) = agent.load_checkpoint()

    game = SnakeGameRL()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(
            final_move
        )

        state_new = agent.get_state(game)

        # train short memory
        agent.train_current_transition(
            state_old,
            final_move,
            reward,
            state_new,
            done
        )

        # remember
        agent.remember(
            state_old,
            final_move,
            reward,
            state_new,
            done
        )

        if done:
            # train long memory, plot result
            game.reset()

            agent.num_games += 1

            agent.train_replay_batch()

            if score > record:
                record = score
                agent.model.save()

            print(
                "Game",
                agent.num_games,
                "Score",
                score,
                "Record:",
                record
            )

            plot_scores.append(score)

            total_score += score

            mean_score = (
                total_score
                / agent.num_games
            )

            plot_mean_scores.append(
                mean_score
            )

            # save latest training state
            agent.save_checkpoint(
                record,
                total_score,
                plot_scores,
                plot_mean_scores
            )

            plot(
                plot_scores,
                plot_mean_scores
            )


if __name__ == "__main__":
    train()