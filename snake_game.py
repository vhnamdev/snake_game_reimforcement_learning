import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

font = pygame.font.SysFont("arial", 24, bold=True)
small_font = pygame.font.SysFont("arial", 14, bold=True)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

BLOCK_SIZE = 20
SPEED = 20

# Background
BACKGROUND = (15, 23, 42)
GRID_COLOR = (30, 41, 59)

# Snake
SNAKE_HEAD = (34, 211, 238)
SNAKE_BODY = (14, 165, 233)
SNAKE_INNER = (56, 189, 248)
SNAKE_EYE = (15, 23, 42)

# Food
FOOD_RED = (248, 70, 70)
FOOD_LIGHT = (255, 120, 120)
LEAF_GREEN = (74, 222, 128)
STEM_BROWN = (120, 72, 40)

# Text
WHITE = (248, 250, 252)
MUTED_TEXT = (148, 163, 184)

# Score panel
PANEL_COLOR = (30, 41, 59)
PANEL_BORDER = (71, 85, 105)


class SnakeGameRL:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        # init game step
        self.direction = Direction.RIGHT

        self.head = Point(
            self.w // 2,
            self.h // 2
        )

        self.snake = [
            self.head,
            Point(
                self.head.x - BLOCK_SIZE,
                self.head.y
            ),
            Point(
                self.head.x - (2 * BLOCK_SIZE),
                self.head.y
            )
        ]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = (
            random.randint(
                0,
                (self.w - BLOCK_SIZE) // BLOCK_SIZE
            )
            * BLOCK_SIZE
        )

        y = (
            random.randint(
                0,
                (self.h - BLOCK_SIZE) // BLOCK_SIZE
            )
            * BLOCK_SIZE
        )

        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        # collect user's input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # check if game over
        reward = 0
        game_over = False

        if (
            self.is_collision()
            or self.frame_iteration > 100 * len(self.snake)
        ):
            game_over = True
            reward = -10

            return reward, game_over, self.score

        # place new food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # hits boundary
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < 0
        ):
            return True

        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BACKGROUND)

        self._draw_grid()
        self._draw_food()
        self._draw_snake()
        self._draw_score_panel()

        pygame.display.flip()

    def _draw_grid(self):
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(
                self.display,
                GRID_COLOR,
                (x, 0),
                (x, self.h),
                1
            )

        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(
                self.display,
                GRID_COLOR,
                (0, y),
                (self.w, y),
                1
            )

    def _draw_snake(self):
        for index, pt in enumerate(self.snake):
            outer_rect = pygame.Rect(
                int(pt.x + 1),
                int(pt.y + 1),
                BLOCK_SIZE - 2,
                BLOCK_SIZE - 2
            )

            if index == 0:
                color = SNAKE_HEAD
            else:
                color = SNAKE_BODY

            pygame.draw.rect(
                self.display,
                color,
                outer_rect,
                border_radius=6
            )

            inner_rect = pygame.Rect(
                int(pt.x + 5),
                int(pt.y + 5),
                BLOCK_SIZE - 10,
                BLOCK_SIZE - 10
            )

            pygame.draw.rect(
                self.display,
                SNAKE_INNER,
                inner_rect,
                border_radius=4
            )

        self._draw_snake_eyes()

    def _draw_snake_eyes(self):
        x = int(self.head.x)
        y = int(self.head.y)

        if self.direction == Direction.RIGHT:
            eye_1 = (x + 14, y + 6)
            eye_2 = (x + 14, y + 14)

        elif self.direction == Direction.LEFT:
            eye_1 = (x + 6, y + 6)
            eye_2 = (x + 6, y + 14)

        elif self.direction == Direction.UP:
            eye_1 = (x + 6, y + 6)
            eye_2 = (x + 14, y + 6)

        else:
            eye_1 = (x + 6, y + 14)
            eye_2 = (x + 14, y + 14)

        pygame.draw.circle(
            self.display,
            WHITE,
            eye_1,
            3
        )

        pygame.draw.circle(
            self.display,
            WHITE,
            eye_2,
            3
        )

        pygame.draw.circle(
            self.display,
            SNAKE_EYE,
            eye_1,
            1
        )

        pygame.draw.circle(
            self.display,
            SNAKE_EYE,
            eye_2,
            1
        )

    def _draw_food(self):
        center_x = int(
            self.food.x + BLOCK_SIZE // 2
        )

        center_y = int(
            self.food.y + BLOCK_SIZE // 2 + 1
        )

        pygame.draw.circle(
            self.display,
            FOOD_RED,
            (center_x, center_y),
            8
        )

        pygame.draw.circle(
            self.display,
            FOOD_LIGHT,
            (center_x - 3, center_y - 3),
            2
        )

        pygame.draw.line(
            self.display,
            STEM_BROWN,
            (center_x, center_y - 8),
            (center_x + 1, center_y - 12),
            3
        )

        leaf_rect = pygame.Rect(
            center_x + 1,
            center_y - 13,
            8,
            5
        )

        pygame.draw.ellipse(
            self.display,
            LEAF_GREEN,
            leaf_rect
        )

    def _draw_score_panel(self):
        panel_rect = pygame.Rect(
            10,
            10,
            145,
            58
        )

        pygame.draw.rect(
            self.display,
            PANEL_COLOR,
            panel_rect,
            border_radius=10
        )

        pygame.draw.rect(
            self.display,
            PANEL_BORDER,
            panel_rect,
            width=2,
            border_radius=10
        )

        label = small_font.render(
            "CURRENT SCORE",
            True,
            MUTED_TEXT
        )

        score_text = font.render(
            str(self.score),
            True,
            WHITE
        )

        self.display.blit(
            label,
            (22, 17)
        )

        self.display.blit(
            score_text,
            (22, 35)
        )

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP
        ]

        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change

        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u

        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE

        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE

        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)