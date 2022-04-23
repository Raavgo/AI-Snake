from tarfile import BLOCKSIZE
import pygame
import random
from enum import Enum
from collections import namedtuple

import numpy as np

Point = namedtuple('Point', 'x, y')

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 4
    DOWN = 3

clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

WHITE = [255, 255, 255]
RED   = [200, 0, 0]
BLUE1 = [0, 0, 255]
BLUE2 = [0, 100, 255]
BLACK = [0, 0, 0]

BLOCKSIZE = 20
SPEED = 000
POSITIVE_REWARD = 10
NEGATIV_REWARD = -10
NEUTRAL_REWARD = 0

class SnakeEnv():
    
    #Private Helper functions
    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCKSIZE, BLOCKSIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCKSIZE, BLOCKSIZE))
        text = self.font.render(f'Score: {self.score}  Game Iteration: {self.game_iter}', True, WHITE)
        
        self.display.blit(text, [0,0])
        pygame.display.flip()


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCKSIZE)//BLOCKSIZE)*BLOCKSIZE
        y = random.randint(0, (self.h-BLOCKSIZE)//BLOCKSIZE)*BLOCKSIZE
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def _move(self, action):
        idx = clock_wise.index(self.direction)
        # [1,0,0] ==> Don't change direction ==> no need for if 
        # [0,1,0] ==> Turn right
        if np.array_equal(action, [0,1,0]):
            self.direction = clock_wise[(idx+1) % 4]
        # [0,0,1] ==> turn left
        elif np.array_equal(action, [0,0,1]):
            self.direction = clock_wise[(idx-1) % 4]

        
        x = self.head.x
        y = self.head.y 

        if self.direction == Direction.UP:
            y -= BLOCKSIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCKSIZE
        elif self.direction == Direction.DOWN:
            y += BLOCKSIZE
        elif self.direction == Direction.RIGHT:
            x += BLOCKSIZE
        
        self.head = Point(x, y)
            
    def is_collision(self, pt=None):
        if not pt:
            pt = self.head

        # Collided with bounding box
        if  pt.x  >  self.w - BLOCKSIZE or\
            pt.x < 0 or\
            pt.y > self.h -BLOCKSIZE or\
            pt.y < 0:
            return NEGATIV_REWARD, True
        
        #Collided with it self
        if pt in self.snake[1:]:
            return NEGATIV_REWARD, True
        
        #Collided with food
        if pt.x == self.food.x and pt.y == self.food.y:
            self.score += 1
            self.frame_iteration = 0
            self._place_food()
            return POSITIVE_REWARD, False

        return NEUTRAL_REWARD, False


    # Main Game functions
    def __init__(self, w=640, h=320):
        pygame.init()
        self.font = pygame.font.Font('Roboto-Regular.ttf', 25)
        self.w = w
        self.h = h
        self.clock = pygame.time.Clock()
        self.game_iter = 0

        self.reset()

    
    def step(self, action, game_iter):
        self.frame_iteration += 1
        self.game_iter = game_iter
        #1. Collect user Input
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.quit()        
    
        #2. Move
        self._move(action)
        self.snake.insert(0, self.head)

        if(len(self.snake)> self.score + 2):
            self.snake.pop()

        reward, collision = self.is_collision()
        
        #3. Place new food if necessary
        if collision:
            return reward, collision, self.score

        if self.frame_iteration > (100 * len(self.snake)):
            return NEGATIV_REWARD, True, self.score

       

        #4. Update UI 
        self.render()

        #5. return game_over, score
        return reward, collision, self.score

    def render(self):
        #init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self._update_ui()
        self.clock.tick(SPEED)
    
    def reset(self):
        #init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [
            self.head, 
            Point(self.head.x - BLOCKSIZE, self.head.y), 
            Point(self.head.x - (2 * BLOCKSIZE), self.head.y)
        ]

        self.score = 0
        self.frame_iteration = 0
        self._place_food()

    def quit(self):
        pygame.quit()
        quit()
