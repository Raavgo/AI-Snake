from select import select
from tarfile import BLOCKSIZE
from tkinter import RIGHT, font
import pygame
import random
from enum import Enum
from collections import namedtuple

Point = namedtuple('Point', 'x, y')

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 4
    DOWN = 3

WHITE = [255, 255, 255]
RED   = [200, 0, 0]
BLUE1 = [0, 0, 255]
BLUE2 = [0, 100, 255]
BLACK = [0, 0, 0]

BLOCKSIZE = 20
SPEED = 40


class SnakeGame:
    
    def __init__(self, w=1280, h=640):
        pygame.init()
        self.font = pygame.font.Font('Roboto-Regular.ttf', 25)
        self.w = w
        self.h = h

        #init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        #init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [
            self.head, 
            Point(self.head.x - BLOCKSIZE, self.head.y), 
            Point(self.head.x - (2 * BLOCKSIZE), self.head.y)
        ]

        self.score = 0
        self.food = None
        self._place_food()

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCKSIZE, BLOCKSIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCKSIZE, BLOCKSIZE))
        text = self.font.render(f'Score: {self.score}', True, WHITE)
        
        self.display.blit(text, [0,0])
        pygame.display.flip()


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCKSIZE)//BLOCKSIZE)*BLOCKSIZE
        y = random.randint(0, (self.h-BLOCKSIZE)//BLOCKSIZE)*BLOCKSIZE
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y 

        if direction == Direction.UP:
            y -= BLOCKSIZE
        elif direction == Direction.LEFT:
            x -= BLOCKSIZE
        elif direction == Direction.DOWN:
            y += BLOCKSIZE
        elif direction == Direction.RIGHT:
            x += BLOCKSIZE
        
        self.head = Point(x, y)
            
    def _is_collision(self):
        # Collided with bounding box
        if  self.head.x  >  self.w - BLOCKSIZE or\
            self.head.x < 0 or\
            self.head.y > self.h -BLOCKSIZE or\
            self.head.y < 0:
            return True
        
        #Collided with it self
        for elem in self.snake[1:]:
            if self.head.x == elem.x and self.head.y == elem.y:
                return True
        
        #Collided with food
        if self.head.x == self.food.x and self.head.y == self.food.y:
            self.score += 1
            self._place_food()
            return False

        return False



    def play_step(self):
        #1. Collect user Input
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.quit()        
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_w and not self.direction == Direction.DOWN:
                    self.direction = Direction.UP
                elif e.key == pygame.K_a and not self.direction == Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif e.key == pygame.K_s and not self.direction == Direction.UP:
                    self.direction = Direction.DOWN
                elif e.key == pygame.K_d and not self.direction == Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif e.key == pygame.K_ESCAPE:
                    self.quit()
                else:
                    continue
        #2. Move
        self._move(self.direction)
        self.snake.insert(0, self.head)

        if(len(self.snake)> self.score + 2):
            self.snake.pop()

        #3. Place new food if necessary
        if self._is_collision():
            return True, self.score

        #4. Update UI 
        self._update_ui()
        self.clock.tick(SPEED)

        #5. return game_over, score
        return False, self.score
         

    def quit(self):
        pygame.quit()
        quit()


if __name__ == '__main__':
    game = SnakeGame()

    while True:
        gameover, score = game.play_step()

        if gameover == True:
            break
    

    print(f'Final Score {score}')
    game.quit()