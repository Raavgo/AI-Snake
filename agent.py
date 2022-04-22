import imp
from tkinter import Y
import torch 
import random
import numpy as np
from collections import deque
from snake import SnakeEnv, Direction, Point, BLOCKSIZE

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LEARNING_RATE = 0.001
MAX_EPSILON = 100

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY) 

        self.model = None
        self.trainer = None

        # TODO: MODEL, TRAINER


    def get_state(self, env):
        head = env.head
        point_l = Point(head.x - BLOCKSIZE, head.y)
        point_r = Point(head.x + BLOCKSIZE , head.y)
        point_u = Point(head.x , head.y - BLOCKSIZE)
        point_d = Point(head.x , head.y + BLOCKSIZE)

        direction_l = env.direction == Direction.LEFT
        direction_r = env.direction == Direction.RIGHT
        direction_u = env.direction == Direction.UP
        direction_d = env.direction == Direction.DOWN

        state = [
            # Danger Straight 
            (direction_r and env.is_collision(point_r)) or
            (direction_l and env.is_collision(point_l)) or
            (direction_u and env.is_collision(point_u)) or
            (direction_d and env.is_collision(point_d)),

            # Danger right 
            (direction_u and env.is_collision(point_r)) or
            (direction_d and env.is_collision(point_l)) or
            (direction_l and env.is_collision(point_u)) or
            (direction_r and env.is_collision(point_d)),

            # Danger left 
            (direction_d and env.is_collision(point_r)) or
            (direction_u and env.is_collision(point_l)) or
            (direction_r and env.is_collision(point_u)) or
            (direction_l and env.is_collision(point_d)),

            # Move direction
            direction_l,
            direction_r,
            direction_u,
            direction_d,

            # Food location
            env.food.x < head.x, # food left of head
            env.food.x > head.x, # food right of head
            env.food.y < head.y, # food above head
            env.food.y > head.y  # food below head 
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_mem(self):
        if len(self.memory) < BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, game_overs)


    def train_short_mem(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # randomness for tradeoff of exploration and exploitation
        self.epsilon = MAX_EPSILON - self.n_games
        move = [0,0,0]

        if random.randint(0, MAX_EPSILON) < self.epsilon:
            idx = random.randint(0,2)
            move[idx] = 1
        else:
            tensor_state = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(tensor_state)
            idx = torch.argmax(prediction).item()
            move[idx] = 1
        
        return move



def run_agent():
    plot_scores = []
    plot_mean_scores = []
    total_scores = []
    high_score = 0

    agent = Agent()
    enviroment = SnakeEnv()

    while True:
        # get old state
        state_old = agent.get_state(enviroment)

        # get move
        final_move = agent.get_action(state_old)
        
        # Change the state
        reward, game_over, score = enviroment.step(final_move)

        state_new = agent.get_state(enviroment)

        # train short memory
        agent.train_short_mem(state_old, final_move, reward, game_over)
        agent.remember(state_old, final_move, reward, game_over)

        if game_over: 
            # reset enviroment
            enviroment.reset()

            # increase the game counter
            agent.n_games +=1

            # train long memory
            agent.train_long_mem()

            if score > high_score:
                high_score = score
                #TODO: agent.model.save()

            print(f'Game: {agent.n_games}, Score: {score}, Highscore: {high_score}')
            # TODO Plot

if __name__ == '__main__':
    run_agent()