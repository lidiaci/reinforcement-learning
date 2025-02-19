from copy import deepcopy

class Connect3Environment():
    def __init__(self, n_rows, n_cols):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.current_state = self.initialize_board()
    
    def initialize_board(self):
        return np.zeros((self.n_rows, self.n_cols))
    
    def reset(self):
        self.current_state = self.initialize_board()
        
        return self.current_state
        
    def get_all_states(self):
        return [*self.value_fun]
    
    def is_terminal(self, state):
        reshaped_state = np.array(state).reshape(self.n_rows, self.n_cols)
        for row in range(4):
            for col in range(5):
                if reshaped_state[row][col] == 0:
                    continue
                player = reshaped_state[row][col]
                if (col <= 2 and all(reshaped_state[row][col+i] == player for i in range(3))) or \
                   (row <= 1 and all(reshaped_state[row+i][col] == player for i in range(3))) or \
                   (row <= 1 and col <= 2 and all(reshaped_state[row+i][col+i] == player for i in range(3))) or \
                   (row >= 2 and col <= 2 and all(reshaped_state[row-i][col+i] == player for i in range(3))):
                    return True
        # checking for draw also
        return not any(0 in row for row in reshaped_state)
    
    def get_possible_actions(self, state):
        reshaped_state = np.array(state).reshape(self.n_rows, self.n_cols)
        return [col for col in range(5) if reshaped_state[0, col] == 0]
    
    def calculate_turn(self, state):
        reshaped_state = np.array(state).reshape(self.n_rows, self.n_cols)
        player1_count = np.sum(reshaped_state==1)
        player2_count = np.sum(reshaped_state==2)
        
        if player1_count == player2_count:
            return 1
        elif player1_count > player2_count:
            return 2
        else:
            print(reshaped_state)
            return None
                
    def step(self, action):
        prev_step = deepcopy(self.current_state)
        reshaped_board = np.array(self.current_state).reshape(self.n_rows, self.n_cols)
        # ruch agenta
        for row in range(3, -1, -1):
            if reshaped_board[row, action]== 0:
                reshaped_board[row, action] = 1
                break
        
        # ruch przeciwnika
        opponents_action = np.random.choice(self.get_possible_actions(reshaped_board))
        for row in range(3, -1, -1):
            if reshaped_board[row, opponents_action]== 0:
                reshaped_board[row, opponents_action] = 2
                break
        self.current_state = reshaped_board
        
        return self.current_state, self.get_reward(prev_step, action, self.current_state), self.is_terminal(self.current_state)
    
    def get_reward(self, state, action, next_state):
        reshaped_state = np.array(state).reshape(self.n_rows, self.n_cols)
        assert action in self.get_possible_actions(reshaped_state)
        
        # draw
        if not any(0 in row for row in reshaped_state):
            return 0
        # current state is terminal - lost game
        elif self.is_terminal(state):
            return -1
        # next_state is terminal - game ended after our move
        elif self.is_terminal(next_state):
            return 1
        else:
            return 0
        
from collections import defaultdict
import random
import numpy as np
import pickle
from copy import deepcopy

class ActiveLearningAgent:
    def __init__(self, alpha, epsilon, gamma, get_legal_actions):
        self.agent_player = 1
        self.get_legal_actions = get_legal_actions
        self.discount = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))

    def flatten_state(self, state):
        return tuple(state.flatten())

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        flat_state = self.flatten_state(state)
        return self._qvalues[flat_state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        flat_state = self.flatten_state(state)
        self._qvalues[flat_state][action] = value

    #---------------------START OF YOUR CODE---------------------#

    def get_value(self, state):
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        Note: please take into account that q-values can be negative.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        #
        # INSERT CODE HERE to get maximum possible value for a given state
        #
        
        max_value = float('-inf')
        for action in possible_actions:
            if max_value < self.get_qvalue(state, action):
                max_value = self.get_qvalue(state, action)
        
        return max_value

    def update(self, state, action, reward, next_state):
        """
        You should do your Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        #
        # INSERT CODE HERE to update value for the given state and action
        #
        
        value = self.get_qvalue(state, action)+learning_rate*(reward+gamma*self.get_value(next_state)-self.get_qvalue(state, action))
        self.set_qvalue(state, action, value)
        
    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        #
        # INSERT CODE HERE to get best possible action in a given state (remember to break ties randomly)
        #
        
        best_actions = []
        action_value = float('-inf')
        for action in possible_actions:
            if action_value < self.get_qvalue(state, action):
                best_actions = [action]
                action_value = self.get_qvalue(state, action)
            elif action_value == self.get_qvalue(state, action):
                best_actions.append(action)
        if len(best_actions) == 1:
            best_action = best_actions[0]
        else:
            best_action = random.choice(best_actions)
                
        return best_action

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        #
        # INSERT CODE HERE to get action in a given state (according to epsilon greedy algorithm)
        #        

        random_number = random.random()
        if random_number <= self.epsilon:
            exploration = True
        else:
            exploration = False
        
        if exploration == False:
            chosen_action = self.get_best_action(state)
        else:
            if len(possible_actions)>1:
                # possible_actions.remove(self.get_best_action(state))
                chosen_action = random.choice(possible_actions)
            else: 
                chosen_action = possible_actions[0]
        
        return chosen_action

    def turn_off_learning(self):
        """
        Function turns off agent learning.
        """
        self.epsilon = 0
        self.alpha = 0

import os

def game_is_terminal(state):
    reshaped_state = np.array(state).reshape(4,5)
    #print(reshaped_state)
    for row in range(4):
        for col in range(5):
            if reshaped_state[row][col] == 0:
                continue
            player = reshaped_state[row][col]
            if (col <= 2 and all(reshaped_state[row][col+i] == player for i in range(3))) or \
               (row <= 1 and all(reshaped_state[row+i][col] == player for i in range(3))) or \
               (row <= 1 and col <= 2 and all(reshaped_state[row+i][col+i] == player for i in range(3))) or \
               (row >= 2 and col <= 2 and all(reshaped_state[row-i][col+i] == player for i in range(3))):
                return True
    # checking for draw also
    return not any(0 in row for row in reshaped_state)

def game_step(board, action, player):
    next_state = deepcopy(board)
    for row in range(3, -1, -1):
        if next_state[row, action] == 0:
            next_state[row, action] = player
            break
    
    return next_state, game_is_terminal(next_state)

def game(env, agent):
    clear = lambda: os.system('cls')
    
    player_round = 1
    board = env.reset()
    print(board)
    
    while True:
        if player_round == 1:
            agent_move = agent.get_action(board)
            board, done = game_step(board, agent_move, player_round)
            player_round = 3 - player_round
        else:
            player_move = int(input('Podaj swój ruch (0-4): '))
            board, done = game_step(board, player_move, player_round)
            player_round = 3 - player_round
        clear()
        print(board)
        
        if env.is_terminal(board):
            print(f'Wygrał gracz {3 - player_round}')
            break
        
def play_and_train(env, agent):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    state = env.reset()

    done = False

    while not done:
        # get agent to pick action given state state.
        action = agent.get_action(state)

        next_state, reward, done = env.step(action)

        #
        # INSERT CODE HERE to train (update) agent for state
        #        
        
        agent.update(state, action, reward, next_state)
        
        state = next_state
        total_reward += reward
        if done:
            break

    return total_reward

environment = Connect3Environment(4, 5)
agent = ActiveLearningAgent(0.5, 0.25, 0.99, environment.get_possible_actions)

print('Uczenie rozpoczęte!')
for i in range(1000000):
    play_and_train(environment, agent)  
print('Agent wytrenowany!')

agent.turn_off_learning()

while True:
    game(environment, agent)
    inp = input("Czy chcesz zagrać ponownie?")
    if inp != 'tak':
        break
