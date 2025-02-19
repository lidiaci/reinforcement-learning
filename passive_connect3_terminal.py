import numpy as np
import os
import pickle
from copy import deepcopy

class Agent:
    def __init__(self, load_policy=False, policy_path=None):
        self.agent_player = 1
        self.board = self.initialize_board(4, 5)
        
        if load_policy and policy_path is not None:
            with open(policy_path, "rb") as f:
                self.policy = pickle.load(f)
            print("Value function loaded from the file.")
        else:
            if policy_path is None:
                print("Path for policy not given. Random policy will be initialized.")
            # we choose first valid action as initial policy
            self.policy = {}
            for state_key in self.value_fun.keys():
                state = state_key
                state_reshaped = np.array(state).reshape(4,5)
                valid_actions_in_state = self.valid_actions(state_reshaped)
                if valid_actions_in_state:
                    self.policy[state_key] = valid_actions_in_state[0]
            print("Initial policy initialized.")
    
    def step_env(self, reshaped_state, action, player):
        next_state = deepcopy(reshaped_state)
        
        for row in range(3, -1, -1):
            if next_state[row, action] == 0:
                next_state[row, action] = player
                break
        
        gameover, winner = self.if_gameover(next_state)    
        if gameover:
            if player == self.agent_player:
                return next_state, 10, winner
            else:
                return next_state, -15, 3-player
        else:
            return next_state, -1, 3-player

            
    def valid_actions(self, reshaped_board):
        return [col for col in range(4) if reshaped_board[0, col] == 0]
    

    def initialize_board(self, rows, columns):
        """function to initialize a board for the game with given size
        
        Args:
            rows (int): number of rows on the board
            columns (int): number if columns on the board

        Returns:
            np.array: empty board with size given based on arguments
        """
        return np.zeros((rows, columns), dtype=int)
    
    def if_gameover(self, reshaped_state):
        """function which checks for game's ending condition
        
        Args:
            state (np.array): reshaped board

        Returns:
            (bool, None or player): returns True if game's ending condition is satisfied and False otherwise
        """
        for row in range(4):
            for col in range(5):
                if reshaped_state[row][col] == 0:
                    continue
                player = reshaped_state[row][col]
                if (col <= 2 and all(reshaped_state[row][col+i] == player for i in range(3))) or \
                   (row <= 1 and all(reshaped_state[row+i][col] == player for i in range(3))) or \
                   (row <= 1 and col <= 2 and all(reshaped_state[row+i][col+i] == player for i in range(3))) or \
                   (row >= 2 and col <= 2 and all(reshaped_state[row-i][col+i] == player for i in range(3))):
                    return True, player
        # checking for draw also
        return not any(0 in row for row in reshaped_state), None
    
    def game(self):
        clear = lambda: os.system('cls')
        
        player_round = 1
        board = self.initialize_board(4,5)
        print(board)
        while True:
            if player_round == 1:
                flattened_board = tuple(board.flatten())
                agent_move = self.policy[flattened_board]
                board, reward, player_round = self.step_env(board, agent_move, self.agent_player)
            else:
                player_move = int(input('Podaj swój ruch (0-4): '))
                board, reward, player_round = self.step_env(board, player_move, 3-self.agent_player)
            clear()
            print(board)
            
            gameover, player = self.if_gameover(board)
            if gameover:
                print(f"Wygrał gracz {player}")
                break
            
if __name__ == '__main__':
    agent = Agent(True, f"passive_learning_policy.pkl")
    agent.game()