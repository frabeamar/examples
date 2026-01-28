
import tkinter as tk
from tkinter import messagebox
from typing import List, Tuple, Optional
import numpy as np

import random
from typing import Dict, List, Tuple

import numpy as np

# Define types for clarity
Action = Tuple[int, int]
StateStr = str


class TicTacToeGame:
    def __init__(self) -> None:
        self.board: np.ndarray = np.zeros((3, 3), dtype=int)

    def reset(self) -> StateStr:
        self.board = np.zeros((3, 3), dtype=int)
        return self.get_state()

    def get_state(self) -> StateStr:
        return "".join(map(str, self.board.flatten()))

    def get_available_moves(self) -> List[Action]:
        # np.where returns a tuple of arrays (rows, cols)
        coords = np.where(self.board == 0)
        return [(int(a), int (b)) for a, b in zip(*coords)]

    def make_move(self, move: Action, player: int) -> None:
        self.board[move] = player

    def check_winner(self) -> int:
        """Returns 1 or 2 for winner, 0 for draw, -1 for ongoing."""
        for i in range(3):
            if all(self.board[i, :] == 1) or all(self.board[:, i] == 1):
                return 1
            if all(self.board[i, :] == 2) or all(self.board[:, i] == 2):
                return 2

        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:
            return self.board[0, 0]
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] != 0:
            return self.board[0, 2]

        if not np.any(self.board == 0):
            return 0
        return -1


class TicTacToeAgent:
    def __init__(
        self, alpha: float = 0.5, epsilon: float = 0.1, gamma: float = 0.9
    ) -> None:
        # The Q-table maps a (State, Action) pair to a float value
        self.q_table: Dict[Tuple[StateStr, Action], float] = {}
        self.alpha: float = alpha
        self.epsilon: float = epsilon
        self.gamma: float = gamma

    def get_q_value(self, state: StateStr, action: Action) -> float:
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state: StateStr, available_moves: List[Action]) -> Action:
        if random.random() < self.epsilon:
            return random.choice(available_moves)

        q_values: List[float] = [self.get_q_value(state, a) for a in available_moves]
        max_q: float = max(q_values)

        best_actions: List[Action] = [
            available_moves[i] for i, q in enumerate(q_values) if q == max_q
        ]
        return random.choice(best_actions)

    def learn(
        self,
        state: StateStr,
        action: Action,
        reward: float,
        next_state: StateStr,
        next_available_moves: List[Action],
    ) -> None:
        old_q: float = self.get_q_value(state, action)

        if not next_available_moves:
            target: float = reward
        else:
            next_max_q: float = max(
                [self.get_q_value(next_state, a) for a in next_available_moves]
            )
            target = reward + self.gamma * next_max_q
        self.q_table[(state, action)] = old_q + self.alpha * (target - old_q)


# Helper to represent board as a string for dictionary keys
def get_state_string(board):
    return "".join(map(str, board.flatten()))


def train(episodes=50000):
    game = TicTacToeGame()
    agent = TicTacToeAgent(alpha=0.2, epsilon=0.3)  # Agent plays as '1' (X)

    for i in range(episodes):
        state = game.reset()
        done = False

        # Decay epsilon to shift from exploration to exploitation
        if i % 5000 == 0:
            agent.epsilon *= 0.9

        while not done:
            # 1. Agent's Turn (X)
            moves = game.get_available_moves()
            action = agent.choose_action(state, moves)
            game.make_move(action, 1)

            new_state = game.get_state()
            winner = game.check_winner()

            if winner != -1:
                # Assign rewards
                reward = 1 if winner == 1 else 0.5 if winner == 0 else -1
                agent.learn(state, action, reward, new_state, [])
                done = True
            else:
                # 2. Opponent's Turn (Random/Simple O)
                # In self-play, you'd have a second agent here
                opp_moves = game.get_available_moves()
                opp_move = random.choice(opp_moves)
                game.make_move(opp_move, 2)

                winner = game.check_winner()
                new_state_after_opp = game.get_state()

                if winner != -1:
                    reward = -1 if winner == 2 else 0.5 if winner == 0 else 1
                    agent.learn(state, action, reward, new_state_after_opp, [])
                    done = True
                else:
                    # No winner yet, update Q-value based on future potential
                    agent.learn(
                        state,
                        action,
                        0,
                        new_state_after_opp,
                        game.get_available_moves(),
                    )
                    state = new_state_after_opp

    return agent


def print_board(board: np.ndarray) -> None:
    chars: Dict[int, str] = {0: " ", 1: "X", 2: "O"}
    print("\n")
    for i in range(3):
        row: List[str] = [chars[board[i, j]] for j in range(3)]
        print(f" {row[0]} | {row[1]} | {row[2]} ")
        if i < 2:
            print("-----------")
    print("\n")


def play_human(agent: TicTacToeAgent) -> None:
    game: TicTacToeGame = TicTacToeGame()
    state: StateStr = game.reset()

    print("--- Game Start! ---")
    done: bool = False

    while not done:
        # Agent (X)
        moves: List[Action] = game.get_available_moves()
        action: Action = agent.choose_action(state, moves)
        game.make_move(action, 1)
        print_board(game.board)

        winner: int = game.check_winner()
        if winner != -1:
            print("Game Over!")
            break

        # Human (O)
        valid_move: bool = False
        while not valid_move:
            user_input: str = input("Your move (row,col): ")
            try:
                r, c = map(int, user_input.split(","))
                if (r, c) in game.get_available_moves():
                    game.make_move((r, c), 2)
                    valid_move = True
            except ValueError:
                print("Invalid input.")

        state = game.get_state()
        if game.check_winner() != -1:
            print_board(game.board)
            break

class TicTacToeGUI:
    def __init__(self, agent: TicTacToeAgent) -> None:
        self.agent = agent
        self.game = TicTacToeGame()
        self.root = tk.Tk()
        self.root.title("RL Tic-Tac-Toe: Human vs AI")
        self.root.lift()  # Lift window to top
        self.root.attributes('-topmost', True) # Force to front
        
        # UI State
        self.buttons: List[List[tk.Button]] = []
        self._setup_grid()
        
        # Start the game: Agent goes first (Player 1)
        self.agent_turn()

    def _setup_grid(self) -> None:
        """Creates a 3x3 grid of buttons."""
        for r in range(3):
            button_row: List[tk.Button] = []
            for c in range(3):
                btn = tk.Button(
                    self.root, text="", font=('normal', 40, 'bold'),
                    width=5, height=2,
                    command=lambda row=r, col=c: self.on_click(row, col)
                )
                btn.grid(row=r, column=c)
                button_row.append(btn)
            self.buttons.append(button_row)

    def on_click(self, r: int, c: int) -> None:
        """Handles human move (Player 2 / O)."""
        state = self.game.get_state()
        move = (r, c)

        if self.game.board[move] == 0:
            self.game.make_move(move, 2)
            self.buttons[r][c].config(text="O", state="disabled", disabledforeground="blue")
            
            if not self.check_game_over():
                self.agent_turn()
        else:
            messagebox.showwarning("Invalid Move", "This cell is already taken!")

    def agent_turn(self) -> None:
        """Handles AI move (Player 1 / X)."""
        state = self.game.get_state()
        moves = self.game.get_available_moves()
        
        # Turn off exploration for the GUI play
        orig_eps = self.agent.epsilon
        self.agent.epsilon = 0.0
        action = self.agent.choose_action(state, moves)
        self.agent.epsilon = orig_eps
        
        self.game.make_move(action, 1)
        r, c = action
        self.buttons[r][c].config(text="X", state="disabled", disabledforeground="red")
        
        self.check_game_over()

    def check_game_over(self) -> bool:
        """Checks for win/loss/draw and shows a popup."""
        winner = self.game.check_winner()
        if winner != -1:
            if winner == 1:
                messagebox.showinfo("Game Over", "AI (X) Wins!")
            elif winner == 2:
                messagebox.showinfo("Game Over", "You (O) Win!")
            else:
                messagebox.showinfo("Game Over", "It's a Draw!")
            self.reset_gui()
            return True
        return False

    def reset_gui(self) -> None:
        """Clears board for a new game."""
        self.game.reset()
        for r in range(3):
            for c in range(3):
                self.buttons[r][c].config(text="", state="normal")
        self.agent_turn()

    def run(self) -> None:
        self.root.mainloop()


agent = train(episodes=20)
TicTacToeGUI(agent).run()
