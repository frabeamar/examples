import random
import tkinter as tk
import numpy as np
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple, Optional

import tqdm

class UltimateDQN(nn.Module):
    def __init__(self) -> None:
        super(UltimateDQN, self).__init__()
        # Input: 1 channel (the board), 9x9 grid
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Flatten()
        self.fc2 = nn.Linear(128 * 9 * 9, 256)
        self.output = nn.Linear(256, 81) # 81 possible move outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.fc1(x)
        x = torch.relu(self.fc2(x))
        return self.output(x)

class ReplayMemory:
    def __init__(self, capacity: int = 10000) -> None:
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

class  UltimateAgent:
    def __init__(self, learning_rate: float = 1e-4) -> None:
        self.policy_net = UltimateDQN().cuda()
        self.target_net = UltimateDQN().cuda() # Stable target for Bellman updates
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory()
        self.epsilon = 1.0  # Exploration rate
        self.gamma = 0.99   # Discount factor
        self.update_target_counter = 0

    def select_action(self, state: torch.Tensor, mask: np.ndarray) -> int:
        """
        state: 9x9 array
        mask: 81-length binary array (1 for legal move, 0 for illegal)
        """
        if random.random() < self.epsilon:
            legal_indices = np.where(mask == 1)[0]
            return random.choice(legal_indices)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).cuda()
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()
            
            # Mask illegal moves by setting Q-value very low
            q_values[mask == 0] = -np.inf
            return np.argmax(q_values)

    def train_step(self, batch_size: int = 64) -> None:
        # 1. Only train if we have enough experiences in memory
        if len(self.memory.memory) < batch_size:
            return

        # 2. Sample a random batch of transitions;
        # we need to do this since episodes are highly correlated
        transitions = self.memory.sample(batch_size)
        state_b, action_b, reward_b, next_state_b, done_b = zip(*transitions)

        # Convert to PyTorch Tensors
        # State shape: [batch, 1, 9, 9]
        state_t = torch.FloatTensor(np.array(state_b)).unsqueeze(1).cuda()
        next_state_t = torch.FloatTensor(np.array(next_state_b)).unsqueeze(1).cuda()
        action_t = torch.LongTensor(action_b).unsqueeze(1).cuda()
        reward_t = torch.FloatTensor(reward_b).cuda()
        done_t = torch.FloatTensor(done_b).cuda()

        # 3. Get current Q-values for the actions taken
        # policy_net(state_t) returns [64, 81]. gather(1, action_t) picks the specific move.
        current_q_values = self.policy_net(state_t).gather(1, action_t).squeeze(1)

        # 4. Compute the "Target" Q-values using the Target Network
        with torch.no_grad():
            # Look ahead: what is the best possible value of the next state?
            max_next_q = self.target_net(next_state_t).max(1)[0]
            # Bellman Equation: If done, target is just reward. Else, reward + discounted future.
            expected_q_values = reward_t + (self.gamma * max_next_q * (1 - done_t))

        # 5. Calculate Loss (Huber Loss is more robust than MSE for RL)
        loss = nn.SmoothL1Loss()(current_q_values, expected_q_values)

        # 6. Optimize the Model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents the "exploding gradient" problem common in RL
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 7. Soft Update of Target Network (Optional but recommended)
        # This slowly moves the target_net toward the policy_net for stability
        self.update_target_counter += 1
        if self.update_target_counter % 1000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

class UltimateTicTacToeGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Ultimate Tic-Tac-Toe")
        
        self.board = np.zeros((9, 9), dtype=int)
        self.buttons: List[List[tk.Button]] = [[None for _ in range(9)] for _ in range(9)]
        self.frames: List[List[tk.Frame]] = [[None for _ in range(3)] for _ in range(3)]
        
        self.current_player = 2  # Human starts (O)
        self.active_block: Optional[Tuple[int, int]] = None  # None means free move
        
        self._build_gui()
        self._highlight_active_area()

    def _build_gui(self) -> None:
        for br in range(3):
            for bc in range(3):
                frame = tk.Frame(self.root, highlightbackground="gray", 
                                 highlightthickness=5, bd=0)
                frame.grid(row=br, column=bc, padx=5, pady=5)
                self.frames[br][bc] = frame

                for r in range(3):
                    for c in range(3):
                        ar, ac = br * 3 + r, bc * 3 + c
                        btn = tk.Button(frame, text="", font=('Arial', 14), width=4, height=2,
                                        command=lambda row=ar, col=ac: self.handle_move(row, col))
                        btn.grid(row=r, column=c)
                        self.buttons[ar][ac] = btn

    def _highlight_active_area(self) -> None:
        """Visual feedback: Yellow for the valid block, gray for others."""
        for br in range(3):
            for bc in range(3):
                if self.active_block is None or self.active_block == (br, bc):
                    self.frames[br][bc].config(highlightbackground="yellow")
                else:
                    self.frames[br][bc].config(highlightbackground="gray")

    def handle_move(self, r: int, c: int) -> None:
        # 1. Logic: Is the move valid?
        block_r, block_c = r // 3, c // 3
        
        if self.active_block is not None and (block_r, block_c) != self.active_block:
            return # Blocked: Must play in the highlighted square

        if self.board[r, c] == 0:
            # 2. Execute Move
            self.board[r, c] = self.current_player
            char = "O" if self.current_player == 2 else "X"
            color = "blue" if self.current_player == 2 else "red"
            self.buttons[r][c].config(text=char, state="disabled", disabledforeground=color)

            # 3. Determine Next Active Block
            # The relative position inside the 3x3 determines the next block
            next_br, next_bc = r % 3, c % 3
            
            # Check if that block is full
            block_slice = self.board[next_br*3:(next_br+1)*3, next_bc*3:(next_bc+1)*3]
            if not np.any(block_slice == 0):
                self.active_block = None # Free move!
            else:
                self.active_block = (next_br, next_bc)

            # 4. Switch Player & Update UI
            self.current_player = 1 if self.current_player == 2 else 2
            self._highlight_active_area()
            
            # If AI turn, trigger it
            if self.current_player == 1:
                self.root.after(600, self.ai_move)

    def ai_move(self) -> None:
        # Filter for legal moves based on active_block
        if self.active_block:
            br, bc = self.active_block
            legal_moves = [(r, c) for r in range(br*3, (br+1)*3) 
                           for c in range(bc*3, (bc+1)*3) if self.board[r, c] == 0]
        else:
            legal_moves = list(zip(*np.where(self.board == 0)))

        if legal_moves:
            move = random.choice(legal_moves)
            self.handle_move(move[0], move[1])

    def run(self) -> None:
        self.root.mainloop()

import numpy as np
from typing import List, Tuple, Optional

class UltimateGameLogic:
    def __init__(self) -> None:
        # Full 9x9 board: 0=Empty, 1=AI, -1=Human
        self.board: np.ndarray = np.zeros((9, 9), dtype=float)
        # 3x3 tracking to see which mega-blocks are won
        self.mega_board: np.ndarray = np.zeros((3, 3), dtype=float)
        self.done: bool = False

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        self.mega_board.fill(0)
        self.done = False
        return self.board.copy()

    def check_3x3_win(self, grid: np.ndarray) -> float:
        """Checks a 3x3 slice for a winner. Returns 1, -1, or 0."""
        # Rows and Columns
        for i in range(3):
            if np.sum(grid[i, :]) == 3 or np.sum(grid[:, i]) == 3: return 1
            if np.sum(grid[i, :]) == -3 or np.sum(grid[:, i]) == -3: return -1
        # Diagonals
        diag1 = np.trace(grid)
        diag2 = np.trace(np.fliplr(grid))
        if diag1 == 3 or diag2 == 3: return 1
        if diag1 == -3 or diag2 == -3: return -1
        return 0

    def get_valid_move_mask(self, active_block: Optional[Tuple[int, int]]) -> np.ndarray:
        mask = np.zeros(81, dtype=int)
        if active_block is None:
            # Free move: Any empty spot on the board
            mask[self.board.flatten() == 0] = 1
        else:
            br, bc = active_block
            # Narrow mask to just the 3x3 sub-grid
            for r in range(br * 3, (br + 1) * 3):
                for c in range(bc * 3, (bc + 1) * 3):
                    if self.board[r, c] == 0:
                        mask[r * 9 + c] = 1
            
            # Fallback: if the sent-to block is full, allow move anywhere
            if np.sum(mask) == 0:
                mask[self.board.flatten() == 0] = 1
        return mask

    def play_move(self, action: int, player: int) -> Tuple[np.ndarray, float, bool, Optional[Tuple[int, int]]]:
        """
        Executes a move.
        Returns: (next_state, reward, done, next_active_block)
        """
        r, c = divmod(action, 9)
        self.board[r, c] = player
        
        br, bc = r // 3, c // 3
        reward = 0.0
        
        # 1. Check if this move won the local 3x3 block
        if self.mega_board[br, bc] == 0:
            local_win = self.check_3x3_win(self.board[br*3:(br+1)*3, bc*3:(bc+1)*3])
            if local_win != 0:
                self.mega_board[br, bc] = local_win
                reward += 1.0 if local_win == player else -1.0

        # 2. Check if this move won the global 9x9 game
        global_win = self.check_3x3_win(self.mega_board)
        if global_win != 0:
            self.done = True
            reward += 10.0 if global_win == player else -10.0
        elif not np.any(self.board == 0):
            self.done = True # Draw
            reward += 0.5

        # 3. Determine the next active block
        # The relative position inside the 3x3 dictates the next block
        next_br, next_bc = r % 3, c % 3
        
        # If the target block is already won or full, next player gets a free move
        if self.mega_board[next_br, next_bc] != 0 or \
           not np.any(self.board[next_br*3:(next_br+1)*3, next_bc*3:(next_bc+1)*3] == 0):
            next_block = None
        else:
            next_block = (next_br, next_bc)

        return self.board.copy(), reward, self.done, next_block

def train_ultimate_dqn(agent: UltimateAgent, episodes: int = 10000, batch_size: int = 64):
    game = UltimateGameLogic() # Helper class to handle board rules
    
    # Epsilon decay parameters
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    agent.epsilon = epsilon_start

    print(f"Starting training for {episodes} episodes...")

    for episode in tqdm.tqdm(range(episodes)):
        state = game.reset()
        active_block = None
        done = False
        total_reward = 0
        
        while not done:
            # 1. Get move mask based on Ultimate rules
            mask = game.get_valid_move_mask(active_block)
            
            # 2. Agent selects an action
            action = agent.select_action(state, mask)
            
            # 3. Execute move in the game environment
            # next_state, reward, done, next_block = game.step(action)
            next_state, reward, done, next_block = game.play_move(action, player=1)
            
            # 4. Store experience in Replay Memory
            agent.memory.push(state, action, reward, next_state, done)
            
            # 5. Perform a learning step
            agent.train_step(batch_size)
            
            # 6. Update state and block
            state = next_state
            active_block = next_block
            total_reward += reward

        # Decay exploration rate
        agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)

        # Logging
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{episodes} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

        # Periodically save the model
        if (episode + 1) % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f"ultimate_dqn_{episode+1}.pth")

    print("Training Complete!")


if __name__ == "__main__":
    agent = UltimateAgent()

    train_ultimate_dqn(agent, episodes=10000)
    # UltimateTicTacToeGUI().run()
