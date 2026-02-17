"""
Chess DDQN Training with Parallel Multi-Environment Support.

Uses multiprocessing to run N chess environments in parallel worker
processes.  Model inference and training happen in the main process;
environment stepping (the CPU bottleneck) runs concurrently across cores.
"""

import torch
from torch import nn
from torch.nn.functional import mse_loss
import numpy as np
import chess
import random
from copy import deepcopy
from tqdm import tqdm
from pettingzoo.classic import chess_v6
import multiprocessing as mp


# --- 1. HYPERPARAMETERS ---
NUM_ENVS = 8
NUM_EPISODES = 55
BATCH_SIZE = 128
BUFFER_CAPACITY = 20000
LEARNING_RATE = 1e-5
GAMMA = 0.99
TAU = 0.005
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 55
MIN_BUFFER_SIZE = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. MODEL ARCHITECTURE ---
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return torch.relu(out)


class ChessPolicy(nn.Module):
    def __init__(self, num_res_blocks=4):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(111, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.res_tower = nn.Sequential(*[ResBlock(128) for _ in range(num_res_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4672),
        )

    def forward(self, obs, mask=None):
        x = self.conv_input(obs)
        x = self.res_tower(x)
        logits = self.policy_head(x)
        if mask is not None:
            logits = torch.where(
                mask.bool(), logits, torch.tensor(-1e9).to(logits.device)
            )
        return logits


# --- 3. REPLAY BUFFER ---
class ReplayBuffer:
    """Replay buffer with explicit next-state storage for multi-env correctness."""

    def __init__(self, capacity, obs_shape=(111, 8, 8)):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.masks = np.zeros((capacity, 4672), dtype=np.bool_)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_masks = np.zeros((capacity, 4672), dtype=np.bool_)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.pos = 0
        self.size = 0

    def push(self, obs, mask, action, reward, next_obs, next_mask, done):
        self.obs[self.pos] = obs
        self.masks[self.pos] = mask
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.next_masks[self.pos] = next_mask
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):
        indices = np.random.randint(0, self.size, size=batch_size)
        b_obs = torch.from_numpy(self.obs[indices]).float().to(device)
        b_mask = torch.from_numpy(self.masks[indices]).to(device)
        b_action = torch.from_numpy(self.actions[indices]).unsqueeze(1).to(device)
        b_reward = torch.from_numpy(self.rewards[indices]).unsqueeze(1).to(device)
        b_next_obs = torch.from_numpy(self.next_obs[indices]).float().to(device)
        b_next_mask = torch.from_numpy(self.next_masks[indices]).to(device)
        b_done = torch.from_numpy(self.dones[indices]).unsqueeze(1).float().to(device)
        return b_obs, b_mask, b_action, b_reward, b_next_obs, b_next_mask, b_done


# --- 4. UTILITIES ---
PIECES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def _material_eval(board):
    """Raw material evaluation (White - Black) / 10."""
    w = sum(len(board.pieces(p, chess.WHITE)) * PIECES[p] for p in PIECES)
    b = sum(len(board.pieces(p, chess.BLACK)) * PIECES[p] for p in PIECES)
    return (w - b) / 10.0


def calculate_reward(board, done):
    """Compute reward from White's perspective."""
    if done:
        if board.is_checkmate():
            return -10.0 if board.turn == chess.WHITE else 10.0
        return 0.0
    return _material_eval(board)


def select_actions(model, obs_batch, mask_batch, eps=0.1):
    """Epsilon-greedy action selection on a batch of observations."""
    model.eval()
    with torch.no_grad():
        if random.random() < eps:
            actions = []
            for i in range(mask_batch.shape[0]):
                legal = np.flatnonzero(mask_batch[i].cpu().numpy())
                actions.append(random.choice(legal))
            return actions
        else:
            logits = model(obs_batch, mask_batch)
            return torch.argmax(logits, dim=1).cpu().tolist()


# --- 5. DDQN TRAINING STEP ---
def train_step(memory, model, target_model, optimizer, batch_size, device):
    if memory.size < batch_size * 5:
        return 0.0

    b_obs, b_mask, b_action, b_reward, b_n_obs, b_n_mask, b_done = memory.sample(
        batch_size, device
    )

    model.train()
    current_q = model(b_obs, b_mask).gather(1, b_action)

    with torch.no_grad():
        next_actions = torch.argmax(model(b_n_obs, b_n_mask), dim=1, keepdim=True)
        max_next_q = target_model(b_n_obs, b_n_mask).gather(1, next_actions)
        target_q = b_reward + GAMMA * (1 - b_done) * max_next_q

    loss = mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# --- 6. PARALLEL ENVIRONMENT WORKERS ---
def _env_worker(conn):
    """
    Worker process that owns one AEC chess environment.

    Protocol (main sends -> worker replies):
        ("reset", None)            -> {obs_hwc, mask, material_eval}
        ("step_pair", action: int) -> {done, ...}
        ("close", None)            -> worker exits
    """
    env = chess_v6.env()

    while True:
        try:
            cmd, data = conn.recv()
        except EOFError:
            break

        if cmd == "reset":
            env.reset()
            obs, _, _, _, _ = env.last()
            board = env.unwrapped.board
            conn.send(
                {
                    "obs_hwc": np.ascontiguousarray(obs["observation"]),
                    "mask": np.ascontiguousarray(obs["action_mask"]),
                    "material_eval": _material_eval(board),
                }
            )

        elif cmd == "step_pair":
            # --- White steps ---
            env.step(data)
            obs, _, term, trunc, _ = env.last()
            board = env.unwrapped.board

            if term or trunc:
                conn.send(
                    {
                        "done": True,
                        "terminal_eval": calculate_reward(board, True),
                    }
                )
                continue

            # --- Black plays a random legal move ---
            b_mask = obs["action_mask"]
            b_legal = np.flatnonzero(b_mask)
            b_action = int(np.random.choice(b_legal))
            env.step(b_action)

            obs, _, term, trunc, _ = env.last()
            board = env.unwrapped.board

            if term or trunc:
                conn.send(
                    {
                        "done": True,
                        "terminal_eval": calculate_reward(board, True),
                    }
                )
            else:
                conn.send(
                    {
                        "done": False,
                        "obs_hwc": np.ascontiguousarray(obs["observation"]),
                        "mask": np.ascontiguousarray(obs["action_mask"]),
                        "material_eval": _material_eval(board),
                    }
                )

        elif cmd == "close":
            env.close()
            break

    conn.close()


class ParallelChessEnvs:
    """Manages N chess AEC environments across worker processes."""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.conns = []
        self.procs = []
        for _ in range(num_envs):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=_env_worker, args=(child_conn,), daemon=True)
            p.start()
            child_conn.close()  # parent only uses its end
            self.conns.append(parent_conn)
            self.procs.append(p)

    # --- public API -------------------------------------------------

    def reset_all(self):
        """Reset every environment; returns list of result dicts."""
        for c in self.conns:
            c.send(("reset", None))
        return [c.recv() for c in self.conns]

    def step_pairs(self, indices, actions):
        """
        Send White actions to the specified envs (by index).
        Each worker steps White, then plays a random Black move.
        All workers run in parallel; results are collected afterward.

        Returns a dict  {env_index: result_dict}.
        """
        for idx, action in zip(indices, actions):
            self.conns[idx].send(("step_pair", action))
        return {idx: self.conns[idx].recv() for idx in indices}

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except BrokenPipeError:
                pass
        for p in self.procs:
            p.join(timeout=5)


# --- 7. TRAINING LOOP ---
def train():
    envs = ParallelChessEnvs(NUM_ENVS)

    model_white = ChessPolicy().to(DEVICE)
    target_white = deepcopy(model_white).to(DEVICE)
    target_white.eval()
    optimizer = torch.optim.Adam(model_white.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(BUFFER_CAPACITY)

    losses = []
    pbar = tqdm(range(NUM_EPISODES))

    try:
        for episode in pbar:
            epsilon = max(EPS_END, EPS_START - (episode / EPS_DECAY_EPISODES))

            # Reset all environments (parallel across workers)
            init_results = envs.reset_all()

            active = [True] * NUM_ENVS

            # Current White observation / mask / material eval per env
            # (populated from reset, then updated after each step_pair)
            curr_obs_hwc = [r["obs_hwc"] for r in init_results]
            curr_mask = [r["mask"] for r in init_results]
            curr_eval = [r["material_eval"] for r in init_results]

            # Delayed-push bookkeeping
            prev_obs_chw = [None] * NUM_ENVS
            prev_mask = [None] * NUM_ENVS
            prev_action = [None] * NUM_ENVS
            prev_reward_base = [0.0] * NUM_ENVS

            while any(active):
                # ------------------------------------------------------
                # Phase 1: Push pending transitions & collect active envs
                # ------------------------------------------------------
                white_indices = []
                batch_obs_hwc = []
                batch_masks = []

                for i in range(NUM_ENVS):
                    if not active[i]:
                        continue

                    curr_chw = np.ascontiguousarray(curr_obs_hwc[i].transpose(2, 0, 1))

                    # Push the *previous* step's non-terminal transition
                    # (current obs is its next_obs).
                    if prev_obs_chw[i] is not None:
                        step_rew = curr_eval[i] - prev_reward_base[i]
                        memory.push(
                            prev_obs_chw[i],
                            prev_mask[i],
                            prev_action[i],
                            step_rew,
                            curr_chw,
                            curr_mask[i],
                            False,
                        )

                    # Remember this step's pre-move state
                    prev_obs_chw[i] = curr_chw
                    prev_mask[i] = curr_mask[i]
                    prev_reward_base[i] = curr_eval[i]

                    white_indices.append(i)
                    batch_obs_hwc.append(curr_obs_hwc[i])
                    batch_masks.append(curr_mask[i])

                if not white_indices:
                    break

                # ------------------------------------------------------
                # Phase 2: Batched model inference (main process / GPU)
                # ------------------------------------------------------
                obs_batch = (
                    torch.from_numpy(np.stack(batch_obs_hwc))
                    .float()
                    .permute(0, 3, 1, 2)
                    .to(DEVICE)
                )
                mask_batch = torch.from_numpy(np.stack(batch_masks)).to(DEVICE)
                actions = select_actions(
                    model_white,
                    obs_batch,
                    mask_batch,
                    eps=epsilon,
                )

                # ------------------------------------------------------
                # Phase 3: Step all envs IN PARALLEL (workers do the work)
                # ------------------------------------------------------
                results = envs.step_pairs(white_indices, actions)

                for j, idx in enumerate(white_indices):
                    prev_action[idx] = actions[j]
                    result = results[idx]

                    if result["done"]:
                        done_rew = result["terminal_eval"] - prev_reward_base[idx]
                        memory.push(
                            prev_obs_chw[idx],
                            prev_mask[idx],
                            prev_action[idx],
                            done_rew,
                            prev_obs_chw[idx],
                            prev_mask[idx],  # don't matter
                            True,
                        )
                        prev_obs_chw[idx] = None
                        active[idx] = False
                    else:
                        # Update current obs for next iteration
                        curr_obs_hwc[idx] = result["obs_hwc"]
                        curr_mask[idx] = result["mask"]
                        curr_eval[idx] = result["material_eval"]

            # ----------------------------------------------------------
            # Training after each episode
            # ----------------------------------------------------------
            if memory.size > MIN_BUFFER_SIZE:
                for _ in range(NUM_ENVS):
                    loss = train_step(
                        memory,
                        model_white,
                        target_white,
                        optimizer,
                        BATCH_SIZE,
                        DEVICE,
                    )
                    losses.append(loss)

            # Soft-update target network
            for tp, op in zip(target_white.parameters(), model_white.parameters()):
                tp.data.copy_(TAU * op.data + (1.0 - TAU) * tp.data)

            pbar.set_postfix(
                loss=np.mean(losses[-10:]) if losses else 0,
                mem=memory.size,
                eps=f"{epsilon:.2f}",
            )

    finally:
        envs.close()

    return model_white, losses


# --- 8. ENTRY POINT ---
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    trained_model, training_losses = train()
    torch.save(trained_model.state_dict(), "chess_ddqn.pth")
    print("Training complete. Model saved to chess_ddqn.pth")
