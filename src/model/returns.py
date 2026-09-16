import torch
from src.core.types import StepRecord

WHITE = 1  # OpenSpiel convention: player 1 = white
BLACK = 0


def compute_returns_for_model(
    steps: list[StepRecord],
    model_id: int,
    bootstrap_value: torch.Tensor,
    gamma: float,
    final_material: torch.Tensor,
    shaping_scale: float = 1.0,
) -> list[torch.Tensor]:
    """Discounted returns for one model with potential-based material shaping.

    Each own step receives the shaped reward
        F = shaping_scale * (gamma * Phi(next own position) - Phi(this position))
    where Phi is material from the model's perspective (Ng, Harada & Russell 1999).
    The terminal position has Phi = 0, so shaping telescopes to zero over a full
    game and the terminal reward is the only net reward. Opponent steps pass the
    running return through unchanged; `done` on any step cuts the chain for both
    players and substitutes that player's terminal reward.

    Args:
        steps: StepRecords from rollout collection (shared by both models).
        model_id: 1 for white, 0 for black (OpenSpiel convention).
        bootstrap_value: [num_envs] value estimate of the final observed position.
        gamma: Discount per own decision.
        final_material: [num_envs] white-view material of the final observed position.
        shaping_scale: Multiplier on the shaping term; 0 disables shaping.

    Returns:
        List of [num_envs] tensors, one per step (entries at opponent steps hold
        the running return and are masked out by the caller).
    """
    sign = 1.0 if model_id == WHITE else -1.0
    R = bootstrap_value.clone()
    phi_next = sign * final_material.clone()
    returns = [None] * len(steps)

    for i in reversed(range(len(steps))):
        step = steps[i]
        is_mine = step.player == model_id
        done_f = step.done.float()

        terminal_r = step.terminal_r_white if model_id == WHITE else step.terminal_r_black
        R = R * (1.0 - done_f) + terminal_r          # game end: future value := terminal reward
        phi_next = phi_next * (1.0 - done_f)         # terminal position has zero potential

        phi = sign * step.potential_white
        shaped = shaping_scale * (gamma * phi_next - phi)
        new_R = shaped + gamma * R
        R = torch.where(is_mine, new_R, R)
        phi_next = torch.where(is_mine, phi, phi_next)

        returns[i] = R.clone()

    return returns


def compute_gae_for_model(
    steps: list[StepRecord],
    model_id: int,
    bootstrap_value: torch.Tensor,
    gamma: float,
    lam: float,
    final_material: torch.Tensor,
    shaping_scale: float = 1.0,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """GAE(gamma, lambda) advantages and value targets for one model's own steps.

    Same walk as `compute_returns_for_model` (shaped reward, opponent steps pass
    through, `done` cuts the chain with the terminal reward), but instead of the
    return it accumulates TD residuals (Schulman et al. 2016, eq. 16):
        delta_t = shaped_t + gamma * V(next own position) - V_old(s_t)
        A_t     = delta_t + gamma * lambda * A_(next own step)
    lambda = 1 gives A = R - V_old (today's advantage); lambda = 0 gives the
    one-step residual. Targets are A + V_old, so at lambda = 1 they are the returns.

    Returns:
        (advantages, targets): lists of [num_envs] tensors, one per step; entries at
        opponent steps are 0 and must be masked out by the caller.
    """
    sign = 1.0 if model_id == WHITE else -1.0
    next_value = bootstrap_value.clone()
    gae = torch.zeros_like(bootstrap_value)
    phi_next = sign * final_material.clone()
    advantages = [None] * len(steps)
    targets = [None] * len(steps)

    for i in reversed(range(len(steps))):
        step = steps[i]
        is_mine = step.player == model_id
        done_f = step.done.float()
        terminal_r = step.terminal_r_white if model_id == WHITE else step.terminal_r_black

        next_value = next_value * (1.0 - done_f) + terminal_r   # game end: future value := terminal reward
        gae = gae * (1.0 - done_f)                               # nothing after the end leaks back
        phi_next = phi_next * (1.0 - done_f)                     # terminal position has zero potential

        phi = sign * step.potential_white
        shaped = shaping_scale * (gamma * phi_next - phi)
        delta = shaped + gamma * next_value - step.old_value
        new_gae = delta + gamma * lam * gae

        gae = torch.where(is_mine, new_gae, gae)
        advantages[i] = torch.where(is_mine, new_gae, torch.zeros_like(new_gae))
        targets[i] = torch.where(is_mine, new_gae + step.old_value, torch.zeros_like(new_gae))
        next_value = torch.where(is_mine, step.old_value, next_value)
        phi_next = torch.where(is_mine, phi, phi_next)

    return advantages, targets
