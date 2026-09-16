import torch
from tqdm import tqdm
from torch.distributions import Categorical
from src.core.types import StepRecord
from src.model.returns import compute_returns_for_model
from src.training.metrics import MetricsAggregator
from src.losses.composed_loss import ComposedLoss

WHITE = 1
BLACK = 0

RESULT_TO_REWARD = {
    "white_win": {"white": "win", "black": "loss"},
    "black_win": {"white": "loss", "black": "win"},
    "draw": {"white": "draw", "black": "draw"},
}


def _precompute_terminal_rewards(info, num_envs, terminal_rewards, players=None):
    """Terminal reward per side for boards whose episode ended this step.

    `puzzle_miss` (one-move puzzle episode without a mate) pays the mover
    `terminal_rewards["puzzle_miss"]` (default 0) and the other side 0; `players`
    is the side that moved on each board this step.
    """
    tr_white = torch.zeros(num_envs)
    tr_black = torch.zeros(num_envs)
    miss_reward = terminal_rewards.get("puzzle_miss", 0.0)
    for env_idx, result in info.get("game_results", {}).items():
        if result == "puzzle_miss":
            if players is None:
                raise ValueError("puzzle_miss requires the players tensor")
            if int(players[env_idx]) == WHITE:
                tr_white[env_idx] = miss_reward
            else:
                tr_black[env_idx] = miss_reward
            continue
        mapping = RESULT_TO_REWARD[result]
        tr_white[env_idx] = terminal_rewards[mapping["white"]]
        tr_black[env_idx] = terminal_rewards[mapping["black"]]
    return tr_white, tr_black


def _collect_rollout(
    envs,
    white_model,
    black_model,
    env_step,
    num_steps,
    num_envs,
    terminal_rewards,
    metrics,
):
    models = {WHITE: white_model, BLACK: black_model}
    steps_data = []

    for _ in range(num_steps):
        players = env_step.current_player
        pre_move_material = env_step.material.clone()
        white_mask = players == WHITE
        black_mask = players == BLACK

        actions = torch.zeros(num_envs, dtype=torch.long)
        val_w = torch.zeros(num_envs, 1)
        val_b = torch.zeros(num_envs, 1)
        lp_w = torch.zeros(num_envs)
        lp_b = torch.zeros(num_envs)
        ent_w = torch.zeros(num_envs)
        ent_b = torch.zeros(num_envs)
        nleg_w = torch.ones(num_envs)
        nleg_b = torch.ones(num_envs)

        for model_id, mask in [(WHITE, white_mask), (BLACK, black_mask)]:
            if not mask.any():
                continue
            p, v = models[model_id](
                env_step.obs[mask], env_step.legal_actions_mask[mask]
            )
            dist = Categorical(probs=p)
            a = dist.sample()
            actions[mask] = a

            if model_id == WHITE:
                val_w[mask] = v
                lp_w[mask] = dist.log_prob(a)
                ent_w[mask] = dist.entropy()
                nleg_w[mask] = env_step.legal_actions_mask[mask].sum(dim=1).float()
            else:
                val_b[mask] = v
                lp_b[mask] = dist.log_prob(a)
                ent_b[mask] = dist.entropy()
                nleg_b[mask] = env_step.legal_actions_mask[mask].sum(dim=1).float()

        sampled_is_legal = (
            env_step.legal_actions_mask.gather(1, actions.unsqueeze(1)).squeeze(1) > 0
        )
        empty_masks = (env_step.legal_actions_mask.sum(dim=1) == 0).sum().item()
        illegal_samples = (~sampled_is_legal).sum().item()
        metrics.add_step(empty_masks=empty_masks, illegal_samples=illegal_samples)

        env_step = envs.step(actions)

        tr_white, tr_black = _precompute_terminal_rewards(
            env_step.info, num_envs, terminal_rewards, players
        )

        if "game_results" in env_step.info:
            puzzle_boards = env_step.info.get("puzzle_boards", {})
            for env_idx, result in env_step.info["game_results"].items():
                if puzzle_boards.get(env_idx, False):
                    mover_won = (result == "white_win") == (int(players[env_idx]) == WHITE)
                    metrics.add_puzzle_result(solved=result != "puzzle_miss" and mover_won)
                    continue
                metrics.add_terminal_return(tr_white[env_idx].item())
                if result == "white_win":
                    metrics.add_terminal_result(white_win=True)
                elif result == "black_win":
                    metrics.add_terminal_result(black_win=True)
                else:
                    metrics.add_terminal_result(draw=True)

        steps_data.append(
            StepRecord(
                player=players.clone(),
                value_white=val_w,
                value_black=val_b,
                log_prob_white=lp_w,
                log_prob_black=lp_b,
                entropy_white=ent_w,
                entropy_black=ent_b,
                num_legal_white=nleg_w,
                num_legal_black=nleg_b,
                potential_white=pre_move_material,
                done=env_step.done.clone(),
                terminal_r_white=tr_white,
                terminal_r_black=tr_black,
            )
        )

    return steps_data, env_step


def _compute_bootstrap(env_step, white_model, black_model, model_id):
    num_envs = env_step.obs.shape[0]
    bootstrap = torch.zeros(num_envs)
    models = {WHITE: white_model, BLACK: black_model}
    opponent_id = WHITE if model_id == BLACK else BLACK

    with torch.no_grad():
        same_player = env_step.current_player == model_id
        opponent_player = env_step.current_player == opponent_id

        if same_player.any():
            _, v = models[model_id](
                env_step.obs[same_player], env_step.legal_actions_mask[same_player]
            )
            bootstrap[same_player] = v.squeeze(-1)

        if opponent_player.any():
            _, v = models[opponent_id](
                env_step.obs[opponent_player],
                env_step.legal_actions_mask[opponent_player],
            )
            bootstrap[opponent_player] = -v.squeeze(-1)

    return bootstrap


def _backpropagate_for_model(
    model, optimizer, steps_data, model_returns, model_id,
    entropy_coef=0.01, grad_clip=1.0,
):
    filtered_log_probs = []
    filtered_advantages = []
    filtered_values = []
    filtered_returns = []
    filtered_entropies = []
    filtered_num_legal = []

    for i, step in enumerate(steps_data):
        is_mine = step.player == model_id
        if not is_mine.any():
            continue
        ret = model_returns[i][is_mine]
        val_tensor = step.value_white if model_id == WHITE else step.value_black
        val = val_tensor.squeeze(-1)[is_mine]
        lp = (step.log_prob_white if model_id == WHITE else step.log_prob_black)[is_mine]
        ent = (step.entropy_white if model_id == WHITE else step.entropy_black)[is_mine]
        nl = (step.num_legal_white if model_id == WHITE else step.num_legal_black)[is_mine]
        advantage = (ret - val).detach()
        filtered_advantages.append(advantage)
        filtered_log_probs.append(lp)
        filtered_values.append(val)
        filtered_returns.append(ret.detach())
        filtered_entropies.append(ent)
        filtered_num_legal.append(nl)

    if not filtered_log_probs:
        return {
            "total_loss": torch.tensor(0.0),
            "entropy_raw": 0.0,
            "entropy_normalized": 0.0,
            "step_count": 0,
        }

    cat_log_probs = torch.cat(filtered_log_probs)
    cat_advantages = torch.cat(filtered_advantages)
    cat_values = torch.cat(filtered_values)
    cat_returns = torch.cat(filtered_returns)
    cat_entropies = torch.cat(filtered_entropies)
    cat_num_legal = torch.cat(filtered_num_legal)
    log_legal = torch.clamp(torch.log(cat_num_legal), min=1e-8)
    normalized_entropy = cat_entropies / log_legal

    policy_loss = -(cat_log_probs * cat_advantages.detach()).mean()
    value_loss = (cat_values - cat_returns).pow(2).mean()
    entropy_bonus = normalized_entropy.mean()

    loss_composer = ComposedLoss()
    loss_dict = loss_composer.compute(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_bonus=entropy_bonus,
        entropy_coef=entropy_coef,
    )

    total_loss = loss_dict["total_loss"]
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=grad_clip)
    optimizer.step()
    return {
        "total_loss": total_loss,
        "entropy_raw": cat_entropies.mean().item(),
        "entropy_normalized": normalized_entropy.mean().item(),
        "step_count": len(cat_entropies),
    }


def run_chess_training(
    envs,
    white_model,
    black_model,
    optimizer_white,
    optimizer_black,
    metrics: MetricsAggregator = None,
    episodes=2000,
    steps=5,
    lr_decay_interval=100,
    terminal_rewards=None,
    gamma=0.99,
    entropy_coef=0.01,
    grad_clip=1.0,
    shaping_scale=1.0,
    eval_fn=None,
    eval_interval=50,
    log_interval=10,
):
    """Run A2C self-play. `eval_fn(white_model, black_model) -> dict` is called every
    `eval_interval` updates and after the last one; its keys are merged into the log
    entry for that update (a log entry is created for it if none is due)."""
    if eval_interval <= 0:
        raise ValueError(f"eval_interval must be a positive integer, got {eval_interval}")
    if metrics is None:
        metrics = MetricsAggregator()
    if terminal_rewards is None:
        terminal_rewards = {"win": 2.0, "loss": -2.0, "draw": -0.5}

    num_envs = envs.num_envs
    logs = []
    losses = []
    env_step = envs.reset()
    pbar = tqdm(range(1, episodes + 1))

    for episode in pbar:
        steps_data, env_step = _collect_rollout(
            envs,
            white_model,
            black_model,
            env_step,
            steps,
            num_envs,
            terminal_rewards,
            metrics,
        )

        bootstrap_white = _compute_bootstrap(
            env_step, white_model, black_model, model_id=WHITE
        )
        bootstrap_black = _compute_bootstrap(
            env_step, white_model, black_model, model_id=BLACK
        )

        returns_white = compute_returns_for_model(
            steps_data, model_id=WHITE, bootstrap_value=bootstrap_white, gamma=gamma,
            final_material=env_step.material, shaping_scale=shaping_scale,
        )
        returns_black = compute_returns_for_model(
            steps_data, model_id=BLACK, bootstrap_value=bootstrap_black, gamma=gamma,
            final_material=env_step.material, shaping_scale=shaping_scale,
        )

        result_w = _backpropagate_for_model(
            white_model, optimizer_white, steps_data, returns_white, model_id=WHITE,
            entropy_coef=entropy_coef, grad_clip=grad_clip,
        )
        result_b = _backpropagate_for_model(
            black_model, optimizer_black, steps_data, returns_black, model_id=BLACK,
            entropy_coef=entropy_coef, grad_clip=grad_clip,
        )

        total_loss = result_w["total_loss"].item() + result_b["total_loss"].item()
        total_steps = result_w["step_count"] + result_b["step_count"]
        if total_steps > 0:
            metrics.add_entropy(
                raw=(
                    result_w["entropy_raw"] * result_w["step_count"]
                    + result_b["entropy_raw"] * result_b["step_count"]
                )
                / total_steps,
                normalized=(
                    result_w["entropy_normalized"] * result_w["step_count"]
                    + result_b["entropy_normalized"] * result_b["step_count"]
                )
                / total_steps,
            )
        losses.append(total_loss)

        pbar.set_postfix(loss=total_loss)

        if episode % lr_decay_interval == 0:
            for opt in [optimizer_white, optimizer_black]:
                opt.param_groups[0]["lr"] = max(3e-4, opt.param_groups[0]["lr"] * 0.5)

        eval_due = eval_fn is not None and (
            episode % eval_interval == 0 or episode == episodes
        )
        eval_metrics = eval_fn(white_model, black_model) if eval_due else {}

        if episode % log_interval == 0:
            summary = metrics.episode_summary()
            logs.append({"episode": episode, "loss": total_loss, **summary, **eval_metrics})
        elif eval_metrics:
            logs.append({"episode": episode, "loss": total_loss, **eval_metrics})

    if hasattr(envs, "close"):
        envs.close()

    return logs, losses, white_model, black_model
