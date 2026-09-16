import torch
from tqdm import tqdm
from torch.distributions import Categorical
from src.core.types import StepRecord
from src.model.returns import compute_returns_for_model
from src.training.metrics import MetricsAggregator
from src.losses.composed_loss import ComposedLoss
from src.model.orientation import orient

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
    oriented=False,
):
    """`oriented`: observations are flipped to the mover's view before the model
    (shared network). The environment still reports the absolute board."""
    models = {WHITE: white_model, BLACK: black_model}
    steps_data = []

    for _ in range(num_steps):
        players = env_step.current_player
        obs_in = orient(env_step.obs, players) if oriented else env_step.obs
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
                obs_in[mask], env_step.legal_actions_mask[mask]
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


def _weighted_mean(result_a, result_b, key, count_key):
    """Step-weighted mean of a per-model metric, None when neither model has steps."""
    total = result_a[count_key] + result_b[count_key]
    if total == 0:
        return None
    acc = 0.0
    for r in (result_a, result_b):
        if r[count_key]:
            acc += r[key] * r[count_key]
    return acc / total


def _compute_bootstrap(env_step, white_model, black_model, model_id, oriented=False):
    num_envs = env_step.obs.shape[0]
    bootstrap = torch.zeros(num_envs)
    models = {WHITE: white_model, BLACK: black_model}
    opponent_id = WHITE if model_id == BLACK else BLACK
    obs_in = orient(env_step.obs, env_step.current_player) if oriented else env_step.obs

    with torch.no_grad():
        same_player = env_step.current_player == model_id
        opponent_player = env_step.current_player == opponent_id

        if same_player.any():
            _, v = models[model_id](
                obs_in[same_player], env_step.legal_actions_mask[same_player]
            )
            bootstrap[same_player] = v.squeeze(-1)

        if opponent_player.any():
            _, v = models[opponent_id](
                obs_in[opponent_player],
                env_step.legal_actions_mask[opponent_player],
            )
            bootstrap[opponent_player] = -v.squeeze(-1)

    return bootstrap


_EMPTY_STATS = {
    "entropy_raw": 0.0, "entropy_normalized": 0.0, "step_count": 0,
    "entropy_normalized_game": None, "entropy_normalized_puzzle": None,
    "game_step_count": 0, "puzzle_step_count": 0,
}


def _gather_side(steps_data, model_returns, model_id, puzzle_env_mask=None):
    """Concatenate one side's own-step tensors over the rollout; None if it never moved."""
    parts = {"log_prob": [], "value": [], "ret": [], "entropy": [], "num_legal": [], "is_puzzle": []}
    for i, step in enumerate(steps_data):
        is_mine = step.player == model_id
        if not is_mine.any():
            continue
        white = model_id == WHITE
        parts["ret"].append(model_returns[i][is_mine].detach())
        parts["value"].append((step.value_white if white else step.value_black).squeeze(-1)[is_mine])
        parts["log_prob"].append((step.log_prob_white if white else step.log_prob_black)[is_mine])
        parts["entropy"].append((step.entropy_white if white else step.entropy_black)[is_mine])
        parts["num_legal"].append((step.num_legal_white if white else step.num_legal_black)[is_mine])
        parts["is_puzzle"].append(
            puzzle_env_mask[is_mine] if puzzle_env_mask is not None
            else torch.zeros(int(is_mine.sum()), dtype=torch.bool)
        )
    if not parts["log_prob"]:
        return None
    g = {k: torch.cat(v) for k, v in parts.items()}
    g["normalized_entropy"] = g["entropy"] / torch.clamp(torch.log(g["num_legal"]), min=1e-8)
    return g


def _side_stats(g):
    if g is None:
        return dict(_EMPTY_STATS)
    ne = g["normalized_entropy"]
    game_ent, puzzle_ent = ne[~g["is_puzzle"]], ne[g["is_puzzle"]]
    return {
        "entropy_raw": g["entropy"].mean().item(),
        "entropy_normalized": ne.mean().item(),
        "step_count": len(ne),
        # Entropy split by board type: puzzle boards are expected to sharpen; the
        # opening boards tell whether real play has gone rigid.
        "entropy_normalized_game": game_ent.mean().item() if len(game_ent) else None,
        "entropy_normalized_puzzle": puzzle_ent.mean().item() if len(puzzle_ent) else None,
        "game_step_count": int(len(game_ent)),
        "puzzle_step_count": int(len(puzzle_ent)),
    }


def _loss(gathered_list, entropy_coef):
    """A2C loss over the concatenation of one or more sides' gathered tensors."""
    gs = [g for g in gathered_list if g is not None]
    if not gs:
        return None
    log_probs = torch.cat([g["log_prob"] for g in gs])
    values = torch.cat([g["value"] for g in gs])
    returns = torch.cat([g["ret"] for g in gs])
    norm_ent = torch.cat([g["normalized_entropy"] for g in gs])
    advantages = (returns - values).detach()
    policy_loss = -(log_probs * advantages).mean()
    value_loss = (values - returns).pow(2).mean()
    return ComposedLoss().compute(
        policy_loss=policy_loss, value_loss=value_loss,
        entropy_bonus=norm_ent.mean(), entropy_coef=entropy_coef,
    )["total_loss"]


def _optimizer_step(model, optimizer, total_loss, grad_clip):
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=grad_clip)
    optimizer.step()


def _backpropagate_for_model(
    model, optimizer, steps_data, model_returns, model_id,
    entropy_coef=0.01, grad_clip=1.0, puzzle_env_mask=None,
):
    """Legacy per-model update (two separate networks)."""
    g = _gather_side(steps_data, model_returns, model_id, puzzle_env_mask)
    stats = _side_stats(g)
    total_loss = _loss([g], entropy_coef)
    if total_loss is None:
        return {"total_loss": torch.tensor(0.0), **stats}
    _optimizer_step(model, optimizer, total_loss, grad_clip)
    return {"total_loss": total_loss, **stats}


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
    lr_decay_factor=0.5,
    min_lr=3e-5,
    terminal_rewards=None,
    gamma=0.99,
    entropy_coef=0.01,
    grad_clip=1.0,
    shaping_scale=1.0,
    eval_fn=None,
    eval_interval=50,
    log_interval=10,
):
    """Run A2C self-play.

    Two modes. Legacy: separate `white_model` / `black_model` with their optimizers.
    Shared (AlphaZero-style): pass `black_model=None`; `white_model` then plays both
    colours on observations oriented to the mover, and both sides' samples update it
    in one backward pass per update with `optimizer_white`.

    `eval_fn(white_model, black_model) -> dict` is called every `eval_interval`
    updates and after the last one; its keys are merged into the log entry for
    that update (a log entry is created for it if none is due)."""
    shared = black_model is None
    if shared:
        black_model = white_model
        optimizer_black = None
    if eval_interval <= 0:
        raise ValueError(f"eval_interval must be a positive integer, got {eval_interval}")
    if metrics is None:
        metrics = MetricsAggregator()
    if terminal_rewards is None:
        terminal_rewards = {"win": 2.0, "loss": -2.0, "draw": -0.5}

    num_envs = envs.num_envs
    num_puzzle_envs = getattr(envs, "num_puzzle_envs", 0)
    puzzle_env_mask = torch.arange(num_envs) < num_puzzle_envs
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
            oriented=shared,
        )

        bootstrap_white = _compute_bootstrap(
            env_step, white_model, black_model, model_id=WHITE, oriented=shared
        )
        bootstrap_black = _compute_bootstrap(
            env_step, white_model, black_model, model_id=BLACK, oriented=shared
        )

        returns_white = compute_returns_for_model(
            steps_data, model_id=WHITE, bootstrap_value=bootstrap_white, gamma=gamma,
            final_material=env_step.material, shaping_scale=shaping_scale,
        )
        returns_black = compute_returns_for_model(
            steps_data, model_id=BLACK, bootstrap_value=bootstrap_black, gamma=gamma,
            final_material=env_step.material, shaping_scale=shaping_scale,
        )

        if shared:
            g_w = _gather_side(steps_data, returns_white, WHITE, puzzle_env_mask)
            g_b = _gather_side(steps_data, returns_black, BLACK, puzzle_env_mask)
            loss = _loss([g_w, g_b], entropy_coef)
            if loss is not None:
                _optimizer_step(white_model, optimizer_white, loss, grad_clip)
            result_w = {"total_loss": loss if loss is not None else torch.tensor(0.0), **_side_stats(g_w)}
            result_b = {"total_loss": torch.tensor(0.0), **_side_stats(g_b)}
        else:
            result_w = _backpropagate_for_model(
                white_model, optimizer_white, steps_data, returns_white, model_id=WHITE,
                entropy_coef=entropy_coef, grad_clip=grad_clip, puzzle_env_mask=puzzle_env_mask,
            )
            result_b = _backpropagate_for_model(
                black_model, optimizer_black, steps_data, returns_black, model_id=BLACK,
                entropy_coef=entropy_coef, grad_clip=grad_clip, puzzle_env_mask=puzzle_env_mask,
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
                game=_weighted_mean(result_w, result_b, "entropy_normalized_game", "game_step_count"),
                puzzle=_weighted_mean(result_w, result_b, "entropy_normalized_puzzle", "puzzle_step_count"),
            )
        losses.append(total_loss)

        pbar.set_postfix(loss=total_loss)

        if episode % lr_decay_interval == 0:
            for opt in [optimizer_white, optimizer_black]:
                if opt is not None:
                    opt.param_groups[0]["lr"] = max(min_lr, opt.param_groups[0]["lr"] * lr_decay_factor)

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
