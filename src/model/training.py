import torch
from pathlib import Path
from tqdm import tqdm
from torch.distributions import Categorical
from src.core.types import StepRecord
from src.model.returns import compute_gae_for_model
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


def behaviour_probs(p, legal, epsilon, alpha, generator=None):
    """Exploration mixture over legal moves (AlphaZero root noise, Silver et al.
    2017 p.14): b = (1 - epsilon) * p + epsilon * Dir(alpha), the Dirichlet drawn over
    the legal moves of each row only. epsilon = 0 returns p. Rows: [n, actions]."""
    if epsilon <= 0:
        return p
    legal = legal.bool()
    noise = torch.zeros_like(p)
    for i in range(p.shape[0]):
        idx = legal[i].nonzero().squeeze(1)
        conc = torch.full((len(idx),), float(alpha))
        gamma = torch._standard_gamma(conc, generator=generator) if generator is not None else torch._standard_gamma(conc)
        noise[i, idx] = gamma / gamma.sum()
    return (1.0 - epsilon) * p + epsilon * noise


OFFPRIOR_THRESHOLD = 0.05   # a pick counts as "against the prior" below this network probability


def guided_probs(p, demos, epsilon):
    """Label-guided behaviour mixture: for each row i with a non-empty demonstration
    set demos[i], b = (1 - eps_i) * p + eps_i * uniform(demos[i]); rows without a
    demonstration keep p. `epsilon` is a scalar or one value per row (label guidance
    and the punisher use different probabilities). Salimans & Chen 2018
    (demonstration-guided starts) turned into a behaviour policy; the update stays
    outcome-driven."""
    b = p.clone()
    for i, demo in enumerate(demos):
        if demo:
            eps = float(epsilon[i]) if hasattr(epsilon, "__len__") else float(epsilon)
            idx = torch.tensor(sorted(demo), dtype=torch.long)
            b[i] = (1.0 - eps) * p[i]
            b[i, idx] += eps / len(idx)
    return b


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
    explore=None,
    guide=None,
    pool=None,
    punish=None,
    referee=None,
):
    """Play `num_steps` plies on every board and record what the update needs.

    Runs under `no_grad`: the rollout stores plain tensors (observation as the
    model saw it, legal mask, action, old log-prob, old value) and the update
    re-runs the network on them. `oriented`: observations are flipped to the
    mover's view before the model (shared network); the environment still
    reports the absolute board.

    `explore`: None, or dict(epsilon, alpha, mask) with `mask` a [num_envs] bool
    tensor of boards whose moves are drawn from the behaviour mixture
    `behaviour_probs`; those steps store log b(a) as `old_log_prob` (the PPO
    ratio is then taken against the distribution the move came from) while the
    logged entropy stays the network's.

    `pool`: None or a `PoolBoards`; on its boards the ply is chosen by the frozen
    opponent whenever the side to move is not the learner's colour. Those plies
    are stored with `learner=False` and leave the batch in `_gather_side`."""
    models = {WHITE: white_model, BLACK: black_model}
    steps_data = []
    # Per board: did the demonstrator's move get played in the current episode? Kept
    # across rollouts on the env (technique curriculum advances on clean episodes only).
    demo_acted = getattr(envs, "_demo_acted", None)
    if demo_acted is None or len(demo_acted) != num_envs:
        demo_acted = [False] * num_envs
        try:
            envs._demo_acted = demo_acted
        except AttributeError:
            pass
    noisy = explore["mask"] if explore is not None and explore.get("epsilon", 0) > 0 else None
    guided = guide["mask"] if guide is not None and guide.get("epsilon", 0) > 0 else None
    # Punisher (change punish-gifts): on its boards, a rules mate or a free capture is
    # the demonstration when the board has no label demonstration, played with
    # punish["epsilon"]. Same mixture machinery as label guidance.
    punishing = punish["mask"] if punish is not None and punish.get("epsilon", 0) > 0 else None
    if noisy is not None and guided is not None:
        raise ValueError("explore (noise) and guide (labels) cannot both be active")

    with torch.no_grad():
        for _ in range(num_steps):
            players = env_step.current_player
            obs_in = orient(env_step.obs, players) if oriented else env_step.obs
            legal = env_step.legal_actions_mask
            pre_move_material = env_step.material.clone()

            actions = torch.zeros(num_envs, dtype=torch.long)
            old_value = torch.zeros(num_envs)
            old_log_prob = torch.zeros(num_envs)
            entropy = torch.zeros(num_envs)
            num_legal = legal.sum(dim=1).float().clamp(min=1.0)
            label_demos = envs.demo_actions(guided) if guided is not None else None
            punish_demos = envs.punish_actions(punishing, punish["threshold"]) if punishing is not None else None
            guided_rows = None
            demos = [set()] * num_envs
            row_eps = torch.zeros(num_envs)
            punish_rows = torch.zeros(num_envs, dtype=torch.bool)
            if label_demos is not None or punish_demos is not None:
                demos = []
                for i in range(num_envs):
                    if label_demos is not None and bool(guided[i]) and label_demos[i]:
                        demos.append(label_demos[i]); row_eps[i] = guide["epsilon"]
                    elif punish_demos is not None and bool(punishing[i]) and punish_demos[i]:
                        demos.append(punish_demos[i]); row_eps[i] = punish["epsilon"]; punish_rows[i] = True
                    else:
                        demos.append(set())
                guided_rows = torch.tensor([bool(d) for d in demos])

            learner = pool.learner_mask(players) if pool is not None else torch.ones(num_envs, dtype=torch.bool)
            if pool is not None:
                # Pool referee (change punish-gifts, arm POOL_REF): the frozen opponent always
                # plays a punishing move when one exists. Its plies are never trained, so no
                # log-prob bookkeeping is needed; only the learner feels the punishment.
                ref_demos = envs.punish_actions(~learner, referee["threshold"]) if referee is not None else None
                for opp_model, ids in pool.opponent_groups(players).items():
                    ids_t = torch.tensor(ids, dtype=torch.long)
                    p, v = opp_model(obs_in[ids_t], legal[ids_t])
                    dist = Categorical(probs=p)
                    a = dist.sample()
                    if ref_demos is not None:
                        picks = 0
                        for j, i in enumerate(ids):
                            if ref_demos[i]:
                                choices = sorted(ref_demos[i])
                                a[j] = choices[int(torch.randint(len(choices), (1,)))]
                                picks += 1
                        metrics.add_referee(count=len(ids), picks=picks)
                    actions[ids_t] = a
                    old_log_prob[ids_t] = dist.log_prob(a)
                    old_value[ids_t] = v.squeeze(-1)
                    entropy[ids_t] = dist.entropy()

            for model_id in (WHITE, BLACK):
                mask = (players == model_id) & learner
                if not mask.any():
                    continue
                p, v = models[model_id](obs_in[mask], legal[mask])
                dist = Categorical(probs=p)
                if noisy is not None and noisy[mask].any():
                    b = p.clone()
                    rows = noisy[mask]
                    b[rows] = behaviour_probs(p[rows], legal[mask][rows], explore["epsilon"], explore["alpha"])
                    bdist = Categorical(probs=b)
                    a = bdist.sample()
                    old_log_prob[mask] = bdist.log_prob(a)
                    picked = p.gather(1, a[:, None]).squeeze(1)[rows]
                    metrics.add_explore(count=int(rows.sum()), offprior=int((picked < OFFPRIOR_THRESHOLD).sum()))
                elif guided_rows is not None and guided_rows[mask].any():
                    env_ids = mask.nonzero().squeeze(1).tolist()
                    row_demos = [demos[e] if guided_rows[e] else set() for e in env_ids]
                    b = guided_probs(p, row_demos, row_eps[mask])
                    bdist = Categorical(probs=b)
                    # Sample the mixture by its components so we know when the demonstrator
                    # fired: with probability epsilon (rows with a demo) a uniform demo move,
                    # otherwise the network's own draw. Same distribution as b; the stored
                    # log-prob is still the mixture's. A network-chosen move that happens to
                    # be the demo move counts as the network's (arm CURSIL2: flagging by
                    # action made every mate "teacher-made" and froze the curriculum).
                    a = dist.sample()
                    fire = torch.rand(len(env_ids)) < row_eps[mask]
                    hits = punish_hits = 0
                    for j, e in enumerate(env_ids):
                        if row_demos[j] and bool(fire[j]):
                            choices = sorted(row_demos[j])
                            a[j] = choices[int(torch.randint(len(choices), (1,)))]
                            if punish_rows[e]:
                                punish_hits += 1
                            else:
                                hits += 1
                            demo_acted[e] = True
                    old_log_prob[mask] = bdist.log_prob(a)
                    label_rows = guided_rows & ~punish_rows
                    if label_rows[mask].any():
                        metrics.add_guide(count=int(label_rows[mask].sum()), demo_picks=hits)
                    if punish_rows[mask].any():
                        metrics.add_punish(count=int(punish_rows[mask].sum()), picks=punish_hits)
                else:
                    a = dist.sample()
                    old_log_prob[mask] = dist.log_prob(a)
                actions[mask] = a
                old_value[mask] = v.squeeze(-1)
                entropy[mask] = dist.entropy()

            sampled_is_legal = legal.gather(1, actions.unsqueeze(1)).squeeze(1) > 0
            empty_masks = (legal.sum(dim=1) == 0).sum().item()
            illegal_samples = (~sampled_is_legal).sum().item()
            metrics.add_step(empty_masks=empty_masks, illegal_samples=illegal_samples)

            env_step = envs.step(actions)

            tr_white, tr_black = _precompute_terminal_rewards(
                env_step.info, num_envs, terminal_rewards, players
            )

            if "game_results" in env_step.info:
                puzzle_boards = env_step.info.get("puzzle_boards", {})
                puzzle_depth = env_step.info.get("puzzle_depth", {})
                finish_boards = env_step.info.get("finish_boards", {})
                pool_done = []
                for env_idx, result in env_step.info["game_results"].items():
                    if pool is not None and env_idx in pool.board_ids:
                        # Pool board: score from the learner's side, by opponent; kept out of
                        # the self-play rates; the board redraws opponent and colour.
                        colour = pool.learner_colour(env_idx)
                        score = 0.5 if result not in ("white_win", "black_win") else float(
                            (result == "white_win") == (colour == WHITE))
                        metrics.add_pool_result(pool.opponent_name(env_idx), score)
                        pool_done.append(env_idx)
                        continue
                    if finish_boards.get(env_idx, False):
                        # Finishing board: success = the record's winner won; kept out of the
                        # game rates like puzzles. Rewards are the normal win/loss/draw.
                        # A labelled (technique) start is counted per label instead of depth.
                        label = env_step.info.get("finish_config", {}).get(env_idx, "")
                        if label:
                            success = env_step.info["finish_success"][env_idx]
                            clean = not demo_acted[env_idx]
                            metrics.add_technique_result(label, success, clean=clean)
                            report_label = getattr(getattr(envs, "finish_sampler", None), "report_label", None)
                            if report_label is not None:
                                report_label(label, success, clean=clean,
                                             depth=env_step.info.get("finish_depth", {}).get(env_idx))
                            demo_acted[env_idx] = False
                        else:
                            metrics.add_finish_result(
                                depth=env_step.info["finish_depth"][env_idx],
                                success=env_step.info["finish_success"][env_idx],
                            )
                        continue
                    if puzzle_boards.get(env_idx, False):
                        mover_won = (result == "white_win") == (int(players[env_idx]) == WHITE)
                        metrics.add_puzzle_result(
                            solved=result != "puzzle_miss" and mover_won,
                            depth=puzzle_depth.get(env_idx, 1),
                        )
                        continue
                    metrics.add_terminal_return(tr_white[env_idx].item())
                    if result == "white_win":
                        metrics.add_terminal_result(white_win=True)
                    elif result == "black_win":
                        metrics.add_terminal_result(black_win=True)
                    else:
                        metrics.add_terminal_result(draw=True)
                if pool is not None and pool_done:
                    pool.redraw(pool_done)
                for env_idx in env_step.info["game_results"]:
                    demo_acted[env_idx] = False

            steps_data.append(
                StepRecord(
                    player=players.clone(),
                    obs=obs_in.clone(),
                    legal_mask=legal.clone(),
                    action=actions,
                    old_log_prob=old_log_prob,
                    old_value=old_value,
                    entropy=entropy,
                    num_legal=num_legal,
                    potential_white=pre_move_material,
                    done=env_step.done.clone(),
                    terminal_r_white=tr_white,
                    terminal_r_black=tr_black,
                    noisy=(noisy.clone() if noisy is not None else
                           (guided_rows.clone() if guided_rows is not None else torch.zeros(num_envs, dtype=torch.bool))),
                    learner=learner.clone(),
                )
            )

    return steps_data, env_step


def _sample_weighted(updates, key):
    """Sample-weighted mean of an update statistic over the models updated; None if absent."""
    pairs = [(u[key], u["sample_count"]) for u in updates if u.get(key) is not None and u["sample_count"]]
    if not pairs:
        return None
    total = sum(c for _, c in pairs)
    return sum(v * c for v, c in pairs) / total


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

_EMPTY_UPDATE = {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                 "clip_fraction": None, "approx_kl": None, "optimizer_steps": 0, "sample_count": 0}


def _gather_side(steps_data, advantages, targets, model_id, puzzle_env_mask=None):
    """Concatenate one side's own-step tensors over the rollout; None if it never moved."""
    keys = ("obs", "legal_mask", "action", "old_log_prob", "old_value", "entropy", "num_legal")
    parts = {k: [] for k in keys + ("adv", "target", "is_puzzle")}
    for i, step in enumerate(steps_data):
        is_mine = step.player == model_id
        if step.learner is not None:
            is_mine = is_mine & step.learner
        if not is_mine.any():
            continue
        for k in keys:
            parts[k].append(getattr(step, k)[is_mine])
        parts["adv"].append(advantages[i][is_mine])
        parts["target"].append(targets[i][is_mine])
        parts["is_puzzle"].append(
            puzzle_env_mask[is_mine] if puzzle_env_mask is not None
            else torch.zeros(int(is_mine.sum()), dtype=torch.bool)
        )
        parts.setdefault("noisy", []).append(
            step.noisy[is_mine] if step.noisy is not None else torch.zeros(int(is_mine.sum()), dtype=torch.bool)
        )
    if not parts["obs"]:
        return None
    g = {k: torch.cat(v) for k, v in parts.items()}
    g["normalized_entropy"] = normalized_entropy(g["entropy"], g["num_legal"])
    return g


def normalized_entropy(entropy, num_legal):
    """entropy / log(legal moves), in [0, 1]. A forced move (one legal move) has
    entropy 0 by definition; dividing its float noise (~1e-7) by a tiny guard used
    to report values like 12, so it is set to 0 explicitly."""
    log_n = torch.log(num_legal)
    return torch.where(num_legal > 1, entropy / torch.clamp(log_n, min=1e-8), torch.zeros_like(entropy))


def _concat_batches(gathered_list):
    gs = [g for g in gathered_list if g is not None]
    if not gs:
        return None
    return {k: torch.cat([g[k] for g in gs]) for k in gs[0]}


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


def _minibatch_indices(n, minibatches, generator=None):
    """Shuffled split of range(n) into at most `minibatches` chunks (never empty)."""
    perm = torch.randperm(n, generator=generator)
    return list(torch.tensor_split(perm, min(minibatches, n)))


def ppo_policy_loss(new_log_prob, old_log_prob, advantages, clip_epsilon):
    """Clipped surrogate (Schulman et al. 2017, eq. 7); `clip_epsilon=None` = plain
    ratio * A, whose gradient at theta = theta_old is the REINFORCE gradient.
    Returns (loss, ratio)."""
    ratio = torch.exp(new_log_prob - old_log_prob)
    if clip_epsilon is None:
        return -(ratio * advantages).mean(), ratio
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return -torch.min(ratio * advantages, clipped * advantages).mean(), ratio


def update_model(
    model, optimizer, batch, *, epochs=1, minibatches=1, clip_epsilon=None,
    normalize_advantage=False, value_coef=1.0, entropy_coef=0.01, grad_clip=1.0,
    generator=None,
):
    """Turn one gathered batch into `epochs * minibatches` optimizer steps.

    Each minibatch re-runs the network on the stored observations, forms the
    probability ratio against the stored log-probs, and minimises
        policy_loss + value_coef * value_loss - entropy_coef * normalised_entropy
    (PPO eq. 9 with the entropy normalised by log(legal moves)). The a2c preset is
    epochs=1, minibatches=1, clip_epsilon=None, normalize_advantage=False.
    Returns per-update means of the losses plus clip fraction and approx KL
    (Huang et al. 2022, detail 12).
    """
    if batch is None or batch["obs"].shape[0] == 0:
        return dict(_EMPTY_UPDATE)
    n = batch["obs"].shape[0]
    sums = {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "clip_fraction": 0.0, "approx_kl": 0.0}
    steps = 0
    for _ in range(epochs):
        for idx in _minibatch_indices(n, minibatches, generator):
            probs, value = model(batch["obs"][idx], batch["legal_mask"][idx])
            dist = Categorical(probs=probs)
            new_log_prob = dist.log_prob(batch["action"][idx])
            adv = batch["adv"][idx]
            if normalize_advantage:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8) if len(adv) > 1 else torch.zeros_like(adv)
            policy_loss, ratio = ppo_policy_loss(new_log_prob, batch["old_log_prob"][idx], adv, clip_epsilon)
            value_loss = (value.squeeze(-1) - batch["target"][idx]).pow(2).mean()
            norm_ent = normalized_entropy(dist.entropy(), batch["num_legal"][idx])
            composed = ComposedLoss().compute(
                policy_loss=policy_loss, value_loss=value_loss, entropy_bonus=norm_ent.mean(),
                entropy_coef=entropy_coef, value_coef=value_coef,
            )
            optimizer.zero_grad()
            composed["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=grad_clip)
            optimizer.step()
            steps += 1
            with torch.no_grad():
                sums["total_loss"] += composed["total_loss"].item()
                sums["policy_loss"] += policy_loss.item()
                sums["value_loss"] += value_loss.item()
                # Diagnostics over clean samples only: on noisy boards the stored
                # log-prob is the behaviour mixture's, so ratio and KL there measure
                # the noise, not how far the policy moved.
                clean = ~batch["noisy"][idx] if "noisy" in batch else torch.ones(len(idx), dtype=torch.bool)
                if not clean.any():
                    clean = torch.ones(len(idx), dtype=torch.bool)
                if clip_epsilon is not None:
                    sums["clip_fraction"] += ((ratio[clean] - 1.0).abs() > clip_epsilon).float().mean().item()
                sums["approx_kl"] += (batch["old_log_prob"][idx][clean] - new_log_prob[clean]).mean().item()
    out = {k: v / steps for k, v in sums.items()}
    if clip_epsilon is None:
        out["clip_fraction"] = None
    out["optimizer_steps"] = steps
    out["sample_count"] = n
    return out


def refresh_norm_stats(model, obs, legal_mask, chunk=256):
    """Recompute BatchNorm running statistics from `obs` and return the model to
    evaluation mode.

    Training keeps the network in eval mode so the probability the update
    re-computes equals the one the rollout sampled from (ratio exactly 1 before
    the first step). Train-mode BatchNorm breaks that: measured 2026-09-16 on a
    fresh network, batch statistics of 12 boards vs 192 samples alone gave clip
    fraction 0.52 and approx KL 0.10. The statistics still track the data because
    they are refreshed here once per update (one update stale)."""
    model.train()
    with torch.no_grad():
        for i in range(0, obs.shape[0], chunk):
            model(obs[i:i + chunk], legal_mask[i:i + chunk])
    model.eval()


A2C_PRESET = dict(epochs=1, minibatches=1, clip_epsilon=None, normalize_advantage=False,
                  value_coef=1.0, gae_lambda=1.0)
PPO_PRESET = dict(epochs=4, minibatches=4, clip_epsilon=0.2, normalize_advantage=True,
                  value_coef=0.5, gae_lambda=0.95)


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
    algorithm=None,
    layout_schedule=None,
    explore=None,
    guide=None,
    pool=None,
    sil=None,
    punish=None,
    referee=None,
    progress_path=None,
):
    """Run A2C or PPO self-play.

    `layout_schedule`: list of {from_update, finish_boards, midgame_boards}; at
    `from_update` the env's board roles after the puzzle boards are switched
    (`envs.set_layout`), e.g. to hand finishing boards back to opening play.
    `explore`: None or dict(epsilon, alpha, boards) with boards `curriculum`
    (puzzle + finishing boards) or `all`; the board mask is rebuilt every update.
    `guide`: None or dict(epsilon, boards): label-guided behaviour, the demonstrator
    (puzzle key move / human line / rules mate) is played with probability epsilon.
    `punish`: None or dict(epsilon, threshold, boards): on its boards a rules mate or
    a capture winning >= threshold net material is the demonstration when the board
    has no label demonstration (change punish-gifts).
    `referee`: None or dict(threshold): pool opponents always play a punishing move when
    one exists (arm POOL_REF); requires `pool`.

    `pool`: None or a `PoolBoards` (src/training/opponent_pool.py): its boards play
    the learner against a frozen pool member; after each update the pool may take
    a snapshot of the learner (`opponent_snapshot_every`).

    `sil`: None or dict(updates, batch, loss_weight, value_weight, buffer): self-imitation
    (Oh et al. 2018, src/training/sil.py) after each update on the learner's own finished
    episodes; shared network only.

    `algorithm`: dict of update settings (`epochs`, `minibatches`, `clip_epsilon`,
    `normalize_advantage`, `value_coef`, `gae_lambda`); default `A2C_PRESET`.

    Two modes. Legacy: separate `white_model` / `black_model` with their optimizers.
    Shared (AlphaZero-style): pass `black_model=None`; `white_model` then plays both
    colours on observations oriented to the mover, and both sides' samples update it
    in one backward pass per update with `optimizer_white`.

    `eval_fn(white_model, black_model) -> dict` is called every `eval_interval`
    updates and after the last one; its keys are merged into the log entry for
    that update (a log entry is created for it if none is due)."""
    algo = {**A2C_PRESET, **(algorithm or {})}
    lam = algo.pop("gae_lambda")
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

    # BatchNorm in eval mode during collection and update; see refresh_norm_stats.
    white_model.eval()
    black_model.eval()

    num_envs = envs.num_envs
    num_puzzle_envs = getattr(envs, "num_puzzle_envs", 0)
    puzzle_env_mask = torch.arange(num_envs) < num_puzzle_envs
    logs = []
    losses = []
    env_step = envs.reset()
    pbar = tqdm(range(1, episodes + 1))

    sil_state = None
    if sil is not None and int(sil.get("updates", 0)) > 0:
        if not shared:
            raise ValueError("self-imitation needs the shared network")
        from src.training.sil import EpisodeAccumulator, ReplayBuffer
        buffer = ReplayBuffer(int(sil["buffer"]), seed=int(sil.get("seed", 0)))
        sil_state = {"buffer": buffer, "acc": EpisodeAccumulator(num_envs, gamma, buffer)}

    finish_sampler = getattr(envs, "finish_sampler", None)
    pending_layouts = sorted(layout_schedule or [], key=lambda d: d["from_update"])
    for episode in pbar:
        while pending_layouts and episode >= pending_layouts[0]["from_update"]:
            change = pending_layouts.pop(0)
            envs.set_layout(int(change["finish_boards"]), int(change["midgame_boards"]))
            print(f"update {episode}: layout -> finish {change['finish_boards']}, mid-game {change['midgame_boards']}")
        if finish_sampler is not None:
            metrics.set_finish_depth(finish_sampler.current_depth)
            if hasattr(finish_sampler, "dials"):
                metrics.set_technique_dials(finish_sampler.dials())
            elif hasattr(finish_sampler, "levels"):
                metrics.set_technique_levels(finish_sampler.levels())
        def _board_mask(which):
            if which == "all":
                return torch.ones(num_envs, dtype=torch.bool)
            if which == "puzzle":
                return torch.tensor([envs.is_puzzle_env(i) for i in range(num_envs)])
            is_finish = getattr(envs, "is_finish_env", lambda i: False)
            return torch.tensor([envs.is_puzzle_env(i) or is_finish(i) for i in range(num_envs)])
        guide_now = None
        if guide is not None and guide.get("epsilon", 0) > 0:
            guide_now = {"epsilon": guide["epsilon"], "mask": _board_mask(guide.get("boards", "curriculum"))}
        punish_now = None
        if punish is not None and punish.get("epsilon", 0) > 0:
            punish_now = {"epsilon": punish["epsilon"], "threshold": int(punish.get("threshold", 3)),
                          "mask": _board_mask(punish.get("boards", "all"))}
        explore_now = None
        if explore is not None and explore.get("epsilon", 0) > 0:
            which = explore.get("boards", "curriculum")
            if which == "all":
                emask = torch.ones(num_envs, dtype=torch.bool)
            elif which == "puzzle":
                emask = torch.tensor([envs.is_puzzle_env(i) for i in range(num_envs)])
            else:
                is_finish = getattr(envs, "is_finish_env", lambda i: False)
                emask = torch.tensor([envs.is_puzzle_env(i) or is_finish(i) for i in range(num_envs)])
            explore_now = {"epsilon": explore["epsilon"], "alpha": explore["alpha"], "mask": emask}
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
            explore=explore_now,
            guide=guide_now,
            pool=pool,
            punish=punish_now,
            referee=referee,
        )

        bootstrap_white = _compute_bootstrap(
            env_step, white_model, black_model, model_id=WHITE, oriented=shared
        )
        bootstrap_black = _compute_bootstrap(
            env_step, white_model, black_model, model_id=BLACK, oriented=shared
        )

        adv_w, tgt_w = compute_gae_for_model(
            steps_data, model_id=WHITE, bootstrap_value=bootstrap_white, gamma=gamma, lam=lam,
            final_material=env_step.material, shaping_scale=shaping_scale,
        )
        adv_b, tgt_b = compute_gae_for_model(
            steps_data, model_id=BLACK, bootstrap_value=bootstrap_black, gamma=gamma, lam=lam,
            final_material=env_step.material, shaping_scale=shaping_scale,
        )
        g_w = _gather_side(steps_data, adv_w, tgt_w, WHITE, puzzle_env_mask)
        g_b = _gather_side(steps_data, adv_b, tgt_b, BLACK, puzzle_env_mask)
        update_kwargs = dict(algo, entropy_coef=entropy_coef, grad_clip=grad_clip)

        if shared:
            # One batch of both colours' own steps, one optimizer.
            batch = _concat_batches([g_w, g_b])
            updates = [update_model(white_model, optimizer_white, batch, **update_kwargs)]
            if batch is not None:
                refresh_norm_stats(white_model, batch["obs"], batch["legal_mask"])
            if sil_state is not None:
                from src.training.sil import sil_update
                for step in steps_data:
                    sil_state["acc"].add_step(step)
                stats = sil_update(white_model, optimizer_white, sil_state["buffer"], updates=int(sil["updates"]),
                                   batch=int(sil["batch"]), loss_weight=float(sil["loss_weight"]),
                                   value_weight=float(sil["value_weight"]), grad_clip=grad_clip)
                metrics.add_sil_stats(len(sil_state["buffer"]), stats)
        else:
            updates = []
            for model, opt, g in ((white_model, optimizer_white, g_w), (black_model, optimizer_black, g_b)):
                updates.append(update_model(model, opt, g, **update_kwargs))
                if g is not None:
                    refresh_norm_stats(model, g["obs"], g["legal_mask"])
        result_w, result_b = _side_stats(g_w), _side_stats(g_b)

        total_loss = sum(u["total_loss"] for u in updates)
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
        metrics.add_update_stats(
            policy_loss=_sample_weighted(updates, "policy_loss"),
            value_loss=_sample_weighted(updates, "value_loss"),
            clip_fraction=_sample_weighted(updates, "clip_fraction"),
            approx_kl=_sample_weighted(updates, "approx_kl"),
        )

        pbar.set_postfix(loss=total_loss)
        if pool is not None:
            pool.pool.maybe_snapshot(white_model, episode)

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
        if progress_path is not None and (episode % log_interval == 0 or eval_metrics or episode == episodes):
            # Live view for scripts/dashboard.py: the logs so far plus where the run is.
            import json, time as _time
            Path(progress_path).write_text(json.dumps({"episode": episode, "episodes": episodes,
                                                       "updated": _time.time(), "logs": logs}))

    if hasattr(envs, "close"):
        envs.close()

    return logs, losses, white_model, black_model
