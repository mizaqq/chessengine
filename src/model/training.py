from torch.nn.functional import mse_loss
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.distributions import Categorical


def model_acc(model, envs):
    state = envs.get_current_states()
    mask = envs.get_current_actions()
    prob_dist, value = model(state, mask)
    dist = Categorical(probs=prob_dist)
    actions = dist.sample().unsqueeze(1)  # [B,1]
    log_prob = dist.log_prob(actions.squeeze(1)).unsqueeze(1)  # [B,1]
    entropy = dist.entropy().unsqueeze(1)
    envs.update_previous_states()
    envs.update_previous_actions()
    envs.move(actions)
    # 1) empty mask rows
    empty_mask_rows = mask.sum(dim=1) == 0  # [B] bool
    num_empty_masks = empty_mask_rows.sum().item()
    # 2) sampled illegal actions
    sampled_is_legal = mask.gather(1, actions).squeeze(1) > 0
    num_illegal_samples = (~sampled_is_legal).sum().item()
    return (
        value,
        log_prob,
        entropy,
        state,
        actions,
        num_empty_masks,
        num_illegal_samples,
    )


def backpropagate(model, R, optimizer, rewards, log_probs, values, entropies):
    actor_loss, critic_loss = 0, 0
    for i in reversed(range(len(rewards))):
        R = rewards[i].unsqueeze(1) + 0.99 * R
        advantage = R - values[i]
        actor_loss -= log_probs[i] * advantage.detach()
        critic_loss += mse_loss(values[i], R.detach())
    loss = actor_loss.mean() + critic_loss.mean() - 0.01 * torch.stack(entropies).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0)
    optimizer.step()
    return loss


def update_rewards(envs, rewards_white, rewards_black):
    for i, env in enumerate(envs.envs):
        if env.is_done():
            if env.env.get_time_step().rewards[1] == 1:
                if rewards_white is not None:
                    rewards_white[i] += 2
                if rewards_black is not None:
                    rewards_black[i] -= 2
            elif env.env.get_time_step().rewards[0] == 1:
                if rewards_white is not None:
                    rewards_white[i] -= 2
                if rewards_black is not None:
                    rewards_black[i] += 2
            else:
                if rewards_white is not None:
                    rewards_white[i] -= 0.5
                if rewards_black is not None:
                    rewards_black[i] -= 0.5
    return rewards_white, rewards_black


def check_win(envs, white_wins, black_wins, draws):
    if any(envs.get_done()):
        for i, env in enumerate(envs.envs):
            if env.is_done():
                if env.env.get_time_step().rewards[1] == 1:
                    white_wins += 1
                elif env.env.get_time_step().rewards[0] == 1:
                    black_wins += 1
                else:
                    draws += 1
    return white_wins, black_wins, draws


def run_chess_training(
    envs,
    white_model,
    black_model,
    optimizer_white,
    optimizer_black,
    episodes=2000,
    steps=5,
    lr_decay_interval=100,
):
    logs = []
    white_wins = 0
    black_wins = 0
    draws = 0
    pbar = tqdm(range(1, episodes))
    losses = []
    moves = 0
    episode_empty_masks = 0
    episode_illegal_samples = 0
    for episode in pbar:
        if any(envs.get_done()):
            white_wins, black_wins, draws = check_win(
                envs, white_wins, black_wins, draws
            )
            envs.reset_all()
            moves = 0
            continue
        (
            rewards_white_list,
            rewards_black_list,
            values_white_list,
            values_black_list,
            log_probs_white_list,
            log_probs_black_list,
            entropies_white_list,
            entropies_black_list,
        ) = [], [], [], [], [], [], [], []

        for _ in range(steps):
            moves += 1
            (
                values_white,
                log_probs_white,
                entropies_white,
                state_white,
                actions_white,
                num_empty_masks,
                num_illegal_samples,
            ) = model_acc(white_model, envs)
            last_next_state_white = envs.get_current_states().detach()
            last_done_white = envs.get_done().float().unsqueeze(1).detach()
            values_white_list.append(values_white)
            log_probs_white_list.append(log_probs_white)
            entropies_white_list.append(entropies_white)
            rewards_white = envs.get_rewards("white") / 50
            if any(envs.get_done()):
                moves = 0

                white_wins, black_wins, draws = check_win(
                    envs, white_wins, black_wins, draws
                )

                rewards_white, rewards_black = update_rewards(envs, rewards_white, None)
                rewards_white_list.append(rewards_white)
                break
            (
                values_black,
                log_probs_black,
                entropies_black,
                state_black,
                actions_black,
                num_empty_masks,
                num_illegal_samples,
            ) = model_acc(black_model, envs)
            last_next_state_black = envs.get_current_states().detach()
            last_done_black = envs.get_done().float().unsqueeze(1).detach()
            values_black_list.append(values_black)
            log_probs_black_list.append(log_probs_black)
            entropies_black_list.append(entropies_black)
            rewards_black = envs.get_rewards("black") / 50
            rewards_white, rewards_black = update_rewards(
                envs, rewards_white, rewards_black
            )

            rewards_white_list.append(rewards_white)
            rewards_black_list.append(rewards_black)
            if any(envs.get_done()):
                white_wins, black_wins, draws = check_win(
                    envs, white_wins, black_wins, draws
                )
                moves = 0
                break
        R_white = None
        R_black = None
        with torch.no_grad():
            if len(rewards_white_list) > 0:
                # next state after white move is black-to-move
                _, v_black_next = black_model(last_next_state_white)
                R_white = (-v_black_next) * (1.0 - last_done_white)

            if len(rewards_black_list) > 0:
                # next state after black move is white-to-move
                _, v_white_next = white_model(last_next_state_black)
                R_black = (-v_white_next) * (1.0 - last_done_black)
        if R_white is not None:
            loss_1 = backpropagate(
                white_model,
                R_white,
                optimizer_white,
                rewards_white_list,
                log_probs_white_list,
                values_white_list,
                entropies_white_list,
            )
        if R_black is not None:
            loss_2 = backpropagate(
                black_model,
                R_black,
                optimizer_black,
                rewards_black_list,
                log_probs_black_list,
                values_black_list,
                entropies_black_list,
            )
        total_loss = 0.0
        if R_white is not None:
            total_loss += loss_1.item()
        if R_black is not None:
            total_loss += loss_2.item()
        losses.append(total_loss)

        pbar.set_postfix(
            loss=total_loss,
            mem=envs.buffers[0].index,
            entropy=torch.stack(entropies_white_list).mean().item()
            if len(entropies_white_list) > 0
            else 0,
            entropy_black=torch.stack(entropies_black_list).mean().item()
            if len(entropies_black_list) > 0
            else 0,
            value_white=torch.stack(values_white_list).mean().item()
            if len(values_white_list) > 0
            else 0,
            value_black=torch.stack(values_black_list).mean().item()
            if len(values_black_list) > 0
            else 0,
        )
        if episode % lr_decay_interval == 0:
            optimizer_white.param_groups[0]["lr"] = max(
                3e-4, optimizer_white.param_groups[0]["lr"] * 0.5
            )
            optimizer_black.param_groups[0]["lr"] = max(
                3e-4, optimizer_black.param_groups[0]["lr"] * 0.5
            )
        episode_empty_masks += num_empty_masks
        episode_illegal_samples += num_illegal_samples
        if episode % 10 == 0:
            logs.append(
                {
                    "episode": episode,
                    "moves": moves,
                    "loss": total_loss,
                    "entropy": torch.stack(entropies_white_list).mean().item()
                    if len(entropies_white_list) > 0
                    else 0,
                    "entropy_black": torch.stack(entropies_black_list).mean().item()
                    if len(entropies_black_list) > 0
                    else 0,
                    "value_white": torch.stack(values_white_list).mean().item()
                    if len(values_white_list) > 0
                    else 0,
                    "value_black": torch.stack(values_black_list).mean().item()
                    if len(values_black_list) > 0
                    else 0,
                    "white_wins": white_wins,
                    "black_wins": black_wins,
                    "draws": draws,
                    "episode_empty_masks": episode_empty_masks,
                    "episode_illegal_samples": episode_illegal_samples,
                }
            )
    print(
        f"Training complete. Model saved to src/models/white_model.pth and src/models/black_model.pth"
    )
    return logs, losses, white_model, black_model
