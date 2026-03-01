import torch
from tqdm import tqdm
from torch.distributions import Categorical
from src.training.metrics import MetricsAggregator
from src.training.rollout_collector import RolloutCollector
from src.losses.policy_gradient import compute_policy_gradient_loss
from src.losses.value_mse import compute_value_loss
from src.losses.entropy_regularization import compute_entropy_bonus
from src.losses.composed_loss import ComposedLoss


def model_acc(model, envs):
    state = envs.get_current_states()
    mask = envs.get_current_actions()
    prob_dist, value = model(state, mask)
    dist = Categorical(probs=prob_dist)
    actions = dist.sample().unsqueeze(1)  # [B,1]
    log_prob = dist.log_prob(actions.squeeze(1)).unsqueeze(1)  # [B,1]
    entropy = dist.entropy().unsqueeze(1)
    
    # Compute normalized entropy (entropy / log(legal_moves))
    legal_moves_count = mask.sum(dim=1).float()
    log_legal_moves = torch.log(torch.clamp(legal_moves_count, min=2.0))
    normalized_entropy = entropy.squeeze(1) / log_legal_moves
    
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
        normalized_entropy.unsqueeze(1),
        state,
        actions,
        num_empty_masks,
        num_illegal_samples,
    )


def backpropagate(model, R, optimizer, rewards, log_probs, values, entropies):
    returns = []
    advantages = []
    for i in reversed(range(len(rewards))):
        R = rewards[i].unsqueeze(1) + 0.99 * R
        returns.insert(0, R)
        advantages.insert(0, R - values[i])
    
    policy_loss = compute_policy_gradient_loss(log_probs, advantages)
    value_loss = compute_value_loss(values, returns)
    entropy_bonus = compute_entropy_bonus(entropies)
    
    loss_composer = ComposedLoss()
    loss_dict = loss_composer.compute(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_bonus=entropy_bonus,
        entropy_coef=0.01
    )
    
    total_loss = loss_dict["total_loss"]
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0)
    optimizer.step()
    return total_loss


def check_win(envs, metrics: MetricsAggregator):
    """Check for terminal states and update metrics once per game."""
    if any(envs.get_done()):
        for i, env in enumerate(envs.envs):
            if env.is_done():
                if env.env.get_time_step().rewards[1] == 1:
                    metrics.add_terminal_result(white_win=True)
                elif env.env.get_time_step().rewards[0] == 1:
                    metrics.add_terminal_result(black_win=True)
                else:
                    metrics.add_terminal_result(draw=True)


def run_chess_training(
    envs,
    white_model,
    black_model,
    optimizer_white,
    optimizer_black,
    metrics: MetricsAggregator = None,  # Dependency Injection!
    episodes=2000,
    steps=5,
    lr_decay_interval=100,
):
    logs = []
    # Create metrics if not provided (for backward compatibility)
    if metrics is None:
        metrics = MetricsAggregator()
    pbar = tqdm(range(1, episodes))
    losses = []
    moves = 0
    
    for episode in pbar:
        if any(envs.get_done()):
            check_win(envs, metrics)
            envs.reset_all()
            moves = 0
            metrics.reset_episode_counters()
            continue
        rollout_white = RolloutCollector()
        rollout_black = RolloutCollector()
        normalized_entropies_white_list = []
        normalized_entropies_black_list = []

        for _ in range(steps):
            moves += 1
            (
                values_white,
                log_probs_white,
                entropies_white,
                normalized_entropy_white,
                state_white,
                actions_white,
                num_empty_masks,
                num_illegal_samples,
            ) = model_acc(white_model, envs)
            
            metrics.add_step(empty_masks=num_empty_masks, illegal_samples=num_illegal_samples)
            last_next_state_white = envs.get_current_states().detach()
            last_done_white = envs.get_done().float().unsqueeze(1).detach()
            
            rewards_white = envs.get_rewards("white") / 50
            
            # Add terminal rewards if game ended
            if any(envs.get_done()):
                moves = 0
                check_win(envs, metrics)
                for i, env in enumerate(envs.envs):
                    if env.is_done():
                        if env.env.get_time_step().rewards[1] == 1:
                            rewards_white[i] += 2
                        elif env.env.get_time_step().rewards[0] == 1:
                            rewards_white[i] -= 2
                        else:
                            rewards_white[i] -= 0.5
            
            rollout_white.add_step(
                reward=rewards_white,
                value=values_white,
                log_prob=log_probs_white,
                entropy=entropies_white,
                state=state_white,
                action=actions_white,
            )
            normalized_entropies_white_list.append(normalized_entropy_white)
            
            if any(envs.get_done()):
                break
            (
                values_black,
                log_probs_black,
                entropies_black,
                normalized_entropy_black,
                state_black,
                actions_black,
                num_empty_masks,
                num_illegal_samples,
            ) = model_acc(black_model, envs)
            
            metrics.add_step(empty_masks=num_empty_masks, illegal_samples=num_illegal_samples)
            
            last_next_state_black = envs.get_current_states().detach()
            last_done_black = envs.get_done().float().unsqueeze(1).detach()
            
            rewards_white = envs.get_rewards("white") / 50
            rewards_black = envs.get_rewards("black") / 50
            
            # Add terminal rewards if game ended
            if any(envs.get_done()):
                check_win(envs, metrics)
                for i, env in enumerate(envs.envs):
                    if env.is_done():
                        if env.env.get_time_step().rewards[1] == 1:
                            rewards_white[i] += 2
                            rewards_black[i] -= 2
                        elif env.env.get_time_step().rewards[0] == 1:
                            rewards_white[i] -= 2
                            rewards_black[i] += 2
                        else:
                            rewards_white[i] -= 0.5
                            rewards_black[i] -= 0.5
                moves = 0
            
            rollout_white.add_step(
                reward=rewards_white,
                value=values_white,
                log_prob=log_probs_white,
                entropy=entropies_white,
            )
            rollout_black.add_step(
                reward=rewards_black,
                value=values_black,
                log_prob=log_probs_black,
                entropy=entropies_black,
            )
            normalized_entropies_black_list.append(normalized_entropy_black)
            
            if any(envs.get_done()):
                break
        R_white = None
        R_black = None
        with torch.no_grad():
            if len(rollout_white) > 0:
                # next state after white move is black-to-move
                _, v_black_next = black_model(last_next_state_white)
                R_white = (-v_black_next) * (1.0 - last_done_white)

            if len(rollout_black) > 0:
                # next state after black move is white-to-move
                _, v_white_next = white_model(last_next_state_black)
                R_black = (-v_white_next) * (1.0 - last_done_black)
        
        if R_white is not None:
            rewards_white_list, values_white_list, log_probs_white_list, entropies_white_list = rollout_white.get_data()
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
            rewards_black_list, values_black_list, log_probs_black_list, entropies_black_list = rollout_black.get_data()
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

        # Compute normalized entropy for logging
        norm_entropy_white = (
            torch.stack(normalized_entropies_white_list).mean().item()
            if len(normalized_entropies_white_list) > 0
            else 0
        )
        norm_entropy_black = (
            torch.stack(normalized_entropies_black_list).mean().item()
            if len(normalized_entropies_black_list) > 0
            else 0
        )
        
        # Get values for logging
        value_white_mean = 0
        value_black_mean = 0
        if len(rollout_white) > 0:
            _, values_w, _, _ = rollout_white.get_data()
            value_white_mean = torch.stack(values_w).mean().item()
        if len(rollout_black) > 0:
            _, values_b, _, _ = rollout_black.get_data()
            value_black_mean = torch.stack(values_b).mean().item()
        
        pbar.set_postfix(
            loss=total_loss,
            mem=envs.buffers[0].index,
            norm_entropy_w=norm_entropy_white,
            norm_entropy_b=norm_entropy_black,
            value_white=value_white_mean,
            value_black=value_black_mean,
        )
        if episode % lr_decay_interval == 0:
            optimizer_white.param_groups[0]["lr"] = max(
                3e-4, optimizer_white.param_groups[0]["lr"] * 0.5
            )
            optimizer_black.param_groups[0]["lr"] = max(
                3e-4, optimizer_black.param_groups[0]["lr"] * 0.5
            )
        
        if episode % 10 == 0:
            summary = metrics.episode_summary()
            logs.append(
                {
                    "episode": episode,
                    "moves": moves,
                    "loss": total_loss,
                    "normalized_entropy_white": norm_entropy_white,
                    "normalized_entropy_black": norm_entropy_black,
                    "value_white": value_white_mean,
                    "value_black": value_black_mean,
                    **summary,
                }
            )
    print("Training complete. Model saved to src/models/white_model.pth and src/models/black_model.pth")
    return logs, losses, white_model, black_model
