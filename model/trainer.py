import torch
import torch.nn as nn

from utils import get_inputs, RewardPlotter


class PPOTrainer:
    def __init__(
        self,
        model,
        track,
        optimizer,
        device="cpu",
        checkpoint_path="checkpoints/last.pth",
        max_ray_dist=500.0,
        steer_smooth_alpha=0.8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.25,
        ent_coef=0.001,
        ppo_epochs=4,
        minibatch_size=2048,
        crash_penalty=20.0,
    ):
        self.model = model
        self.track = track
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.max_ray_dist = max_ray_dist
        self.steer_smooth_alpha = steer_smooth_alpha

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.crash_penalty = crash_penalty

        self.plotter = RewardPlotter()
        self.set_track(track)

    def set_track(self, track):
        self.track = track
        self.gates_tensor = track.gates.clone().to(self.device)
        self.gate_centers = (self.gates_tensor[:, 0:2] + self.gates_tensor[:, 2:4]) / 2.0
        self.track_name = track.track_name

    def _compute_step_reward(self, cars, gate_indices, last_gate_indices, dt=0.1):
        """2-term reward: normalized forward progress along centerline + crash penalty."""
        was_active = cars.active.clone()

        # Tangent vector from last gate center to next gate center
        p_prev = self.gate_centers[last_gate_indices]
        p_next = self.gate_centers[gate_indices]
        tangent = p_next - p_prev
        tangent = tangent / (tangent.norm(dim=1, keepdim=True) + 1e-6)

        # Forward displacement normalized by max possible step distance
        displacement = cars.pos - cars.prev_pos
        forward_dist = (displacement * tangent).sum(dim=1)
        progress_reward = (forward_dist / (cars.max_speed * dt)) * was_active.float()

        # Positive as soon as the car crosses past p_next along the track direction
        passed = (((cars.pos - p_next) * tangent).sum(dim=1) >= 0.0) & was_active
        if passed.any():
            cars.gates_passed[passed] += 1
            last_gate_indices[passed] = gate_indices[passed]
            gate_indices[passed] = (
                gate_indices[passed] + cars.direction[passed]
            ) % self.gates_tensor.shape[0]

        # Wall collision check
        cars.check_collisions(self.track)
        crashed = was_active & (~cars.active)
        crash_pen = -self.crash_penalty * crashed.float()

        return progress_reward + crash_pen, crashed

    @torch.no_grad()
    def collect_rollout(self, cars, n_steps, random_start=True, both_directions=True):
        force_dir = None if both_directions else 1
        start_indices = cars.reset(
            self.track, random_start=random_start, force_direction=force_dir
        )
        n_gates = self.gates_tensor.shape[0]
        last_gate_indices = start_indices.clone()
        gate_indices = (start_indices + cars.direction) % n_gates

        N = cars.n_cars
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf, mask_buf = [], [], [], [], [], [], []

        for _ in range(n_steps):
            if not cars.active.any():
                break

            active_before = cars.active.clone()
            obs, _ = get_inputs(cars, self.track, self.gates_tensor, gate_indices, self.max_ray_dist)
            mean, value = self.model(obs)
            dist = self.model.get_dist(mean)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            # Clamp only when passing to physics so log_prob remains unbiased
            steer = action[:, 0].clamp(-1.0, 1.0) * active_before.float()
            accel = action[:, 1].clamp(-1.0, 1.0) * active_before.float()

            cars.physics_update(steer, accel, dt=0.1, steer_smooth_alpha=self.steer_smooth_alpha)
            reward, crashed = self._compute_step_reward(cars, gate_indices, last_gate_indices, dt=0.1)

            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(log_prob)
            val_buf.append(value)
            rew_buf.append(reward)
            done_buf.append(crashed.float())
            mask_buf.append(active_before.float())

        # Bootstrap value for cars still alive at the end of the rollout
        last_obs, _ = get_inputs(cars, self.track, self.gates_tensor, gate_indices, self.max_ray_dist)
        _, last_val = self.model(last_obs)
        last_val = last_val * cars.active.float()

        obs_t = torch.stack(obs_buf)       # (T, N, D)
        act_t = torch.stack(act_buf)       # (T, N, 2)
        logp_t = torch.stack(logp_buf)     # (T, N)
        val_t = torch.stack(val_buf)       # (T, N)
        rew_t = torch.stack(rew_buf)       # (T, N)
        done_t = torch.stack(done_buf)     # (T, N)
        mask_t = torch.stack(mask_buf)     # (T, N)

        # Compute GAE advantages
        T = obs_t.shape[0]
        adv_t = torch.zeros_like(rew_t)
        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else val_t[t + 1]
            not_done = 1.0 - done_t[t]
            delta = rew_t[t] + self.gamma * next_val * not_done - val_t[t]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            adv_t[t] = gae * mask_t[t]

        ret_t = adv_t + val_t

        # Flatten only valid (active) transitions
        valid = mask_t.bool()
        batch = {
            "obs": obs_t[valid],
            "act": act_t[valid],
            "logp": logp_t[valid],
            "adv": adv_t[valid],
            "ret": ret_t[valid],
        }
        ep_reward = rew_t.sum(dim=0).mean().item()
        mean_gates = cars.gates_passed.float().mean().item()
        alive_pct = cars.active.float().mean().item() * 100.0
        return batch, ep_reward, mean_gates, alive_pct

    def ppo_update(self, batch):
        obs, act, old_logp, adv, ret = (
            batch["obs"], batch["act"], batch["logp"], batch["adv"], batch["ret"]
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n_samples = obs.shape[0]
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, self.minibatch_size):
                idx = perm[start : start + self.minibatch_size]
                mean, value = self.model(obs[idx])
                dist = self.model.get_dist(mean)
                new_logp = dist.log_prob(act[idx]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_logp - old_logp[idx])
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (ret[idx] - value).pow(2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()