import os
import json
import torch
import torch.optim as optim
from tqdm import trange

from track import Track
from model import CarNet, CarState
from model.trainer import PPOTrainer
from utils import args_nn, save_checkpoint, load_checkpoint_tolerant


def main():
    parser = args_nn()
    args = parser.parse_args()

    # Auto-compute single-car input_dim: rays + speed(1) + curvatures(6) + distances(3)
    args.input_dim = args.n_rays + 10

    with open("config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.autograd.set_detect_anomaly(False)

    screen_width, screen_height = 1000, 600
    n_steps = args.n_steps
    n_cars = args.n_cars
    n_rays = args.n_rays
    lr = args.lr

    device = (
        torch.device(args.device)
        if args.device
        else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    )

    tracks_list = args.tracks if args.tracks else [args.track]
    ckpt_folder = tracks_list[0] if len(tracks_list) == 1 else "multi"

    print(f"Tracks: {tracks_list} | Device: {device} | input_dim: {args.input_dim}")

    # Model and optimizer
    model = CarNet(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        n_rays=n_rays,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=lr * 0.1
    )

    # Load checkpoint if specified
    start_epoch = 0
    best_reward = -float("inf")
    checkpoint_path = args.save
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    if args.load and os.path.exists(args.load):
        print(f"Loading model weights from {args.load}...")
        load_checkpoint_tolerant(args.load, model, optimizer=None, device=device, verbose=True)
    else:
        print("Starting fresh.")

    # Preload all tracks into a pool
    track_pool = [
        Track(name, screen_width, screen_height, device=device, ray_method=args.ray_method)
        for name in tracks_list
    ]

    trainer = PPOTrainer(
        model=model,
        track=track_pool[0],
        optimizer=optimizer,
        device=device,
        checkpoint_path=checkpoint_path,
        max_ray_dist=args.max_ray_dist,
        steer_smooth_alpha=args.steer_smooth_alpha,
    )

    cars = CarState(
        n_cars,
        n_rays,
        device=device,
        car_length=args.car_length,
        car_width=args.car_width,
        start_spacing=args.start_spacing,
    )

    epoch_bar = trange(
        start_epoch, args.n_epochs, initial=start_epoch, total=args.n_epochs, desc="PPO Training"
    )

    try:
        for epoch in epoch_bar:
            all_batches = []
            track_stats = []

            for track in track_pool:
                trainer.set_track(track)
                b, ep_r, mean_g, alive_pct = trainer.collect_rollout(
                    cars, n_steps=n_steps, random_start=True, both_directions=True
                )
                all_batches.append(b)
                track_stats.append((track.track_name, ep_r, mean_g, alive_pct))

            # Merge all tracks into one balanced PPO batch
            combined_batch = {
                k: torch.cat([b[k] for b in all_batches], dim=0)
                for k in all_batches[0].keys()
            }
            trainer.ppo_update(combined_batch)
            scheduler.step()

            # Track mean reward across all tracks
            mean_reward = sum(s[1] for s in track_stats) / len(track_stats)
            trainer.plotter.add_reward(mean_reward)

            if mean_reward > best_reward:
                best_reward = mean_reward
                best_path = checkpoint_path.replace(".pth", "_best.pth")
                save_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    best_path,
                    total_epochs=args.n_epochs,
                    best_reward=best_reward,
                    track_name=ckpt_folder,
                )

            if epoch % 5 == 0 or epoch == args.n_epochs - 1:
                std = torch.exp(model.log_std).detach().cpu().numpy().round(3)
                per_track = " | ".join(
                    f"{name}: {r:5.1f} ({g:4.1f}g, {a:4.0f}%)"
                    for name, r, g, a in track_stats
                )
                epoch_bar.write(
                    f"Epoch {epoch:4d} | Mean: {mean_reward:6.2f} | Best: {best_reward:6.2f} | "
                    f"std: {std} || {per_track}"
                )
                save_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    checkpoint_path,
                    total_epochs=args.n_epochs,
                    best_reward=best_reward,
                    track_name=ckpt_folder,
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        save_checkpoint(
            epoch,
            model,
            optimizer,
            checkpoint_path,
            total_epochs=args.n_epochs,
            best_reward=best_reward,
            track_name=ckpt_folder,
        )
    finally:
        save_path = getattr(args, "save_plot", None)
        if save_path:
            try:
                trainer.plotter.save(save_path)
                print(f"Saved reward plot to {save_path}")
            except Exception as e:
                print(f"Warning: failed to save reward plot: {e}")
        try:
            trainer.plotter.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()