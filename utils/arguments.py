import argparse


def args_nn():
    parser = argparse.ArgumentParser(description="Train racing neural network with PPO")

    # Tracks
    parser.add_argument("--track", type=str, default="simple", help="Single track name or path to geojson")
    parser.add_argument("--tracks", nargs="+", type=str, default=None, help="List of track names to sample randomly during training")

    # Run / device / checkpoints
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/mps/cuda)")
    parser.add_argument("--load", type=str, default="", help="Optional .pth checkpoint to load weights from")
    parser.add_argument("--save", type=str, default="checkpoints/base.pth", help="Path to save checkpoint")

    # Model / batch size
    parser.add_argument("--n_cars", type=int, default=128, help="Number of parallel cars per rollout")
    parser.add_argument("--n_rays", type=int, default=17, help="Number of rays per car")
    parser.add_argument("--input_dim", type=int, default=27, help="Input dimension (auto-computed as n_rays + 10)")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--output_dim", type=int, default=2, help="Output dimension size (steer, accel)")

    # Training schedule
    parser.add_argument("--n_epochs", type=int, default=300, help="Number of PPO epochs")
    parser.add_argument("--n_steps", type=int, default=400, help="Steps per rollout")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    # Sim / physics
    parser.add_argument("--ray_method", type=str, default="line", choices=["sphere", "line"], help="Ray casting method")
    parser.add_argument("--car_length", type=float, default=8.0, help="Car length in pixels")
    parser.add_argument("--car_width", type=float, default=4.0, help="Car width in pixels")
    parser.add_argument("--start_spacing", type=float, default=1.5, help="Longitudinal spacing multiplier")
    parser.add_argument("--max_ray_dist", type=float, default=500.0, help="Maximum ray distance for normalization")
    parser.add_argument("--steer_smooth_alpha", type=float, default=0.8, help="Steering smoothing factor (0 disables smoothing)")

    # Output
    parser.add_argument("--save_plot", type=str, default="reward_plot.png", help="Path to save final reward plot")

    return parser