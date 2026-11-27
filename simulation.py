import tkinter as tk
import torch
from model import CarState, CarNet 
from utils import get_inputs, load_checkpoint_tolerant
from track import Track
from utils import args_nn

# -----------------------------
# Simulation parameters
# -----------------------------
PHYSICS_DT = 0.1  # seconds per physics update
CAR_WIDTH = 4
CAR_LENGTH = 8
MAX_RENDER_FPS = 60  # optional cap on rendering speed


# -----------------------------
# Tkinter visualization
# -----------------------------
class Simulation:

    def __init__(self, track, model, n_cars=1, n_rays=13, start_idx=0, device="cpu", max_ray_dist=500, steer_smooth_alpha=0.8, car_length=8.0, car_width=4.0, start_spacing=1.5):
        self.track = track
        self.model = model.to(device)
        self.n_cars = n_cars
        self.n_rays = n_rays
        self.device = device
        self.max_ray_dist = max_ray_dist
        self.steer_smooth_alpha = steer_smooth_alpha
        self.start_idx = start_idx
        self.timestep = 0
        self.epoch = 0

        self.disp_steer = 0.0
        self.disp_accel = 0.0
        self.disp_speed = 0.0
        self.hud_frame = 0
        self.hud_text = ("Steer: +0.00", "Accel: +0.00", "Speed: +0.0 px/s")

        # Initialize LSTM hidden state if using LSTM model
        self.hidden = None
        if hasattr(self.model, 'init_hidden'):
            self.hidden = self.model.init_hidden(n_cars, device)

        self.cars = CarState(n_cars, n_rays=n_rays, device=device, car_length=car_length, car_width=car_width, start_spacing=start_spacing)
        self.gates_tensor = self.track.gates.to(self.device)
        self.gate_centers = (self.gates_tensor[:, 0:2] + self.gates_tensor[:, 2:4]) / 2.0
        self.gate_indices = torch.full((n_cars,), start_idx, dtype=torch.long, device=device)
        self._reset_cars()

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title("Simulation")
        self.canvas = tk.Canvas(self.root, width=track.w, height=track.h, bg="white")
        self.canvas.pack()

        # Draw track once
        self.track_lines = []
        for line in track.all_lines:
            x1, y1, x2, y2 = line.tolist()
            self.track_lines.append(
                self.canvas.create_line(x1, y1, x2, y2, fill="black", width=2)
            )

        # Draw gates and store references
        self.gate_lines = []
        for gate in track.gates:
            x1, y1, x2, y2 = gate.tolist()
            self.gate_lines.append(
                self.canvas.create_line(x1, y1, x2, y2, fill="green", dash=(4, 2), width=1)
            )

        # Car rectangles
        self.car_rects = []
        for i in range(n_cars):
            rect = self.canvas.create_polygon(0, 0, 0, 0, 0, 0, 0, 0, fill="blue")
            self.car_rects.append(rect)

        self.accum_time = 0.0
        self.last_time = None
        t = 0
        self.root.bind("r", self.on_restart)
        self.root.after(0, self.update, t)
        self.root.mainloop()

    def _reset_cars(self):
        start_indices = self.cars.reset(
            self.track, start_idx=self.start_idx, random_start=False, force_direction=1
        )
        self.gate_indices = (start_indices + self.cars.direction) % self.gates_tensor.shape[0]

    def on_restart(self, _event=None):
        self.timestep = 0
        self.epoch += 1
        self._reset_cars()

    # Helper: rectangle corners based on position and angle
    def get_car_corners(self, pos, angle):
        cx, cy = pos
        w, l = CAR_WIDTH / 2, CAR_LENGTH / 2

        # rectangle points relative to center
        corners = torch.tensor(
            [[-l, -w], [-l, w], [l, w], [l, -w]], dtype=torch.float32
        )

        # rotation matrix
        c, s = torch.cos(angle), torch.sin(angle)
        rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
        rotated = corners @ rot.T
        rotated += pos
        return rotated.flatten().tolist()

    def update(self, t):
        self.cars.prev_pos = self.cars.pos.clone()

        # --- Compute NN inputs ---
        next_gate_centers = (
            self.gates_tensor[self.gate_indices, 0:2]
            + self.gates_tensor[self.gate_indices, 2:4]
        ) / 2
        inputs, ray_dists = get_inputs(
            self.cars, self.track, self.gates_tensor, self.gate_indices, self.max_ray_dist
        )

        with torch.no_grad():
            outputs, _ = self.model(inputs)

            active_f = self.cars.active.float()
            steer = torch.clamp(outputs[:, 0], -1.0, 1.0) * active_f
            accel = torch.clamp(outputs[:, 1], -1.0, 1.0) * active_f

            self.cars.physics_update(steer, accel, dt=PHYSICS_DT, steer_smooth_alpha=self.steer_smooth_alpha)
            self.cars.check_collisions(self.track)

            # Advance gate indices when car crosses the plane of the next gate
            n_gates = self.gates_tensor.shape[0]
            prev_indices = (self.gate_indices - self.cars.direction) % n_gates
            p_prev = self.gate_centers[prev_indices]
            p_next = self.gate_centers[self.gate_indices]
            tangent = p_next - p_prev

            passed = (((self.cars.pos - p_next) * tangent).sum(dim=1) >= 0.0) & self.cars.active
            self.gate_indices[passed] = (
                self.gate_indices[passed] + self.cars.direction[passed]
            ) % n_gates

        # Recompute ray end points so drawn rays match current car positions.
        # `ray_dists` returned by get_inputs is the original distances in pixels.
        ray_angles = self.cars.ray_angles  # [n_cars, n_rays]
        ray_lengths = ray_dists  # already in pixels from get_inputs
        dx = torch.cos(ray_angles) * ray_lengths  # [n_cars, n_rays]
        dy = torch.sin(ray_angles) * ray_lengths
        ray_ends = self.cars.pos.unsqueeze(1) + torch.stack([dx, dy], dim=2)

        # --- Draw cars ---
        for i in range(self.n_cars):
            # Skip drawing inactive cars to avoid visual clutter; they also no longer
            # influence active cars due to input masking in get_inputs.
            if hasattr(self.cars, 'active') and not bool(self.cars.active[i].item()):
                self.canvas.itemconfig(self.car_rects[i], state='hidden')

            pos = self.cars.pos[i].cpu()
            angle = self.cars.angle[i].cpu()
            corners = self.get_car_corners(pos, angle)
            self.canvas.coords(self.car_rects[i], *corners)
            color = "red" if self.cars.active[i] else "gray"
            self.canvas.itemconfig(self.car_rects[i], fill=color, state='normal')
        # Draw rays (delete previous ray lines then draw new ones)
        #elf.canvas.delete("ray")
        #f ray_ends is not None:
        #   for i in range(self.n_cars):
        #       if hasattr(self.cars, 'active') and not bool(self.cars.active[i].item()):
        #           continue
        #       start = self.cars.pos[i].cpu().numpy()
        #       for j in range(ray_ends.shape[1]):
        #           if j == self.n_rays // 2:
        #               end = ray_ends[i, j].cpu().numpy()
        #               x0, y0 = float(start[0]), float(start[1])
        #               x1, y1 = float(end[0]), float(end[1])
        #               self.canvas.create_line(x0, y0, x1, y1, fill="orange", width=1, tags="ray")

        self.canvas.delete("text")
        self.canvas.create_text(
            50, 50, text="Timestep:" + str(self.timestep), fill="black", tags="text"
        )

        # Update gate colors: current gate is red, others are green
        for gate_idx, line_obj in enumerate(self.gate_lines):
            color = "red" if gate_idx == self.gate_indices[0].item() else "green"
            self.canvas.itemconfig(line_obj, fill=color)

        if self.n_cars == 1:
            alpha = 0.2  # lower = smoother, higher = more responsive
            self.disp_steer += alpha * (steer[0].item() - self.disp_steer)
            self.disp_accel += alpha * (accel[0].item() - self.disp_accel)
            self.disp_speed += alpha * (self.cars.speed[0].item() - self.disp_speed)

            self.hud_frame += 1
            if self.hud_frame % 6 == 0:  # refresh text ~5-10 times/sec
                self.hud_text = (
                    f"Steer: {self.disp_steer:+.2f}",
                    f"Accel: {self.disp_accel:+.2f}",
                    f"Speed: {self.disp_speed:5.1f} px/s",
                )

            for i, line in enumerate(self.hud_text):
                self.canvas.create_text(
                    10, 80 + i * 20, anchor="w",
                    text=line, fill="blue", font=("Courier", 12), tags="text"
                )

        self.timestep += 1
        # schedule next update
        self.root.after(int(1000 / MAX_RENDER_FPS), self.update, self.timestep)


# -----------------------------
# Model to simulate (change this variable to switch models)

def main():

    MODEL_NAME = "carnet"
    # Manual override: set to an int to force number of cars here, or None to use CLI/config.
    # Example: set `manual_n_cars = 1` to run a single-car legacy simulation.
    manual_n_cars = 1

    # Load config.json if it exists
    import json, os
    config = {}
    if os.path.exists("config.json"):
        with open("config.json") as f:
            config = json.load(f)
        print(f"Loaded config from config.json: {config}")

    parser = args_nn()
    # Set defaults from config
    if config:
        parser.set_defaults(**config)

    args = parser.parse_args()

    # If not manually overridden, prefer `args.group_size` to simulate a single training group
    # (useful when training uses very large batches but we only want one group in the GUI).
    if manual_n_cars is not None:
        n_cars = manual_n_cars
    elif hasattr(args, 'group_size') and args.group_size and args.group_size > 0:
        n_cars = args.group_size
    else:
        n_cars = args.n_cars if hasattr(args, 'n_cars') else 4

    device = torch.device("cpu")

    # Handle multi-track config - default to simple track for simulation
    track_name = args.track

    track = Track(track_name, 1000, 600, device=device, ray_method=args.ray_method)

    if getattr(args, "load", ""):
        checkpoint_path = args.load
    elif getattr(args, "save", ""):
        checkpoint_path = args.save
    else:
        checkpoint_path = "checkpoints/real_tracks_best.pth"

    print(f"Using model: {MODEL_NAME}")
    print(f"Loading checkpoint from: {checkpoint_path}")

    # rays + speed(1) + curvatures(6) + distances(3)
    computed_input_dim = args.n_rays + 10
    print(f"Computed input_dim={computed_input_dim} (n_rays={args.n_rays})")

    # carnet, separate_heads, deep, attention
    model = CarNet(
        input_dim=computed_input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, n_rays=args.n_rays
    ).to(device)

    # Load checkpoint tolerantly so older checkpoints with smaller input dims still partially load.
    ckpt = load_checkpoint_tolerant(checkpoint_path, model, device=device, verbose=True)
    model.eval()

    # If single car, use legacy random starts; for multiple cars use F1-style grouped starts.
    if n_cars == 1:
        sim_start_mode = None
        sim_group_size = None
    else:
        sim_start_mode = 'f1'
        # simulate a single group of `n_cars` vehicles (one training group only)
        sim_group_size = n_cars

    # set drawing sizes from args
    global CAR_LENGTH, CAR_WIDTH
    CAR_LENGTH = args.car_length
    CAR_WIDTH = args.car_width

    sim = Simulation(
        track,
        model,
        n_cars=n_cars,
        n_rays=args.n_rays,
        start_idx=track.gates.shape[0] - 1,
        device=device,
        max_ray_dist=args.max_ray_dist,
        steer_smooth_alpha=args.steer_smooth_alpha,
        car_length=args.car_length,
        car_width=args.car_width,
        start_spacing=args.start_spacing,
    )

if __name__ == "__main__":
    main()