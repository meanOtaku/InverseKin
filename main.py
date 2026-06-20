from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / "venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
VENV_MJPYTHON = VENV_DIR / "bin" / "mjpython"

DEFAULT_MODEL = PROJECT_ROOT / "models" / "3r_planar_pen.xml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "trajectory_data.csv"
DEFAULT_POSE_OUTPUT = PROJECT_ROOT / "outputs" / "final_pose_map.csv"
DEFAULT_PLOT_OUTPUT = PROJECT_ROOT / "outputs" / "workspace_area.svg"
DEFAULT_QPOS_DEGREES = (35.0, -50.0, 45.0)
LINK_LENGTHS = (0.14, 0.11, 0.125)
BASE_XY = (-0.32, -0.32)
JOINT_CAP_DIAMETER = 0.078


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3R planar robot trajectory data with MuJoCo.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the MJCF model.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV output path.")
    parser.add_argument("--pose-output", type=Path, default=DEFAULT_POSE_OUTPUT, help="Final XY and joint-angle CSV output path.")
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT_OUTPUT, help="SVG workspace plot output path.")
    parser.add_argument("--headless", action="store_true", help="Generate CSV data without opening the MuJoCo viewer.")
    parser.add_argument("--speed", type=float, default=1.0, help="Viewer playback speed multiplier.")
    parser.add_argument("--max-radius", type=float, default=0.34, help="Starting quarter-circle radius in meters.")
    parser.add_argument("--min-radius", type=float, default=0.190, help="Final quarter-circle radius in meters.")
    parser.add_argument("--radius-step", type=float, default=0.001, help="Radius reduction per sweep in meters.")
    parser.add_argument("--arc-points", type=int, default=91, help="Number of points per quarter-circle arc.")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration of the full quarter-circle sweep in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.headless:
        ensure_project_venv()
    else:
        ensure_viewer_launcher()

    mujoco = load_mujoco()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    data.qpos[:] = [math.radians(angle) for angle in DEFAULT_QPOS_DEGREES]
    mujoco.mj_forward(model, data)

    if not args.headless:
        run_viewer(mujoco, model, data, args)
        return

    pen_tip_id = model.site("pen_tip").id
    paths = build_required_paths(args)
    rows = []
    qpos_seed = list(data.qpos[:])
    for path_name, targets in paths:
        path_rows, qpos_seed = record_path(
            mujoco=mujoco,
            model=model,
            data=data,
            pen_tip_id=pen_tip_id,
            path_name=path_name,
            targets=targets,
            initial_qpos=qpos_seed,
            duration=args.duration,
        )
        rows.extend(path_rows)

    write_csv(rows, args.output)
    write_pose_csv(rows, args.pose_output)
    write_workspace_plot(rows, args.plot_output)
    print(f"Generated {len(rows)} samples across {len(paths)} paths.")
    print(f"CSV written to: {args.output}")
    print(f"Final pose CSV written to: {args.pose_output}")
    print(f"Workspace plot written to: {args.plot_output}")
    print_summary(rows)


def ensure_project_venv() -> None:
    if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV_DIR.resolve():
        restart_with(VENV_PYTHON)


def ensure_viewer_launcher() -> None:
    if platform.system() == "Darwin":
        if "MJPYTHON_BIN" in os.environ:
            return
        if VENV_MJPYTHON.exists():
            restart_with(VENV_MJPYTHON)
        raise SystemExit("MuJoCo viewer launcher is missing. Run: scripts/setup_venv.sh")
    ensure_project_venv()


def restart_with(executable: Path) -> None:
    os.execv(str(executable), [str(executable), str(Path(__file__).resolve()), *sys.argv[1:]])


def load_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise SystemExit("MuJoCo is not installed. Run: scripts/setup_venv.sh") from exc
    return mujoco


def build_required_paths(args: argparse.Namespace) -> list[tuple[str, list[dict[str, float]]]]:
    return [("quarter_circle_workspace", quarter_circle_targets(args))]


def quarter_circle_targets(args: argparse.Namespace) -> list[dict[str, float]]:
    if args.arc_points < 2:
        raise SystemExit("--arc-points must be at least 2.")
    if args.radius_step <= 0:
        raise SystemExit("--radius-step must be greater than zero.")
    if args.max_radius < args.min_radius:
        raise SystemExit("--max-radius must be greater than or equal to --min-radius.")

    targets = []
    radius_index = 0
    radius = args.max_radius
    tolerance = args.radius_step * 0.5
    while radius >= args.min_radius - tolerance:
        arc_order = range(args.arc_points) if radius_index % 2 == 0 else range(args.arc_points - 1, -1, -1)
        for angle_index in arc_order:
            angle = (math.pi / 2.0) * angle_index / (args.arc_points - 1)
            target_x = BASE_XY[0] + radius * math.cos(angle)
            target_y = BASE_XY[1] + radius * math.sin(angle)
            targets.append(
                {
                    "target_x": target_x,
                    "target_y": target_y,
                    "radius_m": radius,
                    "angle_deg": math.degrees(angle),
                }
            )
        radius_index += 1
        radius = args.max_radius - radius_index * args.radius_step
    return targets


def record_path(
    mujoco,
    model,
    data,
    pen_tip_id: int,
    path_name: str,
    targets: list[dict[str, float]],
    initial_qpos: list[float],
    duration: float,
) -> tuple[list[dict[str, float | int | str]], list[float]]:
    if duration <= 0:
        raise SystemExit("--duration must be greater than zero.")

    qpos_samples = []
    qpos = initial_qpos
    for target in targets:
        qpos = solve_planar_ik(world_to_base_xy((target["target_x"], target["target_y"])), qpos)
        qpos_samples.append(qpos)

    unwrapped_qpos_samples = unwrap_qpos_samples(qpos_samples)
    dt = duration / (len(qpos_samples) - 1)
    qvel_samples = finite_difference(unwrapped_qpos_samples, dt)
    qacc_samples = finite_difference(qvel_samples, dt)

    rows = []
    previous_disableflags = model.opt.disableflags
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    try:
        for index, (target, qpos, qvel, qacc) in enumerate(zip(targets, qpos_samples, qvel_samples, qacc_samples)):
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            data.qacc[:] = qacc
            mujoco.mj_inverse(model, data)
            pen_x, pen_y, pen_z = data.site_xpos[pen_tip_id]
            torque_1, torque_2, torque_3 = data.qfrc_inverse[:3]
            rows.append(
                {
                    "path": path_name,
                    "sample": index,
                    "time_s": index * dt,
                    "radius_m": target["radius_m"],
                    "angle_deg": target["angle_deg"],
                    "target_x_m": target["target_x"],
                    "target_y_m": target["target_y"],
                    "x_m": float(pen_x),
                    "y_m": float(pen_y),
                    "z_m": float(pen_z),
                    "theta1_deg": math.degrees(qpos[0]),
                    "theta2_deg": math.degrees(qpos[1]),
                    "theta3_deg": math.degrees(qpos[2]),
                    "torque1_nm": float(torque_1),
                    "torque2_nm": float(torque_2),
                    "torque3_nm": float(torque_3),
                }
            )
    finally:
        model.opt.disableflags = previous_disableflags

    return rows, qpos_samples[-1]


def solve_planar_ik(target: tuple[float, float], initial_qpos: list[float]) -> list[float]:
    import numpy as np

    q3_target = preferred_joint3(target)
    target_xy = np.array(target, dtype=float)
    best_q = None
    best_error = float("inf")

    for secondary_gain in (0.12, 0.04, 0.0):
        q = np.array(initial_qpos, dtype=float)
        damping = 1e-5
        for _ in range(180):
            current_xy = np.array(forward_planar_xy(q.tolist()))
            error = target_xy - current_xy
            if np.linalg.norm(error) < 1e-7:
                break

            jacobian = planar_jacobian(q.tolist())
            solve_term = np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(2), error)
            primary = jacobian.T @ solve_term
            nullspace = np.eye(3) - jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(2), jacobian)
            secondary = np.array([0.0, 0.0, secondary_gain * wrap_to_pi(q3_target - q[2])])
            q += primary + nullspace @ secondary
            q[0] = wrap_to_pi(float(q[0]))
            q[1] = float(np.clip(q[1], math.radians(-160.0), math.radians(160.0)))
            q[2] = wrap_to_pi(float(q[2]))

        candidate = q.tolist()
        final_error = math.dist(forward_planar_xy(candidate), target)
        if final_error < best_error:
            best_q = candidate
            best_error = final_error
        if final_error <= 1e-4:
            return candidate

    if best_q is None or best_error > 5e-4:
        raise SystemExit(f"IK error is too high: x={target[0]:.3f}, y={target[1]:.3f}, error={best_error:.5f} m")
    return best_q


def preferred_joint3(target: tuple[float, float]) -> float:
    angle = math.atan2(target[1], target[0])
    radius = math.hypot(target[0], target[1])
    radius_phase = (radius - 0.15) / (0.34 - 0.15)
    return math.radians(35.0) * math.sin(2.0 * angle) + math.radians(15.0) * math.cos(math.pi * radius_phase)


def planar_jacobian(qpos: list[float]) -> "object":
    import numpy as np

    theta1 = qpos[0]
    theta2 = qpos[0] + qpos[1]
    theta3 = qpos[0] + qpos[1] + qpos[2]
    l1, l2, l3 = LINK_LENGTHS
    return np.array(
        [
            [
                -l1 * math.sin(theta1) - l2 * math.sin(theta2) - l3 * math.sin(theta3),
                -l2 * math.sin(theta2) - l3 * math.sin(theta3),
                -l3 * math.sin(theta3),
            ],
            [
                l1 * math.cos(theta1) + l2 * math.cos(theta2) + l3 * math.cos(theta3),
                l2 * math.cos(theta2) + l3 * math.cos(theta3),
                l3 * math.cos(theta3),
            ],
        ]
    )


def forward_planar_xy(qpos: list[float]) -> tuple[float, float]:
    theta1 = qpos[0]
    theta2 = qpos[0] + qpos[1]
    theta3 = qpos[0] + qpos[1] + qpos[2]
    x = LINK_LENGTHS[0] * math.cos(theta1) + LINK_LENGTHS[1] * math.cos(theta2) + LINK_LENGTHS[2] * math.cos(theta3)
    y = LINK_LENGTHS[0] * math.sin(theta1) + LINK_LENGTHS[1] * math.sin(theta2) + LINK_LENGTHS[2] * math.sin(theta3)
    return x, y


def world_to_base_xy(point: tuple[float, float]) -> tuple[float, float]:
    return point[0] - BASE_XY[0], point[1] - BASE_XY[1]


def finite_difference(samples: list[list[float]], dt: float) -> list[list[float]]:
    derivatives = []
    for index, sample in enumerate(samples):
        if index == 0:
            derivative = [(samples[1][joint] - sample[joint]) / dt for joint in range(len(sample))]
        elif index == len(samples) - 1:
            derivative = [(sample[joint] - samples[index - 1][joint]) / dt for joint in range(len(sample))]
        else:
            derivative = [(samples[index + 1][joint] - samples[index - 1][joint]) / (2.0 * dt) for joint in range(len(sample))]
        derivatives.append(derivative)
    return derivatives


def unwrap_qpos_samples(samples: list[list[float]]) -> list[list[float]]:
    if not samples:
        return []
    unwrapped = [samples[0][:]]
    for sample in samples[1:]:
        previous = unwrapped[-1]
        unwrapped.append(
            [
                previous[joint] + wrap_to_pi(sample[joint] - previous[joint])
                for joint in range(len(sample))
            ]
        )
    return unwrapped


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def joint_distance(qpos: list[float], reference_qpos: list[float]) -> float:
    return sum(abs(wrap_to_pi(q - ref)) for q, ref in zip(qpos, reference_qpos))


def write_csv(rows: list[dict[str, float | int | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "sample",
        "time_s",
        "radius_m",
        "angle_deg",
        "target_x_m",
        "target_y_m",
        "x_m",
        "y_m",
        "z_m",
        "theta1_deg",
        "theta2_deg",
        "theta3_deg",
        "torque1_nm",
        "torque2_nm",
        "torque3_nm",
    ]
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_pose_csv(rows: list[dict[str, float | int | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "sample",
        "radius_m",
        "angle_deg",
        "target_x_m",
        "target_y_m",
        "final_x_m",
        "final_y_m",
        "theta1_deg",
        "theta2_deg",
        "theta3_deg",
    ]
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row["path"],
                    "sample": row["sample"],
                    "radius_m": row["radius_m"],
                    "angle_deg": row["angle_deg"],
                    "target_x_m": row["target_x_m"],
                    "target_y_m": row["target_y_m"],
                    "final_x_m": row["x_m"],
                    "final_y_m": row["y_m"],
                    "theta1_deg": row["theta1_deg"],
                    "theta2_deg": row["theta2_deg"],
                    "theta3_deg": row["theta3_deg"],
                }
            )


def write_workspace_plot(rows: list[dict[str, float | int | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    xs = [float(row["x_m"]) for row in rows]
    ys = [float(row["y_m"]) for row in rows]
    radii = [float(row["radius_m"]) for row in rows]
    min_radius = min(radii)
    max_radius = max(radii)

    world_min_x = -0.47
    world_max_x = 0.08
    world_min_y = -0.47
    world_max_y = 0.08
    width = 900
    height = 900
    margin = 70

    def sx(x: float) -> float:
        return margin + (x - world_min_x) / (world_max_x - world_min_x) * (width - 2 * margin)

    def sy(y: float) -> float:
        return height - margin - (y - world_min_y) / (world_max_y - world_min_y) * (height - 2 * margin)

    outer_arc = arc_points(max_radius, 0.0, 90.0, 120)
    inner_arc = list(reversed(arc_points(min_radius, 0.0, 90.0, 120)))
    area_points = outer_arc + inner_arc
    area_text = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in area_points)
    outer_text = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in outer_arc)
    inner_text = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in reversed(inner_arc))
    sample_points = downsample(list(zip(xs, ys)), 1600)

    board_x = sx(world_min_x)
    board_y = sy(world_max_y)
    board_width = sx(world_max_x) - sx(world_min_x)
    board_height = sy(world_min_y) - sy(world_max_y)

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="900" viewBox="0 0 900 900">',
        '<rect width="900" height="900" fill="#f8fafc"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#1f2937}.label{font-size:22px;font-weight:700}.small{font-size:14px}</style>',
        f'<rect x="{board_x:.2f}" y="{board_y:.2f}" width="{board_width:.2f}" height="{board_height:.2f}" fill="#ffffff" stroke="#cbd5e1" stroke-width="2"/>',
        f'<polygon points="{area_text}" fill="#bfdbfe" fill-opacity="0.55" stroke="#2563eb" stroke-width="3"/>',
        f'<polyline points="{outer_text}" fill="none" stroke="#1d4ed8" stroke-width="4"/>',
        f'<polyline points="{inner_text}" fill="none" stroke="#60a5fa" stroke-width="3" stroke-dasharray="8 8"/>',
        f'<line x1="{sx(BASE_XY[0]):.2f}" y1="{sy(BASE_XY[1]):.2f}" x2="{sx(BASE_XY[0] + max_radius):.2f}" y2="{sy(BASE_XY[1]):.2f}" stroke="#94a3b8" stroke-width="2"/>',
        f'<line x1="{sx(BASE_XY[0]):.2f}" y1="{sy(BASE_XY[1]):.2f}" x2="{sx(BASE_XY[0]):.2f}" y2="{sy(BASE_XY[1] + max_radius):.2f}" stroke="#94a3b8" stroke-width="2"/>',
    ]
    for x, y in sample_points:
        svg.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="1.4" fill="#0f172a" fill-opacity="0.35"/>')

    svg.extend(
        [
            f'<circle cx="{sx(BASE_XY[0]):.2f}" cy="{sy(BASE_XY[1]):.2f}" r="13" fill="#ef4444" stroke="#7f1d1d" stroke-width="3"/>',
            f'<text x="{sx(BASE_XY[0]) + 18:.2f}" y="{sy(BASE_XY[1]) - 10:.2f}" class="small">robot base</text>',
            '<text x="70" y="46" class="label">Quarter-circle workspace area</text>',
            f'<text x="70" y="74" class="small">radius {min_radius:.3f} m to {max_radius:.3f} m, step 0.001 m, 0 to 90 deg</text>',
            f'<text x="70" y="96" class="small">samples: {len(rows):,}</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(svg), encoding="utf-8")


def arc_points(radius: float, start_deg: float, end_deg: float, count: int) -> list[tuple[float, float]]:
    points = []
    for index in range(count):
        fraction = index / (count - 1)
        angle = math.radians(start_deg + fraction * (end_deg - start_deg))
        points.append((BASE_XY[0] + radius * math.cos(angle), BASE_XY[1] + radius * math.sin(angle)))
    return points


def downsample(points: list[tuple[float, float]], max_count: int) -> list[tuple[float, float]]:
    if len(points) <= max_count:
        return points
    step = len(points) / max_count
    return [points[int(index * step)] for index in range(max_count)]


def print_summary(rows: list[dict[str, float | int | str]]) -> None:
    max_xy_error = max(
        math.hypot(float(row["x_m"]) - float(row["target_x_m"]), float(row["y_m"]) - float(row["target_y_m"]))
        for row in rows
    )
    max_z_error = max(abs(float(row["z_m"])) for row in rows)
    max_torque = max(
        max(abs(float(row["torque1_nm"])), abs(float(row["torque2_nm"])), abs(float(row["torque3_nm"])))
        for row in rows
    )
    print(f"Maximum XY tracking error: {max_xy_error:.6f} m")
    print(f"Maximum pen-tip height error from surface: {max_z_error:.6f} m")
    print(f"Maximum absolute joint torque: {max_torque:.6f} N*m")


def make_playback_frames(args: argparse.Namespace) -> list[dict[str, float | int | str | list[float]]]:
    frames = []
    qpos = [math.radians(angle) for angle in DEFAULT_QPOS_DEGREES]
    for path_name, targets in build_required_paths(args):
        for sample, target in enumerate(targets):
            world_target = (target["target_x"], target["target_y"])
            qpos = solve_planar_ik(world_to_base_xy(world_target), qpos)
            frames.append(
                {
                    "path": path_name,
                    "sample": sample,
                    "radius_m": target["radius_m"],
                    "angle_deg": target["angle_deg"],
                    "target_x_m": world_target[0],
                    "target_y_m": world_target[1],
                    "qpos": qpos,
                }
            )
    return frames


def run_viewer(mujoco, model, data, args: argparse.Namespace) -> None:
    import mujoco.viewer

    if args.speed <= 0:
        raise SystemExit("--speed must be greater than zero.")

    frames = make_playback_frames(args)
    if not frames:
        raise SystemExit("No viewer playback frames were generated.")
    pen_tip_id = model.site("pen_tip").id
    target_marker_id = model.site("target_marker").id
    frame_index = 0
    pose_rows = []
    last_frame_time = time.time()
    total_playback_seconds = args.duration
    frame_interval = total_playback_seconds / len(frames) / args.speed

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            if step_start - last_frame_time >= frame_interval:
                frame = frames[frame_index]
                target = (float(frame["target_x_m"]), float(frame["target_y_m"]))
                qpos = frame["qpos"]
                data.qpos[:] = qpos
                data.qvel[:] = 0
                data.qacc[:] = 0
                model.site_pos[target_marker_id][0] = target[0]
                model.site_pos[target_marker_id][1] = target[1]
                model.site_pos[target_marker_id][2] = 0.006
                mujoco.mj_forward(model, data)
                pen_x, pen_y, _ = data.site_xpos[pen_tip_id]
                pose_rows.append(
                    {
                        "path": frame["path"],
                        "sample": frame["sample"],
                        "radius_m": frame["radius_m"],
                        "angle_deg": frame["angle_deg"],
                        "target_x_m": frame["target_x_m"],
                        "target_y_m": frame["target_y_m"],
                        "x_m": float(pen_x),
                        "y_m": float(pen_y),
                        "theta1_deg": math.degrees(qpos[0]),
                        "theta2_deg": math.degrees(qpos[1]),
                        "theta3_deg": math.degrees(qpos[2]),
                    }
                )
                viewer.sync()
                frame_index += 1
                last_frame_time = step_start
                if frame_index >= len(frames):
                    write_pose_csv(pose_rows, args.pose_output)
                    print(f"Final pose CSV written to: {args.pose_output}")
                    print(f"Saved {len(pose_rows)} final poses. Viewer simulation complete.")
                    break
            else:
                viewer.sync()
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
