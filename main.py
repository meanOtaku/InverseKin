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
DEFAULT_QPOS_DEGREES = (35.0, -50.0, 45.0)
LINK_LENGTHS = (0.14, 0.11, 0.125)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3R planar robot trajectory data with MuJoCo.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the MJCF model.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV output path.")
    parser.add_argument("--viewer", action="store_true", help="Open the interactive MuJoCo viewer.")
    parser.add_argument("--tool-angle", type=float, default=-30.0, help="Link 3 drawing angle in degrees.")
    parser.add_argument("--samples-per-line", type=int, default=101, help="Samples for horizontal and vertical lines.")
    parser.add_argument("--zigzag-rows", type=int, default=7, help="Number of horizontal strokes in the zig-zag path.")
    parser.add_argument("--zigzag-samples-per-row", type=int, default=41, help="Samples per zig-zag row.")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration per named trajectory in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.viewer:
        ensure_viewer_launcher()
    else:
        ensure_project_venv()

    mujoco = load_mujoco()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    data.qpos[:] = [math.radians(angle) for angle in DEFAULT_QPOS_DEGREES]
    mujoco.mj_forward(model, data)

    if args.viewer:
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
            tool_angle=math.radians(args.tool_angle),
            duration=args.duration,
        )
        rows.extend(path_rows)

    write_csv(rows, args.output)
    print(f"Generated {len(rows)} samples across {len(paths)} paths.")
    print(f"CSV written to: {args.output}")
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


def build_required_paths(args: argparse.Namespace) -> list[tuple[str, list[tuple[float, float]]]]:
    horizontal = line_targets((0.18, 0.0), (0.32, 0.0), args.samples_per_line)
    vertical = line_targets((0.26, -0.06), (0.26, 0.06), args.samples_per_line)
    zigzag = zigzag_targets(
        min_x=0.19,
        max_x=0.31,
        min_y=-0.06,
        max_y=0.06,
        rows=args.zigzag_rows,
        samples_per_row=args.zigzag_samples_per_row,
    )
    return [
        ("horizontal", horizontal),
        ("vertical", vertical),
        ("zigzag", zigzag),
    ]


def line_targets(start: tuple[float, float], end: tuple[float, float], count: int) -> list[tuple[float, float]]:
    if count < 2:
        raise SystemExit("--samples-per-line must be at least 2.")
    return [
        (
            start[0] + index / (count - 1) * (end[0] - start[0]),
            start[1] + index / (count - 1) * (end[1] - start[1]),
        )
        for index in range(count)
    ]


def zigzag_targets(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    rows: int,
    samples_per_row: int,
) -> list[tuple[float, float]]:
    if rows < 2:
        raise SystemExit("--zigzag-rows must be at least 2.")
    if samples_per_row < 2:
        raise SystemExit("--zigzag-samples-per-row must be at least 2.")

    targets = []
    for row in range(rows):
        y = min_y + row / (rows - 1) * (max_y - min_y)
        start_x, end_x = (min_x, max_x) if row % 2 == 0 else (max_x, min_x)
        row_targets = line_targets((start_x, y), (end_x, y), samples_per_row)
        if targets:
            targets.extend(row_targets[1:])
        else:
            targets.extend(row_targets)
    return targets


def record_path(
    mujoco,
    model,
    data,
    pen_tip_id: int,
    path_name: str,
    targets: list[tuple[float, float]],
    initial_qpos: list[float],
    tool_angle: float,
    duration: float,
) -> tuple[list[dict[str, float | int | str]], list[float]]:
    if duration <= 0:
        raise SystemExit("--duration must be greater than zero.")

    qpos_samples = []
    qpos = initial_qpos
    for target in targets:
        qpos = solve_planar_ik(target, qpos, tool_angle)
        qpos_samples.append(qpos)

    dt = duration / (len(qpos_samples) - 1)
    qvel_samples = finite_difference(qpos_samples, dt)
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
                    "target_x_m": target[0],
                    "target_y_m": target[1],
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


def solve_planar_ik(target: tuple[float, float], initial_qpos: list[float], tool_angle: float) -> list[float]:
    l1, l2, l3 = LINK_LENGTHS
    wrist_x = target[0] - l3 * math.cos(tool_angle)
    wrist_y = target[1] - l3 * math.sin(tool_angle)
    wrist_distance_sq = wrist_x * wrist_x + wrist_y * wrist_y
    cos_q2 = (wrist_distance_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)

    if cos_q2 < -1.0 or cos_q2 > 1.0:
        raise SystemExit(
            f"Target is outside the workspace for tool angle {math.degrees(tool_angle):.1f} deg: "
            f"x={target[0]:.3f}, y={target[1]:.3f}"
        )

    cos_q2 = max(-1.0, min(1.0, cos_q2))
    candidates = []
    for q2 in (math.acos(cos_q2), -math.acos(cos_q2)):
        if not math.radians(-160.0) <= q2 <= math.radians(160.0):
            continue
        q1 = math.atan2(wrist_y, wrist_x) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
        q3 = tool_angle - q1 - q2
        candidate = [wrap_to_pi(q1), q2, wrap_to_pi(q3)]
        candidates.append(candidate)

    if not candidates:
        raise SystemExit(f"Target requires joint2 outside its limit: x={target[0]:.3f}, y={target[1]:.3f}")

    q = min(candidates, key=lambda candidate: joint_distance(candidate, initial_qpos))
    final_error = math.dist(forward_planar_xy(q), target)
    if final_error > 1e-5:
        raise SystemExit(f"IK error is too high: x={target[0]:.3f}, y={target[1]:.3f}, error={final_error:.5f} m")
    return q


def forward_planar_xy(qpos: list[float]) -> tuple[float, float]:
    theta1 = qpos[0]
    theta2 = qpos[0] + qpos[1]
    theta3 = qpos[0] + qpos[1] + qpos[2]
    x = LINK_LENGTHS[0] * math.cos(theta1) + LINK_LENGTHS[1] * math.cos(theta2) + LINK_LENGTHS[2] * math.cos(theta3)
    y = LINK_LENGTHS[0] * math.sin(theta1) + LINK_LENGTHS[1] * math.sin(theta2) + LINK_LENGTHS[2] * math.sin(theta3)
    return x, y


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


def make_playback_frames(args: argparse.Namespace) -> list[tuple[str, tuple[float, float], list[float]]]:
    frames = []
    qpos = [math.radians(angle) for angle in DEFAULT_QPOS_DEGREES]
    tool_angle = math.radians(args.tool_angle)
    for path_name, targets in build_required_paths(args):
        for target in targets:
            qpos = solve_planar_ik(target, qpos, tool_angle)
            frames.append((path_name, target, qpos))
    return frames


def run_viewer(mujoco, model, data, args: argparse.Namespace) -> None:
    import mujoco.viewer

    frames = make_playback_frames(args)
    if not frames:
        raise SystemExit("No viewer playback frames were generated.")
    target_marker_id = model.site("target_marker").id
    frame_index = 0
    last_frame_time = time.time()
    frame_interval = 1.0 / 60.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            if step_start - last_frame_time >= frame_interval:
                path_name, target, qpos = frames[frame_index]
                data.qpos[:] = qpos
                data.qvel[:] = 0
                data.qacc[:] = 0
                model.site_pos[target_marker_id][0] = target[0]
                model.site_pos[target_marker_id][1] = target[1]
                model.site_pos[target_marker_id][2] = 0.006
                mujoco.mj_forward(model, data)
                viewer.sync()
                frame_index = (frame_index + 1) % len(frames)
                last_frame_time = step_start
            else:
                viewer.sync()
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
