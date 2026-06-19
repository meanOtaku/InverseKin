from __future__ import annotations

import argparse
import math
import time
from pathlib import Path


DEFAULT_MODEL = Path("models/3r_planar_pen.xml")
DEFAULT_QPOS_DEGREES = (35.0, -50.0, 45.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 3R planar pen robot in MuJoCo.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the MJCF model.")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps to run.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the interactive MuJoCo viewer.",
    )
    parser.add_argument("--theta1", type=float, default=DEFAULT_QPOS_DEGREES[0], help="Joint 1 angle in degrees.")
    parser.add_argument("--theta2", type=float, default=DEFAULT_QPOS_DEGREES[1], help="Joint 2 angle in degrees.")
    parser.add_argument("--theta3", type=float, default=DEFAULT_QPOS_DEGREES[2], help="Joint 3 angle in degrees.")
    return parser.parse_args()


def load_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MuJoCo is not installed in this Python environment.\n"
            "Install it with: python3 -m pip install -r requirements.txt"
        ) from exc
    return mujoco


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}")

    mujoco = load_mujoco()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    data.qpos[:] = [math.radians(args.theta1), math.radians(args.theta2), math.radians(args.theta3)]
    mujoco.mj_forward(model, data)

    pen_tip_id = model.site("pen_tip").id
    print("Loaded MuJoCo model:", args.model)
    print(f"Initial joint angles: {args.theta1:.1f}, {args.theta2:.1f}, {args.theta3:.1f} degrees")
    print_pen_tip(data.site_xpos[pen_tip_id])

    if args.headless:
        for _ in range(args.steps):
            mujoco.mj_step(model, data)
        print(f"Ran {args.steps} simulation steps.")
        print_pen_tip(data.site_xpos[pen_tip_id])
    else:
        run_viewer(mujoco, model, data)


def print_pen_tip(position) -> None:
    x, y, z = position
    print(f"Pen tip position: x={x:.4f}, y={y:.4f}, z={z:.4f}")


def run_viewer(mujoco, model, data) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
