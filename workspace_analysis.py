from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

DEFAULT_MODEL = Path("models/3r_planar_pen.xml")
DEFAULT_OUTPUT = Path("outputs/workspace.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map the reachable pen-tip workspace on the writing surface.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the MJCF model.")
    parser.add_argument("--step", type=float, default=2.0, help="Joint angle sampling step, in degrees.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to save the workspace plot.")
    return parser.parse_args()


def sample_joint_angles(jnt_range: np.ndarray, step_deg: float) -> np.ndarray:
    lo_deg = np.degrees(jnt_range[0])
    hi_deg = np.degrees(jnt_range[1])
    num_points = max(2, round((hi_deg - lo_deg) / step_deg) + 1)
    return np.radians(np.linspace(lo_deg, hi_deg, num_points))


def compute_reachable_points(model, step_deg: float) -> np.ndarray:
    data = mujoco.MjData(model)
    pen_tip_id = model.site("pen_tip").id

    angles_1 = sample_joint_angles(model.jnt_range[0], step_deg)
    angles_2 = sample_joint_angles(model.jnt_range[1], step_deg)
    angles_3 = sample_joint_angles(model.jnt_range[2], step_deg)

    points = np.empty((angles_1.size * angles_2.size * angles_3.size, 2))
    index = 0
    for theta1, theta2, theta3 in itertools.product(angles_1, angles_2, angles_3):
        data.qpos[:] = (theta1, theta2, theta3)
        mujoco.mj_kinematics(model, data)
        points[index] = data.site_xpos[pen_tip_id][:2]
        index += 1
    return points


def plot_workspace(points: np.ndarray, hull: ConvexHull, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(points[:, 0], points[:, 1], s=1, color="#1f6feb", alpha=0.15, label="reachable pen-tip points")

    hull_points = points[hull.vertices]
    hull_loop = np.vstack([hull_points, hull_points[:1]])
    ax.plot(hull_loop[:, 0], hull_loop[:, 1], color="#d97706", linewidth=2, label="workspace boundary (convex hull)")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("3R planar robot - reachable workspace on writing surface")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved {output}")


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    points = compute_reachable_points(model, args.step)
    hull = ConvexHull(points)

    print(f"Sampled {len(points)} joint configurations (step={args.step} deg).")
    print(f"Workspace area (convex hull): {hull.volume:.6f} m^2")
    print(f"x range: [{points[:, 0].min():.4f}, {points[:, 0].max():.4f}] m")
    print(f"y range: [{points[:, 1].min():.4f}, {points[:, 1].max():.4f}] m")

    plot_workspace(points, hull, args.output)


if __name__ == "__main__":
    main()
