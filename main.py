from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class JointState:
    theta1: float
    theta2: float
    theta3: float


@dataclass(frozen=True)
class Planar3RRobot:
    link_1: float = 140.0
    link_2: float = 110.0
    link_3: float = 125.0
    base_height: float = 55.0
    middle_link_height: float = 110.0
    pen_length: float = 55.0

    def forward_kinematics(self, state: JointState) -> list[Point3D]:
        """Return base, three revolute joints, and pen tip positions."""
        a1 = math.radians(state.theta1)
        a2 = a1 + math.radians(state.theta2)
        a3 = a2 + math.radians(state.theta3)

        base = Point3D(0.0, 0.0, self.base_height)
        joint_1 = Point3D(
            self.link_1 * math.cos(a1),
            self.link_1 * math.sin(a1),
            self.base_height,
        )
        joint_2 = Point3D(
            joint_1.x + self.link_2 * math.cos(a2),
            joint_1.y + self.link_2 * math.sin(a2),
            self.middle_link_height,
        )
        joint_3 = Point3D(
            joint_2.x + self.link_3 * math.cos(a3),
            joint_2.y + self.link_3 * math.sin(a3),
            self.base_height,
        )
        pen_tip = Point3D(joint_3.x, joint_3.y, self.base_height - self.pen_length)
        return [base, joint_1, joint_2, joint_3, pen_tip]


def svg_polyline(points: Iterable[tuple[float, float]], **attrs: str) -> str:
    attr_text = " ".join(f'{svg_attr_name(key)}="{value}"' for key, value in attrs.items())
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{point_text}" {attr_text} />'


def svg_circle(point: tuple[float, float], radius: float, **attrs: str) -> str:
    attr_text = " ".join(f'{svg_attr_name(key)}="{value}"' for key, value in attrs.items())
    return f'<circle cx="{point[0]:.2f}" cy="{point[1]:.2f}" r="{radius}" {attr_text} />'


def svg_text(point: tuple[float, float], label: str, **attrs: str) -> str:
    attr_text = " ".join(f'{svg_attr_name(key)}="{value}"' for key, value in attrs.items())
    return f'<text x="{point[0]:.2f}" y="{point[1]:.2f}" {attr_text}>{label}</text>'


def svg_attr_name(name: str) -> str:
    return "class" if name == "class_" else name.replace("_", "-")


def translate_xy(points: list[Point3D], dx: float, dy: float) -> list[tuple[float, float]]:
    return [(point.x + dx, dy - point.y) for point in points]


def translate_xz(points: list[Point3D], dx: float, baseline: float) -> list[tuple[float, float]]:
    return [(point.x + dx, baseline - point.z) for point in points]


def render_robot_svg(robot: Planar3RRobot, state: JointState) -> str:
    points = robot.forward_kinematics(state)
    top_points = translate_xy(points[:4], 170.0, 260.0)
    side_points = translate_xz(points[:4], 170.0, 560.0)
    pen_top = translate_xy([points[3], points[4]], 170.0, 260.0)
    pen_side = translate_xz([points[3], points[4]], 170.0, 560.0)

    x_values = [point.x for point in points[:4]]
    y_values = [point.y for point in points[:4]]
    work_radius = robot.link_1 + robot.link_2 + robot.link_3
    surface_y = 560.0

    svg_parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="650" viewBox="0 0 900 650">',
        "<style>",
        "text { font-family: Inter, Arial, sans-serif; fill: #23313d; }",
        ".label { font-size: 17px; font-weight: 700; }",
        ".small { font-size: 13px; }",
        "</style>",
        '<rect width="900" height="650" fill="#f7f9fb" />',
        '<rect x="40" y="40" width="820" height="560" rx="8" fill="#ffffff" stroke="#d8e0e8" />',
        svg_text((70, 80), "3R planar robot - top view", class_="label"),
        svg_text((70, 390), "Side view: raised middle link and pen on writing surface", class_="label"),
        f'<circle cx="170" cy="260" r="{work_radius:.2f}" fill="none" stroke="#d9e6ef" stroke-dasharray="8 8" />',
        '<line x1="70" y1="260" x2="820" y2="260" stroke="#edf2f6" />',
        '<line x1="170" y1="80" x2="170" y2="350" stroke="#edf2f6" />',
        svg_polyline(top_points, fill="none", stroke="#1f6feb", stroke_width="18", stroke_linecap="round", stroke_linejoin="round"),
        svg_polyline(top_points, fill="none", stroke="#93c5fd", stroke_width="6", stroke_linecap="round", stroke_linejoin="round"),
        svg_polyline(pen_top, fill="none", stroke="#263238", stroke_width="5", stroke_linecap="round"),
        '<circle cx="170" cy="260" r="18" fill="#334155" />',
        '<circle cx="170" cy="260" r="7" fill="#f8fafc" />',
        '<rect x="70" y="555" width="750" height="16" fill="#f8fafc" stroke="#d8e0e8" />',
        '<line x1="70" y1="560" x2="820" y2="560" stroke="#b8c4d0" stroke-width="2" />',
        svg_polyline(side_points[:2], fill="none", stroke="#1f6feb", stroke_width="16", stroke_linecap="round"),
        svg_polyline(side_points[1:3], fill="none", stroke="#f59e0b", stroke_width="16", stroke_linecap="round"),
        svg_polyline(side_points[2:4], fill="none", stroke="#1f6feb", stroke_width="16", stroke_linecap="round"),
        svg_polyline(side_points, fill="none", stroke="#fef3c7", stroke_width="5", stroke_linecap="round", stroke_linejoin="round"),
        svg_polyline(pen_side, fill="none", stroke="#263238", stroke_width="6", stroke_linecap="round"),
        svg_text((points[3].x + 182.0, surface_y - 16.0), "pen", class_="small"),
        svg_text((70, 590), "white writing surface", class_="small"),
        svg_text((560, 104), f"joint angles: {state.theta1:.0f} deg, {state.theta2:.0f} deg, {state.theta3:.0f} deg", class_="small"),
        svg_text((560, 126), f"pen mount: x={points[3].x:.1f}, y={points[3].y:.1f}, z={points[3].z:.1f}", class_="small"),
        svg_text((560, 148), f"pen tip: x={points[4].x:.1f}, y={points[4].y:.1f}, z={points[4].z:.1f}", class_="small"),
        svg_text((560, 170), f"workspace x=[{min(x_values):.1f}, {max(x_values):.1f}], y=[{min(y_values):.1f}, {max(y_values):.1f}]", class_="small"),
    ]

    for index, point in enumerate(top_points[:4]):
        fill = "#f8fafc" if index else "#334155"
        stroke = "#0f172a" if index else "#334155"
        svg_parts.append(svg_circle(point, 10.0, fill=fill, stroke=stroke, stroke_width="3"))
        svg_parts.append(svg_text((point[0] + 12.0, point[1] - 12.0), f"J{index}", class_="small"))

    for index, point in enumerate(side_points[:4]):
        fill = "#f8fafc" if index else "#334155"
        stroke = "#0f172a" if index else "#334155"
        svg_parts.append(svg_circle(point, 8.0, fill=fill, stroke=stroke, stroke_width="3"))

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a 3R planar robot diagram.")
    parser.add_argument("--theta1", type=float, default=35.0, help="Joint 1 angle in degrees.")
    parser.add_argument("--theta2", type=float, default=-50.0, help="Joint 2 angle in degrees.")
    parser.add_argument("--theta3", type=float, default=45.0, help="Joint 3 angle in degrees.")
    parser.add_argument("--output", type=Path, default=Path("outputs/3r_planar_robot.svg"), help="SVG output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot = Planar3RRobot()
    state = JointState(args.theta1, args.theta2, args.theta3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_robot_svg(robot, state), encoding="utf-8")
    tip = robot.forward_kinematics(state)[-1]
    print(f"Created {args.output}")
    print(f"Pen tip on writing surface at x={tip.x:.1f}, y={tip.y:.1f}, z={tip.z:.1f}")


if __name__ == "__main__":
    main()
