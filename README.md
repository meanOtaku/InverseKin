# InverseKin

This project starts with a 3R planar robot model:

- Link 1 and link 3 sit at the same height.
- Link 2 is raised and connects the two same-height links.
- A pen is mounted at the end of link 3 and reaches down to a white writing surface.

Run the first model:

```bash
python3 main.py
```

This creates:

```text
outputs/3r_planar_robot.svg
```

You can change the joint angles:

```bash
python3 main.py --theta1 30 --theta2 -45 --theta3 60
```

## MuJoCo simulation

The MuJoCo model is here:

```text
models/3r_planar_pen.xml
```

It contains:

- Three revolute joints rotating about the z-axis for planar motion.
- Link 1 and link 3 at the same height.
- A raised bridge-style link 2 connecting link 1 to link 3.
- A pen mounted at the end of link 3, with the pen tip touching the white writing plane.
- Position actuators (not torque motors) on each joint, so MuJoCo's contact solver physically blocks link 1 and link 3 from passing through each other when the arm folds back on itself.

Create and use the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install MuJoCo:

```bash
python -m pip install -r requirements.txt
```

Open the interactive viewer (default). Each joint continuously sweeps back and forth across its full range so you can see the arm move:

```bash
python simulate_mujoco.py
```

Run a headless simulation check instead (holds the given pose, no sweeping motion):

```bash
python simulate_mujoco.py --headless
```

`--theta1`/`--theta2`/`--theta3` only affect the headless check and the initial pose printout; the live viewer overrides them every frame to drive the sweep:

```bash
python simulate_mujoco.py --headless --theta1 30 --theta2 -45 --theta3 60
```

## Workspace analysis

Sample the MuJoCo model's joint limits to map every reachable pen-tip position on the writing surface, compute the workspace area (convex hull), and plot it:

```bash
python workspace_analysis.py
```

This creates:

```text
outputs/workspace.png
```

Use `--step` to change the joint-angle sampling resolution in degrees (default `2.0`; smaller is slower but denser):

```bash
python workspace_analysis.py --step 1.0
```
