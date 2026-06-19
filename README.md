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

Create and use the virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install MuJoCo:

```bash
python -m pip install -r requirements.txt
```

Run a headless simulation check:

```bash
python simulate_mujoco.py
```

Open the interactive viewer:

```bash
python simulate_mujoco.py --viewer
```

Try another pose:

```bash
python simulate_mujoco.py --theta1 30 --theta2 -45 --theta3 60
```
