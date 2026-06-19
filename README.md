# InverseKin

This project simulates a 3R planar robot in MuJoCo. The robot has:

- Three revolute joints rotating about the z-axis for planar motion.
- Link 1 and link 3 at the same height.
- A raised bridge-style link 2 connecting link 1 to link 3.
- A pen mounted at the end of link 3, with the pen tip touching the white writing plane.
- Industrial-style cylindrical joint housings and tube links, with a low planar base.

The MuJoCo model is:

```text
models/3r_planar_pen.xml
```

## Setup

Create or repair the virtual environment:

```bash
scripts/setup_venv.sh
```

You can activate it if you want an interactive shell:

```bash
source venv/bin/activate
```

## Generate Data

Use `main.py` as the entry point:

```bash
python3 main.py
```

This generates one combined CSV:

```text
outputs/trajectory_data.csv
```

The CSV contains horizontal, vertical, and zig-zag writing-surface paths. Each row records:

```text
path,sample,time_s,target_x_m,target_y_m,x_m,y_m,z_m,theta1_deg,theta2_deg,theta3_deg,torque1_nm,torque2_nm,torque3_nm
```

Customize the data generation:

```bash
python3 main.py --samples-per-line 151 --zigzag-rows 9 --zigzag-samples-per-row 51 --output outputs/custom_trajectory_data.csv
```

## Viewer

Open the MuJoCo viewer:

```bash
scripts/open_viewer.sh
```

The viewer plays the horizontal, vertical, and zig-zag trajectories in a loop.

On macOS, you can also double-click:

```text
open_viewer.command
```
