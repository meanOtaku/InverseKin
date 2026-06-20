# InverseKin

This project simulates a 3R planar robot in MuJoCo. The robot has:

- Three revolute joints rotating about the z-axis for planar motion.
- Link 1 and link 3 at the same height.
- A raised bridge-style link 2 connecting link 1 to link 3.
- A pen mounted at the end of link 3, with the pen tip touching the white writing plane.
- Industrial-style cylindrical joint housings and tube links, with a low planar base.
- A corner-mounted base at `x=-0.32 m`, `y=-0.32 m` on the writing board.

The MuJoCo model is:

```text
models/3r_planar_pen.xml
```

## Setup

Create or repair the virtual environment:

```bash
scripts/setup_venv.sh
```

## Run Viewer

Open the animated MuJoCo viewer:

```bash
python3 main.py
```

The viewer runs one quarter-circle workspace sweep from the corner-mounted robot base. The end effector traces a full 0 to 90 degree arc, then the radius is reduced by `1 mm`, and the sweep continues until all points are reached. After the final point is saved, the viewer simulation stops.

Adjust viewer speed:

```bash
python3 main.py --speed 2.0
python3 main.py --speed 0.5
```

## Generate CSV

Run headless at full speed:

```bash
python3 main.py --headless
```

This writes:

```text
outputs/trajectory_data.csv
outputs/final_pose_map.csv
outputs/workspace_area.svg
```

Each CSV row records:

```text
path,sample,time_s,radius_m,angle_deg,target_x_m,target_y_m,x_m,y_m,z_m,theta1_deg,theta2_deg,theta3_deg,torque1_nm,torque2_nm,torque3_nm
```

The final pose map records one compact row per reached workspace point:

```text
path,sample,radius_m,angle_deg,target_x_m,target_y_m,final_x_m,final_y_m,theta1_deg,theta2_deg,theta3_deg
```

Customize the generated data:

```bash
python3 main.py --headless --max-radius 0.34 --min-radius 0.19 --radius-step 0.001 --arc-points 91 --output outputs/custom_trajectory_data.csv --pose-output outputs/custom_final_pose_map.csv --plot-output outputs/custom_workspace_area.svg
```

The default sweep creates `151` quarter-circle radius bands and `13,741` samples. All three joints move during the sweep, including joint 3. The default path keeps joint centers at least `0.0803 m` apart, clearing the visual joint cap diameter of about `0.078 m`.
