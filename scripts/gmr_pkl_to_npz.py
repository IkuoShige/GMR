"""Convert GMR K1 pkl to npz format using mujoco FK.

Supports two body orderings:
  --body_order mujoco    : MuJoCo DFS order (default, as produced by mj_forward)
  --body_order isaacsim  : Isaac Sim BFS order (for booster_train / Isaac Lab)

Both orderings store body_names in the npz so downstream code can verify.
"""

import argparse
import os
import pickle
import numpy as np
import mujoco

MUJOCO_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "booster_k1", "K1_22dof.xml")


def _get_mujoco_body_names(model):
    """Return body names in MuJoCo DFS order, skipping the world body."""
    return [model.body(i).name for i in range(1, model.nbody)]


def _build_isaacsim_reorder(mj_body_names, isaac_body_names):
    """Build index array to reorder MuJoCo body arrays into Isaac Sim order."""
    reorder = []
    for name in isaac_body_names:
        reorder.append(mj_body_names.index(name))
    return reorder


# Isaac Sim BFS body order for K1_22dof.urdf (obtained from Isaac Lab runtime).
# This is stable as long as the URDF topology does not change.
ISAACSIM_K1_BODY_NAMES = [
    "Trunk", "Head_1", "Left_Arm_1", "Right_Arm_1", "Left_Hip_Pitch", "Right_Hip_Pitch",
    "Head_2", "Left_Arm_2", "Right_Arm_2", "Left_Hip_Roll", "Right_Hip_Roll",
    "Left_Arm_3", "Right_Arm_3", "Left_Hip_Yaw", "Right_Hip_Yaw",
    "left_hand_link", "right_hand_link", "Left_Shank", "Right_Shank",
    "Left_Ankle_Cross", "Right_Ankle_Cross", "left_foot_link", "right_foot_link",
]


def quat_slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        return q0 + t * (q1 - q0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1


def interpolate_motion(root_pos, root_rot_wxyz, dof_pos, input_fps, output_fps):
    n_frames = root_pos.shape[0]
    duration = (n_frames - 1) / input_fps
    out_times = np.arange(0, duration, 1.0 / output_fps)
    n_out = len(out_times)

    phase = out_times / duration
    idx0 = np.floor(phase * (n_frames - 1)).astype(int)
    idx1 = np.minimum(idx0 + 1, n_frames - 1)
    blend = phase * (n_frames - 1) - idx0

    out_pos = root_pos[idx0] * (1 - blend[:, None]) + root_pos[idx1] * blend[:, None]
    out_dof = dof_pos[idx0] * (1 - blend[:, None]) + dof_pos[idx1] * blend[:, None]

    out_rot = np.zeros((n_out, 4))
    for i in range(n_out):
        out_rot[i] = quat_slerp(root_rot_wxyz[idx0[i]], root_rot_wxyz[idx1[i]], blend[i])
        out_rot[i] /= np.linalg.norm(out_rot[i])

    return out_pos, out_rot, out_dof, n_out


def axis_angle_from_quat(q):
    """WXYZ quaternion → axis-angle (3-vector)."""
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    sin_half = np.clip(sin_half, 1e-12, None)
    angle = 2.0 * np.arctan2(sin_half, np.abs(w))
    sign = np.sign(w)
    sign[sign == 0] = 1.0
    return (xyz / sin_half) * angle * sign


def quat_conjugate(q):
    """WXYZ quaternion conjugate."""
    return q * np.array([1, -1, -1, -1])


def quat_mul(q1, q2):
    """WXYZ quaternion multiplication."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def compute_angular_velocity(quats_wxyz, dt):
    """Compute angular velocity from WXYZ quaternion sequence."""
    n = quats_wxyz.shape[0]
    q_prev = quats_wxyz[:-2]
    q_next = quats_wxyz[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    omega = np.concatenate([omega[:1], omega, omega[-1:]], axis=0)
    return omega


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="GMR K1 pkl file")
    parser.add_argument("--output", required=True, help="Output npz file")
    parser.add_argument("--output_fps", type=int, default=50)
    parser.add_argument("--kick_leg", type=str, default=None, choices=["left", "right"])
    parser.add_argument("--mujoco_xml", type=str, default=MUJOCO_XML)
    parser.add_argument("--body_order", type=str, default="mujoco",
                        choices=["mujoco", "isaacsim"],
                        help="Body ordering in output npz: 'mujoco' (DFS) or 'isaacsim' (BFS)")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    input_fps = data["fps"]
    root_pos = data["root_pos"]         # (N, 3)
    root_rot_xyzw = data["root_rot"]    # (N, 4) XYZW
    dof_pos = data["dof_pos"]           # (N, 22)

    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]

    out_pos, out_rot, out_dof, n_out = interpolate_motion(
        root_pos, root_rot_wxyz, dof_pos, input_fps, args.output_fps
    )

    model = mujoco.MjModel.from_xml_path(args.mujoco_xml)
    mj_data = mujoco.MjData(model)

    n_bodies = model.nbody - 1  # exclude world body
    body_pos_w = np.zeros((n_out, n_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((n_out, n_bodies, 4), dtype=np.float32)

    for i in range(n_out):
        mj_data.qpos[:3] = out_pos[i]
        mj_data.qpos[3:7] = out_rot[i]  # WXYZ for mujoco
        mj_data.qpos[7:] = out_dof[i]
        mujoco.mj_forward(model, mj_data)
        body_pos_w[i] = mj_data.xpos[1:]     # skip world body
        body_quat_w[i] = mj_data.xquat[1:]   # WXYZ

    joint_pos = out_dof.astype(np.float32)

    dt = 1.0 / args.output_fps
    joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)

    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)

    body_ang_vel_w = np.zeros_like(body_pos_w)
    for b in range(n_bodies):
        body_ang_vel_w[:, b, :] = compute_angular_velocity(body_quat_w[:, b, :], dt)
    body_ang_vel_w = body_ang_vel_w.astype(np.float32)

    mj_body_names = _get_mujoco_body_names(model)

    if args.body_order == "isaacsim":
        reorder = _build_isaacsim_reorder(mj_body_names, ISAACSIM_K1_BODY_NAMES)
        body_pos_w = body_pos_w[:, reorder]
        body_quat_w = body_quat_w[:, reorder]
        body_lin_vel_w = body_lin_vel_w[:, reorder]
        body_ang_vel_w = body_ang_vel_w[:, reorder]
        out_body_names = ISAACSIM_K1_BODY_NAMES
        print(f"Reordered bodies: mujoco -> isaacsim ({len(reorder)} bodies)")
    else:
        out_body_names = mj_body_names
        print(f"Body order: mujoco DFS ({len(mj_body_names)} bodies)")

    output = {
        "fps": np.array([args.output_fps]),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "body_names": np.array(out_body_names),
    }
    if args.kick_leg:
        output["kick_leg"] = np.array(args.kick_leg)

    np.savez(args.output, **output)
    print(f"Saved {args.output}: {n_out} frames at {args.output_fps}fps, body_order={args.body_order}")
    for k, v in output.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
