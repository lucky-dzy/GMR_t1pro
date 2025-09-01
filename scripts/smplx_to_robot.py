import argparse
import pathlib
import os
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print
import mujoco
from scipy.spatial.transform import Rotation as R

def output_motion(frames, out_filename, motion_weight, frame_duration):
  with open(out_filename, "w") as f:
    f.write("{\n")
    f.write("\"LoopMode\": \"Wrap\",\n")
    f.write("\"FrameDuration\": " + str(frame_duration) + ",\n")
    f.write("\"EnableCycleOffsetPosition\": true,\n")
    f.write("\"EnableCycleOffsetRotation\": true,\n")
    f.write("\"MotionWeight\": " + str(motion_weight) + ",\n")
    f.write("\n")

    f.write("\"Frames\":\n")

    f.write("[")
    for i in range(frames.shape[0]):
      curr_frame = frames[i]

      if i != 0:
        f.write(",")
      f.write("\n  [")

      for j in range(frames.shape[1]):
        curr_val = curr_frame[j]
        if j != 0:
          f.write(", ")
        f.write("%.5f" % curr_val)

      f.write("]")

    f.write("\n]")
    f.write("\n}")

  return

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsKicks_c3d/G8_-__roundhouse_left_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/TWIST-dev/motion_data/AMASS/KIT_572_dance_chacha11_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsPunches_c3d/E1_-__Jab_left_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1Running_c3d/Run_C24_-_quick_side_step_left_stageii.npz",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openlong", "tlibot_t1pro", "tlibot_t1pro_20dof", "tlibot_t1pro_23dof"],
        default="openlong",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )

    args = parser.parse_args()


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
   
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",)
    

    curr_frame = 0
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []
    
    # Start the viewer
    i = 0
    toe_pos_list = []
    pos_vel_list = []
    root_vel_list = []
    root_angvel_list = []
    root_pos_list = []
    root_ang_list = []
    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
        
        # Update task targets.
        smplx_data = smplx_data_frames[i]

        # retarget
        qpos = retarget.retarget(smplx_data)

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            # human_motion_data=smplx_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
        )
        if args.save_path is not None:
            qpos_list.append(qpos)
        
        # save txt
        mujoco.mj_forward(robot_motion_viewer.model, robot_motion_viewer.data)
        al_id = mujoco.mj_name2id(robot_motion_viewer.model, mujoco.mjtObj.mjOBJ_BODY, "arm_left_wrist_yaw_link")
        ar_id = mujoco.mj_name2id(robot_motion_viewer.model, mujoco.mjtObj.mjOBJ_BODY, "arm_right_wrist_yaw_link")
        ll_id = mujoco.mj_name2id(robot_motion_viewer.model, mujoco.mjtObj.mjOBJ_BODY, "leg_left_ankle_pitch_link")
        lr_id = mujoco.mj_name2id(robot_motion_viewer.model, mujoco.mjtObj.mjOBJ_BODY, "leg_right_ankle_pitch_link")
        base_id = mujoco.mj_name2id(robot_motion_viewer.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        pos_base = robot_motion_viewer.data.xpos[base_id].copy()
        quat_base = robot_motion_viewer.data.xquat[base_id].copy()
      
        al_world = robot_motion_viewer.data.xpos[al_id].copy()
        ar_world = robot_motion_viewer.data.xpos[ar_id].copy()
        ll_world = robot_motion_viewer.data.xpos[ll_id].copy()
        lr_world = robot_motion_viewer.data.xpos[lr_id].copy()

        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, quat_base)
        R_base = mat.reshape(3, 3)
        al_rel = (al_world - pos_base) @ R_base.T
        ar_rel = (ar_world - pos_base) @ R_base.T
        ll_rel = (ll_world - pos_base) @ R_base.T
        lr_rel = (lr_world - pos_base) @ R_base.T
        tar_toe_pos_local = np.squeeze(np.concatenate([al_rel, ar_rel, ll_rel, lr_rel], axis=-1))
        toe_pos_list.append(tar_toe_pos_local)
        if fps_counter > 1:
            euler_angles = R.from_quat(qpos[3:7][[1,2,3,0]]).as_euler('xyz')
            
            # visualize
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                # human_motion_data=smplx_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
            )
            root_pos_list.append(qpos[:3])
            root_ang_list.append(euler_angles)
            root_vel = (qpos[:3] - root_pos_list[-2]) / (1.0 / tgt_fps)
            root_vel_list.append(root_vel)
            root_angvel = (euler_angles - root_ang_list[-2]) / (1.0 / tgt_fps)
            root_angvel_list.append(root_angvel)
        else:
            root_pos_list.append(np.zeros(3))
            root_ang_list.append(np.zeros(3))
            root_vel_list.append(np.zeros(3))
            root_angvel_list.append(np.zeros(3))
            
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

        # save txt
        qvel = np.zeros_like(qpos[7:])
        pos_vel_list.append(qvel)
        for i in range(len(qpos_list)-1):
            fps_time = 1.0 / tgt_fps
            if i > 0:
                qvel = (dof_pos[i+1] - dof_pos[i]) / fps_time
            pos_vel_list.append(qvel)
        # joint_pose, joint_pose_vel, end_pose
        new_frames = np.zeros([len(qpos_list), 52])
        for i in range(len(dof_pos)):
            curr_pose = np.concatenate([
                dof_pos[i], pos_vel_list[i], toe_pos_list[i]
            ])
            new_frames[i] = curr_pose
            
        max_value = np.max(new_frames[:, 20:40])
        print("最大值是:", max_value)
        mean_value = np.mean(new_frames[:, 20:40])
        print("平均值是:", mean_value)
        
        txt_path = args.save_path.replace('.pkl', '.txt')
        output_motion(new_frames, txt_path, 0.5, (1.0/tgt_fps))
        print(f"Saved txt to {txt_path}")
       
        new_frames = np.zeros([len(qpos_list), 52])
        for i in range(len(dof_pos)):
            curr_pose = np.concatenate([
                root_pos_list[i], root_ang_list[i],    dof_pos[i], 
                root_vel_list[i], root_angvel_list[i], pos_vel_list[i]
            ])
            new_frames[i] = curr_pose  
        txt_path = args.save_path.replace('.pkl', '_vis.txt')
        output_motion(new_frames, txt_path, 0.5, (1.0/tgt_fps))
        print(f"Saved txt to {txt_path}")  
    
    robot_motion_viewer.close()
