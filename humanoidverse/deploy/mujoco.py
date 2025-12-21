# Unified Robot Control Interface for HumanoidVerse
# Weiji Xie @ 2025.03.04

import mujoco, mujoco_viewer, mujoco.viewer
import glfw 
import os
import time
import signal
import sys
from pathlib import Path

import torch
from humanoidverse.deploy import URCIRobot
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as sRot
import logging
from utils.config_utils import *  # noqa: E402, F403
# add argparse arguments

from typing import Dict, Optional, Tuple
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger
from humanoidverse.utils.helpers import np2torch, torch2np
from humanoidverse.utils.real.rotation_helper import *
from humanoidverse.utils.noise_tool import noise_process_dict, RadialPerturbation
from description.robots.dtype import RobotExitException

import numpy as np

def wrap_to_pi_float(angles:float):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles



class MViewerPlugin:
    # TODO: visualize the motion keypoint in MujocoViewer
    is_pause = False
    
    def _make_viewer(self):
        def _key_callback(key):
                #  Keyboard mapping:
                #  ----------------
                #  |   K L ; '    |
                #  |   , . /      |
                #  ----------------
                key_char = chr(key)
                
                print(key, key_char)
                
                if key == 27: # Esc
                    self.viewer.close()
                
                if key_char == ' ':
                    self.is_pause = not self.is_pause
                
                if key_char == 'w':
                    self.cmd[1] += 0.1
                elif key_char == 's':
                    self.cmd[1] -= 0.1
                elif key_char == 'a': #
                    self.cmd[0] += 0.1
                elif key_char == 'd':
                    self.cmd[0] -= 0.1
                elif key_char == 'e':
                    if self.heading_cmd:
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]+np.pi/20)
                    else:
                        self.cmd[2] += 0.1
                elif key_char == 'q': 
                    if self.heading_cmd:                            
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]-np.pi/20)
                    else:
                        self.cmd[2] -= 0.1
                elif key_char == 'r':
                    self.cmd = np.array(self.cfg.deploy.defcmd)
                elif key == 257: # Enter
                    # self.Reset()
                    self._ref_pid = -2
                elif key_char == '[':
                    self._ref_pid -= 1
                elif key_char == ']':
                    self._ref_pid += 1
                
                print(self.cmd, self._ref_pid)
        # glfw.set_key_callback(self.viewer.window, _key_callback)
        ...
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=_key_callback)
        self.viewer.cam.lookat[:] = np.array([0,0,0.8])
        self.viewer.cam.distance = 3.0        
        self.viewer.cam.azimuth = 30                         # 可根据需要调整角度
        self.viewer.cam.elevation = -30                      # 负值表示从上往下看
        
        
    def render_step(self):
        while self.is_pause:
            time.sleep(0.01)
        
        if self.is_render:
            if self.viewer.is_running():
                self.viewer.sync()
            
                # breakpoint()
            else:
                raise RobotExitException("Mujoco Robot Exit")
        ...


class ViewerPlugin:
    # TODO: visualize the motion keypoint in MujocoViewer
    
    is_recording = False
    fps = 50  # 录屏目标帧率
    _video_buffer_max_size = 1000  # buffer最大帧数，超过后写入临时文件
    
    def _make_viewer(self):
        ...
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.lookat[:] = np.array([0,0,0.6])
        self.viewer.cam.distance = 3.0        
        self.viewer.cam.azimuth = 180                         # 可根据需要调整角度
        self.viewer.cam.elevation = -20                      # 负值表示从上往下看
        def _key_callback(window, key, scancode, action, mods):
            if action == glfw.PRESS:
                #  Keyboard mapping:
                #  ----------------
                #  |   K L ; '    |
                #  |   , . /      |
                #  ----------------

                if key == glfw.KEY_COMMA:
                    self.cmd[1] += 0.1
                elif key == glfw.KEY_SLASH:
                    self.cmd[1] -= 0.1
                elif key == glfw.KEY_L: #
                    self.cmd[0] += 0.1
                elif key == glfw.KEY_PERIOD:
                    self.cmd[0] -= 0.1
                elif key == glfw.KEY_K:
                    if self.heading_cmd:
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]+np.pi/10)
                    else:
                        self.cmd[2] += 0.1
                elif key == glfw.KEY_SEMICOLON:
                    if self.heading_cmd:                            
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]-np.pi/10)
                    else:
                        self.cmd[2] -= 0.1
                elif key == glfw.KEY_APOSTROPHE:
                    self.cmd = np.array(self.cfg.deploy.defcmd)
                elif key == glfw.KEY_ENTER:
                    # self.Reset()
                    self._ref_pid = -2
                elif key == glfw.KEY_LEFT_BRACKET:
                    self._ref_pid -= 1
                elif key == glfw.KEY_RIGHT_BRACKET:
                    self._ref_pid += 1
                elif key>=320 and key<=329:
                    # 小键盘的数字 0-9 ,控制 策略切换
                    self._ref_pid = key-320
                elif key >= 48 and key <= 57:
                    # 0-9 ,控制 策略切换
                    self._ref_pid = key-48
                elif key == glfw.KEY_N:
                    # 切换到下一个动作
                    if self.motion_lib is not None:
                        # 获取动作总数 
                        if hasattr(self.motion_lib, '_num_unique_motions'):
                            total_motions = self.motion_lib._num_unique_motions
                        else:
                            logger.warning("_num_unique_motions not found; fallback to 1.")
                            total_motions = 1 # 保底
                            
                        # 计算下一个动作 ID
                        self.cur_motion_id = (self.cur_motion_id + 1) % total_motions
                        logger.info(f"Loading Motion ID: {self.cur_motion_id} / {total_motions}")
                        
                        # [关键点] 计算机器人当前的朝向 (Yaw)
                        # 我们希望新动作加载进来时，是顺着机器人当前朝向的
                        # MuJoCo 的四元数是 [w, x, y, z]
                        curr_quat_wxyz = self.quat # 假设 self.quat 是实时更新的
                        curr_quat_xyzw = curr_quat_wxyz[[1, 2, 3, 0]]
                        
                        # 计算当前的 Yaw 角度
                        r = sRot.from_quat(curr_quat_xyzw)
                        yaw = r.as_euler('xyz')[2]
                        
                        # 构造一个新的四元数作为 target_heading (绕Z轴旋转 yaw)
                        target_quat = sRot.from_euler('z', yaw).as_quat() # xyzw
                        
                        # [热重载] 加载新动作，并应用朝向对齐
                        # 注意：这里我们假设 load_motions 支持 target_heading 参数
                        self.motion_lib.load_motions(
                            random_sample=False, 
                            start_idx=self.cur_motion_id,
                            target_heading=target_quat # 告诉MotionLib把动作转到这个方向
                        )
                        
                        # 仅重置计时器，不重置物理状态
                        # 不调用 self.Reset() 
                        self.timer = 0 
                        
                        # 如果父类有其他计数器也要清零
                        if hasattr(self, 'motion_times'): self.motion_times[:] = 0
                        if hasattr(self, '_kick_motion_res_counter'): self._kick_motion_res_counter = -1
                        
                        # 从 motion_lib 获取当前动作的真实时长 (秒)
                        # 注意：get_motion_length 通常返回 Tensor，需要转 float
                        real_len = self.motion_lib.get_motion_length(0)
                        if hasattr(real_len, 'item'):
                            real_len = real_len.item()
                        
                        # 覆盖父类的 motion_len
                        self.motion_len = real_len  + 10 #等待10s没反应就复位了
                        
                        logger.info("Transition applied: Motion reloaded aligned to current heading.")
                    else:
                        logger.warning("No motion_lib found, cannot switch motion.")
                else:
                    print('Press key: ',key)
                
                print(self.cmd, self._ref_pid)
            self.viewer._key_callback(window, key, scancode, action, mods)
        glfw.set_key_callback(self.viewer.window, _key_callback)
        
        if self.is_recording:
            logger.info("Recording is True")
            # 初始化基于仿真步的录屏参数
            # 这些参数会在MujocoRobot.__init__中设置
            if not hasattr(self, '_sim_fps'):
                # 如果还没有设置，使用默认值（会在MujocoRobot中覆盖）
                self._sim_fps = 500
            self._recording_frame_counter = 0  # 仿真步计数器
            self._recording_frame_interval = max(1, int(self._sim_fps / self.fps))  # 采样间隔
            logger.info(f"Recording: sim_fps={self._sim_fps}, target_fps={self.fps}, frame_interval={self._recording_frame_interval}")
            
            self._video_buffer = []
            self._video_temp_files = []  # 临时文件列表
            self._video_saved = False  # 标记视频是否已保存，避免重复保存
            viewport = self.viewer.viewport
            self._frame_buffer = np.zeros((viewport.height, viewport.width, 3), dtype=np.uint8)
            self._frame_size = (viewport.height // self.macro_block_size * self.macro_block_size,  #
                                viewport.width // self.macro_block_size * self.macro_block_size)
    macro_block_size = 16
    def render_step(self):
        if self.is_render:
            # if self.viewer.is_running():
            if self.viewer.is_alive:
                self.viewer.render()
                # self.viewer.sync()
                if self.is_recording:
                    # 基于仿真步的采样：每 _recording_frame_interval 个仿真步采样一帧
                    self._recording_frame_counter += 1
                    if self._recording_frame_counter >= self._recording_frame_interval:
                        self._recording_frame_counter = 0
                        
                        mujoco.mjr_readPixels(self._frame_buffer, None, self.viewer.viewport, self.viewer.ctx)
                        frame = self._frame_buffer[::-1].copy()[:self._frame_size[0], :self._frame_size[1]]
                        
                        self._video_buffer.append(frame)
                        
                        # 检查buffer大小，超过限制时写入临时文件
                        if len(self._video_buffer) >= self._video_buffer_max_size:
                            self._flush_video_buffer_to_temp_file()
            else:
                if self.is_recording:
                    logger.info("Mujoco: Saving video ...")
                    self._save_video()
                raise RobotExitException("Mujoco Robot Exit")
    
    def _flush_video_buffer_to_temp_file(self):
        """将当前buffer写入临时文件，清空buffer"""
        if len(self._video_buffer) == 0:
            return
        
        import imageio
        
        # 创建临时文件
        temp_dir = Path(self.cfg.checkpoint).parent.parent / 'renderings' / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / f'video_temp_{len(self._video_temp_files):06d}.mp4'
        
        # 保存当前buffer到临时文件
        imageio.mimsave(temp_file, self._video_buffer, fps=self.fps, macro_block_size=self.macro_block_size)
        self._video_temp_files.append(temp_file)
        
        logger.debug(f"Flushed {len(self._video_buffer)} frames to temp file: {temp_file}")
        self._video_buffer = []  # 清空buffer
    
    def _save_video(self):
        """保存视频，合并所有临时文件和当前buffer"""
        # 如果已经保存过，直接返回，避免重复保存
        if hasattr(self, '_video_saved') and self._video_saved:
            logger.debug("Video already saved, skipping...")
            return
        
        import imageio
        
        model_path:Path = self.cfg.checkpoint
        model_id = model_path.stem.replace('model_', 'ckpt_')
        video_path:Path = model_path.parent.parent / 'renderings' / model_id / f'video_{time.strftime("%Y%m%d_%H%M%S")}.mp4'
        video_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果有临时文件，需要合并
        if len(self._video_temp_files) > 0:
            # 先保存当前buffer到临时文件（如果有内容）
            if len(self._video_buffer) > 0:
                self._flush_video_buffer_to_temp_file()
            
            # 使用流式处理合并临时文件，避免内存溢出
            logger.info(f"Merging {len(self._video_temp_files)} temp files...")
            
            # 使用流式写入，避免一次性加载所有帧到内存
            writer = None
            total_frames = 0
            valid_files = []
            
            try:
                # 创建writer
                writer = imageio.get_writer(video_path, fps=self.fps, macro_block_size=self.macro_block_size)
                
                # 逐个读取临时文件并写入，使用流式处理
                for temp_file in self._video_temp_files:
                    if not temp_file.exists():
                        logger.warning(f"Temp file not found: {temp_file}, skipping...")
                        continue
                    
                    try:
                        reader = imageio.get_reader(temp_file)
                        file_frame_count = 0
                        # 流式读取：逐帧读取并写入，不全部加载到内存
                        for frame in reader:
                            writer.append_data(frame)
                            file_frame_count += 1
                            total_frames += 1
                        reader.close()
                        valid_files.append(temp_file)
                        logger.debug(f"Processed {file_frame_count} frames from {temp_file.name}")
                    except Exception as e:
                        logger.warning(f"Failed to read temp file {temp_file}: {e}, skipping...")
                        continue
                
                writer.close()
                
                if total_frames > 0:
                    logger.info(f"Video saved to {video_path} ({total_frames} frames from {len(valid_files)} files)")
                    self._video_saved = True  # 标记已保存
                else:
                    logger.warning("No frames to save!")
                    # 如果没有任何有效帧，删除空文件
                    try:
                        video_path.unlink()
                    except Exception:
                        pass
                        
            except Exception as e:
                logger.error(f"Failed to merge video files: {e}")
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass
                raise
            
            # 清理临时文件
            temp_dir = None
            for temp_file in self._video_temp_files:
                if temp_dir is None:
                    temp_dir = temp_file.parent
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")
            
            # 清理临时目录（如果为空）
            if temp_dir is not None:
                try:
                    if temp_dir.exists() and not any(temp_dir.iterdir()):
                        temp_dir.rmdir()
                except Exception:
                    pass
        else:
            # 没有临时文件，直接保存当前buffer
            if len(self._video_buffer) > 0:
                imageio.mimsave(video_path, self._video_buffer, fps=self.fps, macro_block_size=self.macro_block_size)
                logger.info(f"Video saved to {video_path} ({len(self._video_buffer)} frames)")
                self._video_saved = True  # 标记已保存
            else:
                logger.warning("No frames to save!")
        

class MujocoRobot(URCIRobot, ViewerPlugin):
    REAL=False
    
    HANG=False
    
    RAND_NOISE  : bool = False
    RAND_DELAY  : bool = True
    RAND_MASK   : bool = False
    RAND_IMU    : bool = False
    RAND_RADIAL : bool = False
    RAND_OFFSET : bool = False
    TANH_ATTACK : bool = False
    
    noise_imu_type:str = 'ou' # 'white' or 'ou' or 'pink'
    noise_imu_rpy:float = 3         # degree
    noise_imu_omega:float = 0.2     # rad/s
    # noise_imu_omega:float = 0.3     # rad/s
    noise_imu_ou_theta:float = 1    # 1/s
    # noise_imu_ou_theta:float = 0.3    # 1/s
    
    noise_ratio = 3e-2
    # delay_ratio = (4, 25) # unit: ms
    delay_ratio = (4, 20) # unit: ms
    mask_ratio = 0.7
    mk_rand_noise = lambda tens, ratio: (
                                            (tens * (1 + torch.randn_like(tens) * ratio) ) 
                                    if isinstance(tens, torch.Tensor) 
                                    else    (tens * (1 + np.random.randn(*tens.shape).astype(tens.dtype) * ratio)) )# type: ignore
    # mk_rand_noise = lambda tens, ratio: tens * (1 + np.random.randn(*tens.shape).astype(tens.dtype) * ratio) # type: ignore
    
    # print_torque = lambda tau: print(f"tau (norm, max) = {torch.norm(tau):.2f}, \t{torch.max(tau):.2f}", end='\r')
    print_torque = lambda tau: print(f"tau (norm, max) = {np.linalg.norm(tau):.2f}, \t{np.max(tau):.2f}", end='\r')
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        def signal_handler(sig, frame):
            logger.info("Ctrl+C  Exiting safely...")
            raise RobotExitException("Mujoco Robot Exiting")
        signal.signal(signal.SIGINT, signal_handler)
        
        self.decimation = cfg.simulator.config.sim.control_decimation
        self.sim_dt = 1/cfg.simulator.config.sim.fps
        assert self.dt == self.decimation * self.sim_dt
        # self._subtimer = 0
        
        # 设置仿真FPS，用于录屏采样
        self._sim_fps = cfg.simulator.config.sim.fps
        
        self.model = mujoco.MjModel.from_xml_path(os.path.join(cfg.robot.asset.asset_root, cfg.robot.asset.xml_file)) # type: ignore
        print("XML", cfg.robot.asset.xml_file)
        self.data = mujoco.MjData(self.model) # type: ignore
        self.model.opt.timestep = self.sim_dt
        print("timestep", self.model.opt.timestep)
        if cfg.deploy.render:
            self.is_render = True
            if cfg.__contains__('recording') and cfg.recording:
                self.is_recording = True
            self._make_viewer()
        else:
            self.is_render = False
            
        self.num_ctrl = self.data.ctrl.shape[0]
        assert self.num_ctrl == self.num_actions, f"Number of control DOFs {self.num_ctrl} does not match number of actions {self.num_actions}"
        
        # noise init
        if self.RAND_IMU:
            if self.noise_imu_type == 'ou':
                sigma = np.sqrt(2*self.noise_imu_ou_theta)
                theta = self.noise_imu_ou_theta
            elif self.noise_imu_type == 'white':
                sigma = 1
                theta = 0
            else:
                raise ValueError(f"Invalid noise type: {self.noise_imu_type}")
            self.noise_imu = noise_process_dict[self.noise_imu_type]((6,), mu=0,sigma=sigma, theta=theta, dt=self.sim_dt)
            
            # self.noise_imu = noise_process_dict['empty']((3,))
            # self.noise_imu = noise_process_dict['ou']((3,), mu=0, sigma=250*np.sqrt(0.02), theta=25, dt=self.sim_dt)
            # self.noise_imu = noise_process_dict['ou']((3,), mu=0, sigma=250, theta=25, dt=self.sim_dt)
            
            # self.noise_imu = noise_process_dict['ou']((6,), mu=0, sigma=1, theta=0.15, dt=self.sim_dt)
            # self.noise_imu = noise_process_dict['white']((6,), mu=0, sigma=1.825)
            
            # self.noise_imu = noise_process_dict['ou']((6,), mu=0, sigma=np.sqrt(2*3), theta=3, dt=self.sim_dt)
            # self.noise_imu = noise_process_dict['white']((6,), mu=0, sigma=1)
            
            # self.noise_imu = noise_process_dict['ou']((3,), mu=0, sigma=np.sqrt(250), theta=25, dt=self.sim_dt)
            # self.noise_imu = noise_process_dict['white']((3,), mu=0, sigma=35.355)
            # self.noise_imu = noise_process_dict['white']((3,), mu=0, sigma=5)
            # self.noise_imu = noise_process_dict['white']((3,), mu=0, sigma=2.236)
            print(self.noise_imu)
            # print('OUProcess', 'mu', self.noise_imu.mu, 'sigma', self.noise_imu.sigma, 'theta', self.noise_imu.theta, 'dt', self.noise_imu.dt)
            # self.noise_imu = noise_process_dict[self.noise_imu_type]((2,), mu=0, sigma=1.0, theta=self.noise_imu_omega, dt=self.sim_dt)
        
        if self.RAND_RADIAL:
            self._radial_perturbation = RadialPerturbation(sigma=0.03, kappa=100)
            self.radial_perturbation = lambda x: self._radial_perturbation(torch.from_numpy(x).reshape(1, -1)).numpy().reshape(-1)
        
        logger.info("Initializing Mujoco Robot")
        logger.info("Task Name: {}".format(cfg.log_task_name))
        logger.info("Robot Type: {}".format(cfg.robot.asset.robot_type))
        
        
        logger.info(f"decimation: {self.decimation}, sim_dt: {self.sim_dt}, dt: {self.dt}")
        logger.info(f"xml_file: {cfg.robot.asset.xml_file}")
        # print(self.decimation, self.sim_dt, self.dt)
        self.Reset()
        
        mujoco.mj_step(self.model, self.data) # type: ignore    


    def _reset(self):
        # 尝试从 MotionLib 初始化状态
        initialized_from_motion = False
        if hasattr(self, 'motion_lib') and self.motion_lib is not None:
            try:
                # 获取第 cur_motion_id 个动作，在时间 0 的状态
                motion_ids = torch.tensor([self.cur_motion_id], dtype=torch.long)
                motion_times = torch.tensor([0.0], dtype=torch.float32)
                
                # 调用 MotionLib 获取状态
                root_res = self.motion_lib.get_motion_state(motion_ids, motion_times)
                
                # 数据转换: Torch -> Numpy
                def to_np(x): return x.detach().cpu().numpy()[0]
                
                root_pos = to_np(root_res["root_pos"])
                root_rot = to_np(root_res["root_rot"]) # XYZW
                dof_pos = to_np(root_res["dof_pos"])
                dof_vel = to_np(root_res["dof_vel"])
                
                # 应用到 MuJoCo data
                self.data.qpos[:3] = root_pos
                # MuJoCo 使用 WXYZ，MotionLib 输出通常是 XYZW
                # 需要根据你的具体 MotionLib 输出确认。
                # 参照原本代码: self.data.qpos[3:7] = np.array(...)[[3,0,1,2]] 说明 MuJoCo 需要 WXYZ
                # 假设 root_rot 是 XYZW:
                self.data.qpos[3] = root_rot[3] # W
                self.data.qpos[4:7] = root_rot[0:3] # XYZ
                
                self.data.qpos[7:] = dof_pos
                self.data.qvel[:] = 0 # 或者使用 dof_vel
                
                logger.info(f"Reset robot to start of Motion ID {self.cur_motion_id}")
                initialized_from_motion = True
            except Exception as e:
                logger.error(f"Failed to reset from motion lib: {e}")
                initialized_from_motion = False
                
        if not initialized_from_motion:
            logger.info("Resetting from Config (Fixed State)")
            self.data.qpos[:3] = np.array(self.cfg.robot.init_state.pos)
            self.data.qpos[3:7] = np.array(self.cfg.robot.init_state.rot)[[3,0,1,2]] # XYZW to WXYZ
            # self.data.qpos[3:7] = np.array([0,0,0.7184,-0.6956])[[3,0,1,2]]  #DEBUG: init quat of JingjiTaiji
            # self.data.qpos[3:7] = np.array([0,0,0.7455,-0.6665])[[3,0,1,2]]  #DEBUG: init quat of NewTaiji
            # self.data.qpos[3:7] = np.array([0,0,0.6894,0.7244])[[3,0,1,2]]  #DEBUG: init quat of Shaolinquan
            self.data.qpos[7:] = self.dof_init_pose
            self.data.qvel[:]   = 0
            
        self.cmd = np.array(self.cfg.deploy.defcmd)
        
        if self.RAND_IMU:
            self.noise_imu.reset()


    @staticmethod
    def pd_control(target_q, q, kp, target_dq, dq, kd):
        '''Calculates torques from position commands
        '''
        if MujocoRobot.RAND_NOISE:
            kp,kd = MujocoRobot.mk_rand_noise(np.array([kp, kd]), MujocoRobot.noise_ratio)
        return (target_q - q) * kp + (target_dq - dq) * kd

    def _get_state(self):
        '''Extracts physical states from the mujoco data structure
        '''
        data = self.data
        self.q_raw = data.qpos.astype(np.double)[7:] # 19 dim
        self.q = self.q_raw.copy()
            # WXYZ
            # 3 dim base pos + 4 dim quat + 12 dim actuation angles
        self.dq_raw = data.qvel.astype(np.double)[6:] # 18 dim ?????
        self.dq = self.dq_raw.copy()
            # 3 dim base vel + 3 dim omega + 12 dim actuation vel
        
        self.pos = data.qpos.astype(np.double)[:3]
        self.quat_raw = data.qpos.astype(np.double)[3:7][[1,2,3,0]] # WXYZ to XYZW
        self.quat = self.quat_raw.copy()
        self.vel = data.qvel.astype(np.double)[:3]
        self.omega_raw = data.qvel.astype(np.double)[3:6]
        self.omega = self.omega_raw.copy()
        
        # print(self.omega_raw)
        if self.RAND_OFFSET:
            self.q = self.q_raw - self._motor_offset
        
        if self.RAND_IMU:
            step_noise_imu = self.noise_imu.x
            
            self.rpy = (quaternion_to_euler_array(self.quat)) + step_noise_imu[:3] * self.noise_imu_rpy *(np.pi/180)  
            self.quat = rpy_to_quaternion_array(self.rpy)
            quat_dist = np.arccos(np.clip(np.abs(np.dot(self.quat_raw, self.quat)), -1.0, 1.0))
            
            self.omega = self.omega + step_noise_imu[3:] * self.noise_imu_omega
            
            # print('quat_dist:', quat_dist)
            # print('rpy', self.rpy)
            # print('omega', self.omega, '\t| omega_raw', self.omega_raw, '\t| step_noise_imu', step_noise_imu[3:])
            # breakpoint()
            
        if self.RAND_RADIAL:
            # self.omega = self.radial_perturbation(self.omega)
            self.dq = self.radial_perturbation(self.dq)
            
            # print(f"{self.omega=}, {self.omega_raw=}")
            print(f"{self.dq=}, {self.dq_raw=}")
        
        r = R.from_quat(self.quat)  # R.from_quat: need xyzw
        self.rpy = quaternion_to_euler_array(self.quat) # need xyzw
        self.rpy[self.rpy > math.pi] -= 2 * math.pi
        self.gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    
        
    
    def _sanity_check(self, target_q):
        unsafe_dof = np.where((np.abs(target_q - self.q) > 2.2 ) | 
                              (np.abs(self.dq) > 20))[0]
        if len(unsafe_dof) > 0:
            for motor_idx in unsafe_dof:
                logger.error(f"Action of joint {motor_idx} is too large.\n"
                                f"target q\t: {target_q[motor_idx]} \n"
                                f"target dq\t: {0} \n"
                                f"q\t\t: {self.q[motor_idx]} \n"
                                f"dq\t\t: {self.dq[motor_idx]}\n")
                # breakpoint()  
        
    _motor_offset = np.array([
        3, 0.5, 2, -0.5, -1, 1,
        -2, 1, -.3, 1, 0.3, 0.1,
        0, 0, 0,
        0, 1, 0, -1,
        -2, 0, 0, 0
    ])*(np.pi/180)   # [23]
        
    def _apply_action(self, target_q):
        
        self._sanity_check(target_q)
        
        rand_mask = np.random.random(self.num_actions) < self.mask_ratio

        rand_delay = np.random.randint((1e-3*self.delay_ratio[0])//self.sim_dt, (1e-3*self.delay_ratio[1])//self.sim_dt) * self.RAND_DELAY
        step_delay = rand_delay//self.decimation
        substep_delay = rand_delay - step_delay * self.decimation
        
        if step_delay ==0:
            old_action = self.history_handler.history['actions'][0, step_delay]
            old_trg_q = np.clip(old_action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
            cur_trg_q = target_q
        else:
            old_action = self.history_handler.history['actions'][0, step_delay]
            cur_action = self.history_handler.history['actions'][0, step_delay+1]
            old_trg_q = np.clip(old_action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
            cur_trg_q = np.clip(cur_action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
                
        for i in range(self.decimation):
            self.GetState()
            
            if self.RAND_DELAY and i < substep_delay:
                target_q = old_trg_q
            elif self.RAND_MASK:
                target_q = cur_trg_q * rand_mask + old_trg_q.numpy() * (1 - rand_mask)
            else:
                target_q = cur_trg_q
                
            # if self.RAND_OFFSET:
            #     target_q = target_q + self._motor_offset
                
            tau = self.pd_control(target_q, self.q, self.kp,
                            0, self.dq, self.kd)  # Calc torques
            
            if self.TANH_ATTACK:
                kappa = {
                    4:15,
                    10:15,
                    13:200,
                    # 14:200,
                }
                # tau = np.tanh(tau)
                for i in kappa.keys():
                    tau[i] = np.tanh(tau[i]/kappa[i]) * kappa[i]
            
            if self.RAND_NOISE: tau = MujocoRobot.mk_rand_noise(tau, MujocoRobot.noise_ratio)
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)  # Clamp torques
            
            
            # MujocoRobot.print_torque(tau)
            # tau*=0
            # print(np.linalg.norm(target_q-self.q), np.linalg.norm(self.dq), np.linalg.norm(tau))
            # self.data.qpos[:3] = np.array([0,0,1])
            self.data.ctrl[:] = tau
            if self.HANG:
                # self.data.ctrl[14] = 0.5
                self.data.qpos[:3] = np.array([0,0,1])
                self.data.qpos[3:7] = np.array([1,0,0,0])

            mujoco.mj_step(self.model, self.data) # type: ignore
    
            if self.RAND_IMU:
                self.noise_imu.step()
            
            self.tracking()
            self.render_step()
            # self._subtimer += 1
        
        
    # Noise Version ApplyAction & Obs
    def ApplyAction(self, action): 
        if self.RAND_NOISE: action = MujocoRobot.mk_rand_noise(action, MujocoRobot.noise_ratio)
        URCIRobot.ApplyAction(self, action)


            
        # breakpoint()

    def Obs(self):
        # return {k: torch2np(v) for k, v in self.obs_buf_dict.items()}

        actor_obs = torch2np(self.obs_buf_dict["actor_obs"]).reshape(1, -1)
        if self.RAND_NOISE:
            actor_obs = MujocoRobot.mk_rand_noise(actor_obs, MujocoRobot.noise_ratio)

        inputs = {"actor_obs": actor_obs}

        if "future_motion_targets" in self.obs_buf_dict:
            inputs["future_motion_targets"] = torch2np(self.obs_buf_dict["future_motion_targets"]).reshape(1, -1)
        if "prop_history" in self.obs_buf_dict:
            inputs["prop_history"] = torch2np(self.obs_buf_dict["prop_history"]).reshape(1, -1)
        return inputs
        

    def _get_motion_to_save_torch(self)->Tuple[float, Dict[str, torch.Tensor]]:
        
        from scipy.spatial.transform import Rotation as sRot
        
        motion_time = (self.timer) * self.dt 

        root_trans = self.pos
        root_rot = self.quat # XYZW
        root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot).as_rotvec()).float() # type: ignore
        dof = self.q
        # T, num_env, J, 3
        # print(self._motion_lib.mesh_parsers.dof_axis)
        pose_aa = torch.cat([root_rot_vec[..., None, :], 
                                torch.from_numpy(self._dof_axis * dof[..., None]), 
                                torch.zeros((self.num_augment_joint, 3))], axis = 0) # type: ignore
        
        return motion_time, {
            'root_trans_offset': torch.from_numpy(root_trans),
            'root_rot': torch.from_numpy(root_rot), # 统一save xyzw
            'dof': torch.from_numpy(dof),
            'pose_aa': pose_aa,
            'action': torch.from_numpy(self.act),
            'actor_obs': np2torch(self.obs_buf_dict['actor_obs']),
            'terminate': torch.zeros((1,)),
            'dof_vel': torch.from_numpy(self.dq),
            'root_lin_vel': torch.from_numpy(self.vel),
            'root_ang_vel': torch.from_numpy(self.omega),
            
            'clock_time': torch.from_numpy(np.array([time.time()])),
            'tau': torch.from_numpy(self.data.ctrl).clone(),
            'cmd': torch.from_numpy(self.cmd),
            'root_rot_raw': torch.from_numpy(self.quat_raw),
            'root_ang_vel_raw': torch.from_numpy(self.omega_raw),
        }
        
    def _get_motion_to_save_np(self)->Tuple[float, Dict[str, np.ndarray]]:
        
        from scipy.spatial.transform import Rotation as sRot
        
        motion_time = (self.timer) * self.dt 

        root_trans = self.pos
        root_rot = self.quat # XYZW
        root_rot_vec = np.array(sRot.from_quat(root_rot).as_rotvec(), dtype=np.float32) # type: ignore
        dof = self.q
        # T, num_env, J, 3
        # print(self._motion_lib.mesh_parsers.dof_axis)
        pose_aa = np.concatenate([root_rot_vec[..., None, :], 
                                np.array(self._dof_axis * dof[..., None]), 
                                np.zeros((self.num_augment_joint, 3))], axis = 0) # type: ignore
        
        return motion_time, {
            'root_trans_offset': (root_trans).copy(),
            'root_rot': (root_rot).copy(), # 统一save xyzw
            'dof': (dof).copy(),
            'pose_aa': pose_aa,
            'action': (self.act).copy(),
            'actor_obs': (self.obs_buf_dict['actor_obs']),
            'terminate': np.zeros((1,)),
            'dof_vel': (self.dq).copy(),
            'root_lin_vel': (self.vel).copy(),
            'root_ang_vel': (self.omega).copy(),
            
            'clock_time': (np.array([time.time()])),
            'tau': (self.data.ctrl).copy(),
            'cmd': (self.cmd).copy()
        }
        
    _get_motion_to_save = _get_motion_to_save_np


    def tracking(self):
        # breakpoint()
        # print(np.linalg.norm(self.data.xpos[6]-self.data.xpos[12]))
        if np.any(self.data.contact.pos[:,2] > 0.01):
            names_list = self.model.names.decode('utf-8').split('\x00')[:40]
            res = np.zeros((6,1),dtype=np.float64)
            geom_name = lambda x: (names_list[self.model.geom_bodyid[x] + 1])
            geom_force = lambda x:mujoco.mj_contactForce(self.model,self.data,x,res) #type:ignore
            
            for contact in self.data.contact:
                if contact.pos[2] > 0.01 and contact.geom1 != 0 and contact.geom2 != 0:
                    geom1_name = geom_name(contact.geom1)
                    geom2_name = geom_name(contact.geom2)
                    logger.warning(f"Warning!!! Collision between '{geom1_name,contact.geom1}' and '{geom2_name,contact.geom2}' at position {contact.pos}.")
                    # breakpoint()
                    
