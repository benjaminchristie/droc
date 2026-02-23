import torch
import numpy as np
from .....src.franka_panda.franka import Franka
from scipy.spatial.transform import Rotation


class GripperState:
    def __init__(self, state):
        self.width = state


class Gripper:
    def __init__(self, panda, conn_gripper):
        self.panda = panda
        self.conn_gripper = conn_gripper
        self.state = 0

    def _get_state(self):
        message = str(self.conn_gripper.recv(2048))[2:-2]
        state_str = list(message.split(','))
        for idx in range(len(state_str)):
            if state_str[idx] == "g":
                state_str = state_str[idx+1:idx+1+3]
        try:
            state_vector = [float(item) for item in state_str]
            states = {}
            states["gripper_width"] = state_vector[0]
            states["max_width"] = state_vector[1]
            states["temperature"] = state_vector[2]
        except:
            return None
        return states

    def get_state(self):
        x = self._get_state()
        return GripperState(x["gripper_width"])  # robot_policy.py expects this format
    
    def grasp(self, **kwargs):
        self.goto(0.0)

    def goto(self, width: float, speed=None, force=None):
        if width <= 0.078 / 2.0: # max_width / 2.0
            send_msg = "s,c,"
        else:
            send_msg = "s,o,"
        self.conn_gripper.send(send_msg.encode())


class Robot:
    def __init__(self, *args, **kwargs):
        self.panda = Franka(*args, **kwargs)
        self.conn_robot = self.panda.connect(8080)
        self.conn_gripper = self.panda.connect(8081)

        self.panda.go2position(self.conn_robot)
        self.panda.send2gripper(self.conn_gripper, "0")
        self.gripper_open = float(1)
        
        
    def _get_gripper_state(self):
        message = str(self.conn_gripper.recv(2048))[2:-2]
        state_str = list(message.split(','))
        for idx in range(len(state_str)):
            if state_str[idx] == "g":
                state_str = state_str[idx+1:idx+1+3]
        try:
            state_vector = [float(item) for item in state_str]
            states = {}
            states["gripper_width"] = state_vector[0]
            states["max_width"] = state_vector[1]
            states["temperature"] = state_vector[2]
        except:
            return None
        return states
        
    def _get_state(self) -> dict:
        state = self.panda.readState(self.conn_robot)
        state["gripper_open"] = [self.gripper_open]
        return state

    def _move_to_ee_pose(self, p: list[float], euler: list[float]) -> int:
        position_threshold = 0.005
        orientation_threshold = 0.05
        target_pos = np.array(p, dtype=float)
        target_euler = np.array(euler, dtype=float)
        for _ in range(1000):
            state = self._get_state()
            q_curr_inv = Rotation.from_euler("xyz", state["x"][3:], degrees=False).inv()
            q_goal = Rotation.from_euler("xyz", target_euler, degrees=False)
            q_diff = q_goal * q_curr_inv
            delta_rot = q_diff.as_euler("xyz", degrees=False)
            xdot = np.concatenate(
                [0.3 * (target_pos - state["x"][:3]), 0.3 * delta_rot]
            )
            if (
                np.linalg.norm(p - state["x"][:3]) <= position_threshold
                and np.linalg.norm(target_euler - state["x"][3:])
                <= orientation_threshold
            ):
                return 0
            qdot = self.panda.xdot2qdot(xdot, state)
            self.panda.send2robot(self.conn_robot, qdot)
        return 0

    def move_to_ee_pose(self, position, orientation) -> int:
        p = None
        o = None
        if isinstance(position, torch.tensor):
            p = position.cpu().detach().numpy().tolist()
        elif isinstance(position, np.ndarray):
            p = position.tolist()
        else:
            p = position
        if isinstance(orientation, torch.tensor):
            o = orientation.cpu().detach().numpy().tolist()
        elif isinstance(orientation, np.ndarray):
            o = orientation.tolist()
        else:
            o = orientation
        euler = Rotation.from_quat(o)
        return self._move_to_ee_pose(p, euler)

    def get_ee_pos(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._get_state()
        pos = state["x"][:3]
        orn = state["x"][3:]
        quat = Rotation.from_euler("xyz", orn, degrees=False)
        quat = quat.as_quat()
        return (torch.FloatTensor(pos), torch.FloatTensor(quat))


class PandaEnv:
    """
    Put your own implementation for Franka Panda here
    """

    def __init__(self):
        self.robot = Robot()
        self.gripper = Gripper(self.robot.panda, self.robot.conn_gripper)

    def reset(self, reset_gripper: bool = True):
        self.robot.panda.go2position(self.robot.conn_robot)
        if reset_gripper:
            self.gripper.goto(1.0)


# need to implement
# gripper.grasp
# gripper.goto(width, speed=speed, force=force)
# robot.move_to_ee_pos(popsition=ee_pos, orientation=quat)
# robot.get_ee_pos() -> torch.Tensor [0]: position [1]: quaterion orientation
# reset(reset_gripper=bool)
