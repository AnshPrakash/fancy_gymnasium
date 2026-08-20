"""MuJoCo 3 port of the NDP Jaco joint-position tasks."""

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv

_ASSETS = Path(__file__).with_name("assets")
_INITIAL_JOINTS = np.array(
    [-0.07855885, -1.5, 0.6731897, 1.5, -2.83926611, 0.0], dtype=np.float64
)


class _DNCJacoEnv(MujocoEnv, utils.EzPickle):
    """The released NDP Jaco position-control dynamics without its legacy runtime."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    low: np.ndarray
    high: np.ndarray

    def __init__(self, asset_name: str, render_mode: str | None = None):
        self.scale = 0.05
        self.frame_skip = 5
        self.low = np.array([-0.65, -2.0, -1.4, -1.55, -2.6, -1.8, -0.26, -0.35, -0.24])
        self.high = np.array([0.68, 0.86, 1.0, 1.75, 1.5, 1.8, 1.5, 1.5, 1.5])
        super().__init__(
            model_path=str(_ASSETS / asset_name),
            frame_skip=self.frame_skip,
            observation_space=spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
            render_mode=render_mode,
        )
        self.model.opt.timestep = 0.01
        self.action_space = spaces.Box(-5.0, 5.0, shape=(9,), dtype=np.float32)
        self.init_qpos[:6] = _INITIAL_JOINTS

    def _finger_com(self) -> np.ndarray:
        return np.mean(
            [
                self.data.body(name).xpos
                for name in (
                    "jaco_link_finger_1",
                    "jaco_link_finger_2",
                    "jaco_link_finger_3",
                )
            ],
            axis=0,
        )

    def _apply_joint_delta(self, action: np.ndarray) -> None:
        delta = np.asarray(action, dtype=np.float64).copy()
        if delta.shape != (9,):
            raise ValueError(f"Expected a 9D action, got {delta.shape}")
        delta[6:] = delta[7]  # The released task ties all three fingers.
        self.data.qpos[:9] = np.clip(
            self.data.qpos[:9] + self.scale * delta, self.low, self.high
        )
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = self.data.qfrc_bias[: self.model.nu]

    def _set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)

    def viewer_setup(self) -> None:
        if self.mujoco_renderer.viewer is not None:
            self.mujoco_renderer.viewer.cam.distance = 4.0


class DNCPickPosEnv(_DNCJacoEnv):
    """Contact-conditioned Jaco approach, grasp, and lift task from NDP."""

    def __init__(self, render_mode: str | None = None):
        super().__init__("picker_pos.xml", render_mode=render_mode)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(34,), dtype=np.float64
        )
        utils.EzPickle.__init__(self, render_mode=render_mode)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate((self.data.qpos, self.data.qvel, self._finger_com()))

    def reset_model(self) -> np.ndarray:
        qpos, qvel = self.init_qpos.copy(), self.init_qvel.copy()
        qpos[1] = -1.0
        noise = self.np_random.uniform(-0.02, 0.02, size=3)
        noise[-1] = 0.0
        qpos[9:12] = np.array([0.55, 0.15, 0.03]) + noise
        qvel[9:12] = 0.0
        self.num_close = 0
        self._set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action: np.ndarray):
        self._apply_joint_delta(action)
        reward = 0.0
        in_hand_frames = 0
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            object_position = self.data.body("object").xpos
            distance = np.linalg.norm(object_position - self._finger_com())
            frame_reward = (
                object_position[2]
                if object_position[2] >= 0.08 and distance < 0.1
                else 0.0
            )
            self.num_close += frame_reward > 0.0
            in_hand_frames += frame_reward > 0.0
            reward += frame_reward
        object_position = self.data.body("object").xpos
        distance = np.linalg.norm(object_position - self._finger_com())
        success = distance < 0.15 and object_position[2] > 0.08
        return (
            self._get_obs(),
            float(reward),
            False,
            False,
            {
                "distance": distance,
                "timeInHand": in_hand_frames,
                "success": bool(success),
            },
        )


class DNCThrowPosEnv(_DNCJacoEnv):
    """Jaco ballistic-release-to-bin task from NDP."""

    def __init__(self, render_mode: str | None = None):
        super().__init__("throw_pos.xml", render_mode=render_mode)
        self.low = np.array([-0.65, -1.7, -1.4, -1.8, -2.6, -1.8, -0.26, -0.35, -0.24])
        self.high = np.array([0.68, 1.5, 1.0, 1.8, 1.5, 1.8, 1.2, 1.0, 1.0])
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(41,), dtype=np.float64
        )
        utils.EzPickle.__init__(self, render_mode=render_mode)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            (
                self.data.qpos,
                self.data.qvel,
                self._finger_com(),
                self.relative_box_position,
            )
        )

    def reset_model(self) -> np.ndarray:
        qpos, qvel = self.init_qpos.copy(), self.init_qvel.copy()
        qpos[1] = -0.1
        qpos[6:9] = 0.9
        noise = self.np_random.uniform(-0.02, 0.02, size=3)
        noise[-1] = 0.0
        qpos[9:12] = np.array([0.6, 0.1, 0.03]) + noise
        qvel[9:12] = 0.0
        self._set_state(qpos, qvel)
        object_position = self.data.body("object").xpos
        goal_position = self.data.body("goal").xpos
        self.relative_box_position = goal_position - object_position
        self.initial_goal_distance = np.linalg.norm(object_position - goal_position)
        return self._get_obs()

    def step(self, action: np.ndarray):
        self._apply_joint_delta(action)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        object_position = self.data.body("object").xpos
        goal_position = self.data.body("goal").xpos
        distance = np.linalg.norm((goal_position - object_position)[:2])
        reward = (
            40.0
            if distance < 0.18
            else 40.0 * (1.0 - min(1.0, distance / self.initial_goal_distance))
        )
        return (
            self._get_obs(),
            float(reward),
            False,
            False,
            {
                "distance": distance,
                "norm_distance": distance / self.initial_goal_distance,
                "success": bool(distance < 0.19),
            },
        )
