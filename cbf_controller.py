"""
Control Barrier Function (CBF) based controller for tracker agent.
This module implements CBF-QP with hard constraints and dual CBF formulation:
- Visibility CBF: maintains target visibility via polygon_sdf_grad_lse
- Obstacle CBF: avoids obstacles via SDF_RT_circular
- Hard constraints: if QP is infeasible, fallback to reference input
"""
import math
import numpy as np
import cvxpy as cp
import map_config
from map_config import EnvParameters

# ============================================================================
# Tunable Parameters (可调参数)
# ============================================================================

GAMMA_DISTANCE_THRESHOLD = 15.0      # gamma = 1 if distance > threshold else 5
CBF_VISIBILITY_MARGIN = 10.0         # Reduced from 40 for better feasibility
CBF_OBSTACLE_MARGIN = map_config.agent_radius * 1.5  # Reduced from 4x for better feasibility
CBF_OBSTACLE_L_WEIGHT = 1.5          # Angular weight for obstacle constraint (l in cbf_qp.py)

# Tracker Control Gains (追踪控制增益)
TRACKER_VELOCITY_GAIN = 20
TRACKER_ANGULAR_GAIN = 20

NAVIGATION_GAP_THRESHOLD = map_config.agent_radius * 2.5

# ============================================================================
# Utility Functions
# ============================================================================

def _normalize_angle(angle_deg: float):
    """Normalize angle to [-180, 180] range."""
    angle_deg = angle_deg % 360.0
    if angle_deg > 180.0:
        angle_deg -= 360.0
    return float(angle_deg)

# ============================================================================
# CBF Controller with Dual Constraints
# ============================================================================

class CBFController:
    """
    Dual-constraint CBF controller:
    1) Visibility CBF (keeps target visible using polygon_sdf_grad_lse)
    2) Obstacle CBF (avoids obstacles using SDF_RT_circular raytracing)

    Hard constraints are enforced in the QP. If the QP is infeasible, the
    controller falls back to the reference input.
    """

    def __init__(self, env=None, max_speed=None, max_turn_rate=None):
        """
        Args:
            env: Environment object exposing polygon_sdf_grad_lse and SDF_RT_circular
            max_speed: Maximum forward speed
            max_turn_rate: Maximum angular velocity (rad/s)
        """
        self.env = env
        self.max_speed = max_speed if max_speed is not None else float(
            getattr(map_config, 'tracker_speed', 2.4)
        )

        if max_turn_rate is not None:
            self.max_turn_rate = max_turn_rate
        else:
            max_turn_deg = float(getattr(map_config, "tracker_max_angular_speed", 10.0))
            dt = 1.0 / 40.0
            self.max_turn_rate = math.radians(max_turn_deg) / dt

    def _sample_closest_obstacle(self, rbt_state):
        """
        Sample the closest obstacle point using raytracing.
        Directly ported from cbf_qp.py's sample_cbf method.

        Args:
            rbt_state: np.ndarray of shape (3,) -> [x, y, theta]

        Returns:
            np.ndarray([x, y]) of closest obstacle point or None
        """
        if self.env is None or not hasattr(self.env, 'SDF_RT_circular'):
            return None

        try:
            # Use circular raytracing with radius=100, resolution=300
            # This matches cbf_qp.py's sample_cbf
            rt_pts = self.env.SDF_RT_circular(rbt_state, 100, 300)
            if rt_pts is None or len(rt_pts) == 0:
                return None

            rt_pts = np.array(rt_pts)
            if rt_pts.shape[0] == 0:
                return None
            
            # Calculate distances from robot to all scan points
            rbt_center = rbt_state[:2] + map_config.pixel_size * 0.5
            dists = np.linalg.norm(rt_pts - rbt_center, axis=1)
            
            # Filter by inner and outer radius (matching cbf_qp.py)
            inner_radius = 0.1
            outer_radius = 30.0
            mask = (dists >= inner_radius) & (dists <= outer_radius)
            
            if not np.any(mask):
                return None
            
            # Find the closest valid point
            valid_indices = np.where(mask)[0]
            min_idx = valid_indices[np.argmin(dists[mask])]
            
            return rt_pts[min_idx]
            
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: obstacle sampling failed: {exc}")
            return None

    def _rotation_matrix(self, theta):
        """2D rotation matrix."""
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

    def solve_cbf_qp(self, rbt_state, target_state, u_ref, ydot=None):
        """
        Solve the hard-constrained dual CBF QP.
        Implementation follows cbf_qp.py's solvecvx method.

        Args:
            rbt_state: np.ndarray [x, y, theta] 像素坐标
            target_state: np.ndarray [x, y, theta] or [x, y] 像素坐标
            u_ref: np.ndarray [v_ref, w_ref]
            ydot: np.ndarray target velocity [vx, vy, vtheta] (optional)

        Returns:
            np.ndarray [v, w] safe control or reference on fallback
        """
        if u_ref is None or not np.all(np.isfinite(u_ref)):
            return np.array([0.0, 0.0])
        if self.env is None or not hasattr(self.env, 'polygon_sdf_grad_lse'):
            return np.array(u_ref, dtype=np.float64)

        # Normalize inputs
        if target_state.shape[0] == 2:
            target_state = np.append(target_state, [0.0])
        if ydot is None:
            ydot = np.zeros(3)

        rbt_state = rbt_state.reshape((3, 1))
        target_state = target_state.reshape((3, 1))
        ydot = ydot.reshape((3, 1))
        u_ref_2d = u_ref.reshape((2, 1))

        try:
            # ================= Visibility CBF =================
            rdrx, rdry, sdf_vis = self.env.polygon_sdf_grad_lse(
                rbt_state.flatten(), target_state.flatten(), debug=False
            )
            
            theta = rbt_state[2, 0]
            
            # G(x) matrix for unicycle dynamics
            G_mat = np.array([
                [np.cos(theta), 0],
                [np.sin(theta), 0],
                [0, 1]
            ])

            # Visibility barrier function: h = -sdf - margin
            h_vis = -sdf_vis - CBF_VISIBILITY_MARGIN
            
            # Lie derivatives
            L_G_h_vis = -rdrx @ G_mat  # (1,2)
            dh_vis_dt = -rdry @ ydot   # (1,1)
            
            # Adaptive gamma based on distance
            dist_vis = abs(sdf_vis)
            gamma_vis = 1.0 if dist_vis > GAMMA_DISTANCE_THRESHOLD else 5.0

            # CBF constraint: LGh * u + dhdt + gamma * h >= 0
            u_var = cp.Variable((2, 1))
            # 修正：去掉多余的负号
            vis_constraint = L_G_h_vis @ u_var + dh_vis_dt + gamma_vis * h_vis >= 0

            # ================= Obstacle CBF ===================
            # Sample closest obstacle using raytracing (following cbf_qp.py)
            obs_pos = self._sample_closest_obstacle(rbt_state.flatten())
            obs_constraint = None
            
            if obs_pos is not None:
                # Vector from robot to obstacle
                rbt_center = rbt_state[:2].flatten() + map_config.pixel_size * 0.5
                obs_vec = (obs_pos - rbt_center).reshape((2, 1))
                obs_dist = float(np.linalg.norm(obs_vec))
                
                # Gradient direction (normalized)
                obs_grad = obs_vec / max(obs_dist, 1e-6)
                
                # Barrier function: h = distance - safety_margin
                h_obs = obs_dist - CBF_OBSTACLE_MARGIN
                
                # Adaptive gamma for obstacle
                # FIXED: Lower gamma when close to strictly limit approach speed
                gamma_obs = 0.5 if obs_dist < GAMMA_DISTANCE_THRESHOLD else 2.0
                
                # Rotation matrix at current heading
                R_theta = self._rotation_matrix(theta)
                
                # Increase rotation penalty slightly
                l_weight = 2.0
                
                # DR-CBF constraint (following cbf_qp.py formula)
                # -grad^T * R(theta) * [[1, 0], [0, l]] * u + gamma * h >= 0
                A_obs = -obs_grad.T @ R_theta @ np.array([
                    [1, 0],
                    [0, l_weight]
                ])

                obs_constraint = A_obs @ u_var + gamma_obs * h_obs >= 0

            # ================= QP Solve =======================
            # Objective: minimize ||u - u_ref||^2
            objective = cp.Minimize(cp.sum_squares(u_var - u_ref_2d))
            
            # Constraints
            constraints = [
                u_var[0] >= 0,
                u_var[0] <= self.max_speed,
                u_var[1] >= -self.max_turn_rate,
                u_var[1] <= self.max_turn_rate,
                vis_constraint
            ]
            
            if obs_constraint is not None:
                constraints.append(obs_constraint)

            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.OSQP, verbose=False, warm_start=False)

            if problem.status == cp.OPTIMAL and u_var.value is not None:
                return u_var.value.flatten()
            else:
                # Fallback to reference input if QP infeasible
                print(f"[CBF QP] Infeasible: status={problem.status}, falling back to reference")
                return np.array(u_ref, dtype=np.float64)
                
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: CBF QP solver failed: {exc}")
            return np.array(u_ref, dtype=np.float64)


# ============================================================================
# Helper Functions
# ============================================================================

def _get_navigation_heading(radar_norm, current_heading_rad, target_bearing_rad, max_range):
    """Follow-the-Gap heuristic to pick a collision-free heading."""
    dists = (radar_norm + 1.0) * 0.5 * max_range
    n_rays = len(dists)
    angle_step = 2.0 * math.pi / n_rays

    target_global = current_heading_rad + target_bearing_rad
    target_idx = int(round(target_global / angle_step)) % n_rays
    check_indices = [(target_idx - 1) % n_rays, target_idx, (target_idx + 1) % n_rays]

    blocked = any(dists[idx] < NAVIGATION_GAP_THRESHOLD for idx in check_indices)
    if not blocked:
        return target_bearing_rad

    safe_indices = [i for i, d in enumerate(dists) if d > NAVIGATION_GAP_THRESHOLD]
    if not safe_indices:
        best_idx = int(np.argmax(dists))
    else:
        target_global = (target_global + math.pi) % (2 * math.pi) - math.pi
        best_idx = -1
        min_diff = float('inf')
        for idx in safe_indices:
            ray_global = idx * angle_step
            diff = ray_global - target_global
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) < min_diff:
                min_diff = abs(diff)
                best_idx = idx

    if best_idx != -1:
        best_global = best_idx * angle_step
        best_rel = best_global - current_heading_rad
        best_rel = (best_rel + math.pi) % (2 * math.pi) - math.pi
        return best_rel

    return target_bearing_rad


# ============================================================================
# CBF Tracker Policy
# ============================================================================

class CBFTracker:
    """Tracker policy using dual-constraint hard CBF-QP controller."""

    def __init__(self, env=None):
        self.env = env
        self.controller = CBFController(env=env)
        self.last_u = np.zeros(2)

    def reset(self):
        self.last_u = np.zeros(2)

    def get_action(self, observation, privileged_state=None):
        """
        Generate action given observation (optionally using privileged state).

        Args:
            observation: Tracker observation vector
            privileged_state: dict with absolute states (expects tracker/target keys)

        Returns:
            np.ndarray [ang_acc_norm, lin_acc_norm]
        """
        obs = np.asarray(observation, dtype=np.float64)

        # Basic kinematics
        max_speed = float(getattr(map_config, 'tracker_speed', 2.4))
        max_turn_deg = float(getattr(map_config, "tracker_max_angular_speed", 10.0))
        dt = 1.0 / 40.0
        phys_max_w = math.radians(max_turn_deg) / dt

        heading_deg = (float(obs[2]) + 1.0) * 180.0
        heading_rad = math.radians(heading_deg)

        # Radar processing
        radar_start = 11
        radar_end = radar_start + EnvParameters.RADAR_RAYS
        radar_norm = obs[radar_start:radar_end]
        max_range = float(EnvParameters.FOV_RANGE)

        dist_norm = float(obs[3])
        bearing_norm = float(obs[4])
        target_dist = (dist_norm + 1.0) / 2.0 * max_range
        target_bearing_rad = math.radians(bearing_norm * 180.0)

        nav_bearing_rad = _get_navigation_heading(
            radar_norm, heading_rad, target_bearing_rad, max_range
        )

        v_ref = np.clip(TRACKER_VELOCITY_GAIN * target_dist, 0, max_speed)
        w_ref = np.clip(TRACKER_ANGULAR_GAIN * nav_bearing_rad, -phys_max_w, phys_max_w)
        u_ref = np.array([v_ref, w_ref])

        # Use privileged state when available to run the hard CBF-QP
        if (
            privileged_state is not None
            and isinstance(privileged_state, dict)
            and 'tracker' in privileged_state
            and 'target' in privileged_state
        ):
            rbt_state = np.array([
                privileged_state['tracker'].get('x', 0.0),
                privileged_state['tracker'].get('y', 0.0),
                privileged_state['tracker'].get('theta', 0.0),
            ])
            tgt_state = np.array([
                privileged_state['target'].get('x', 0.0),
                privileged_state['target'].get('y', 0.0),
                privileged_state['target'].get('theta', 0.0),
            ])
            u_star = self.controller.solve_cbf_qp(rbt_state, tgt_state, u_ref)
        else:
            u_star = u_ref

        v_cmd, w_cmd = u_star[0], u_star[1]

        # Optimization: Use QP output directly as command reference
        # Bypassing the normalization round-trip preserves precision and avoids scaling errors
        desired_speed = float(v_cmd)
        desired_angle_delta = math.degrees(w_cmd * dt)

        v_cmd, w_cmd = u_star[0], u_star[1]

        # Convert QP velocity [v, w] to Environment Action [delta_deg, speed_frac]
        # The environment uses env_lib.agent_move(angle_delta, speed_factor)
        # where speed_factor is [0, 1] multiplier of max_speed.
        
        # 1. Angular Change (Degrees per step)
        delta_deg = math.degrees(w_cmd * dt)
        
        # 2. Linear Speed Fraction (0.0 to 1.0)
        speed_frac = v_cmd / max_speed if max_speed > 1e-6 else 0.0
        
        # Clip for safety (though QP handles most constraints)
        speed_frac = np.clip(speed_frac, 0.0, 1.0)
        
        return np.array([delta_deg, speed_frac], dtype=np.float32)

