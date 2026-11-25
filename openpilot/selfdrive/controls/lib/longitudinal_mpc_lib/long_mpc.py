#!/usr/bin/env python3
import os
import time
import math
import struct
import numpy as np
from openpilot.cereal import log
from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
# WARNING: imports outside of constants will not trigger a rebuild
from openpilot.selfdrive.modeld.constants import index_function
from openpilot.selfdrive.controls.radar_constants import LEAD_ACCEL_TAU
from openpilot.selfdrive.carrot.traffic_stop import get_traffic_stop_distance_adjust, get_traffic_stop_obstacle_distance
from openpilot.selfdrive.controls.lib.longitudinal_preview import LEAD_ACCEL_MIN_TRACK_FRAMES, LeadAccelResponseState, get_lead_accel_mpc_request
from openpilot.selfdrive.controls.lib.longitudinal_cutout import cutout_obstacle_relief
from openpilot.selfdrive.controls.lib.longitudinal_gap_recovery import MIN_HEADROOM_TF, LeadGapState, displayed_follow_distance
from openpilot.selfdrive.controls.lib.longitudinal_braking_gap import (
  COMFORT_RELIEF_MIN_TF, LeadComfortState, comfort_path_has_reserve,
)
from openpilot.selfdrive.controls.lib import longitudinal_gap_safe as gap_safe
from openpilot.selfdrive.controls.lib import longitudinal_soft_brake as soft_brake
from openpilot.selfdrive.controls.lib.longitudinal_approach import ApproachState
from openpilot.selfdrive.controls.lib.longitudinal_stop_gap import StopGapWeightState
from openpilot.selfdrive.carrot.radar_motion.lane_change_gap import LaneChangeGapPlan

if __name__ == '__main__':  # generating code
  from acados.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
else:
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython

from casadi import SX, vertcat

MODEL_NAME = 'long'
LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")
JSON_FILE = os.path.join(LONG_MPC_DIR, "acados_ocp_long.json")

SOURCES = ['lead0', 'lead1', 'cruise', 'e2e']

X_DIM = 3
U_DIM = 1
PARAM_DIM = 8
COST_E_DIM = 5
COST_DIM = COST_E_DIM + 1
CONSTR_DIM = 4

X_EGO_OBSTACLE_COST = 5.
X_EGO_COST = 0.
V_EGO_COST = 0.
A_EGO_COST = 0.
J_EGO_COST = 20.0
A_CHANGE_COST = 200.
A_CHANGE_COST_STARTING = 10. #30.
DANGER_ZONE_COST = 100.
CRASH_DISTANCE = .25
LEAD_DANGER_FACTOR = 0.8 # 0.75
# Cost-side lead prediction: a positive aLeadK decays this many times faster (half the extrapolated
# speed gain), so ego does not copy every small lead acceleration. Braking (aLeadK < 0), the obstacle /
# danger constraint and a lead ego closes on (vRel below the first speed, m/s; full from the second) keep
# the full extrapolation: closing at 28 km/h on a lead speeding up, the weaker one added a brake entry.
LEAD_ACCEL_COST_POS_TAU_FACTOR = 4.0
LEAD_ACCEL_COST_POS_REL_SPEEDS = (-2.0, -0.5)
# Ego speed (m/s): full prediction at or below the first, the weaker one from the second. Level-1 closed-loop
# replay (FOL 47c/479): weaker at every speed added a brake entry per 15 min (2.02 -> 2.09) and lead-departure
# rubber banding 19 -> 22 %; gated to 22-32 km/h it kept 2.02 brake entries, cut sign changes 6.63 -> 6.20/min.
LEAD_ACCEL_COST_POS_EGO_SPEEDS = (6.0, 9.0)
# Blended lead constraint: share of the closing terms kept on top of the steady comfort target, full at or
# below the first planned ego speed (m/s), none from the second.
BLENDED_CLOSING_SPEEDS = (4.0, 7.0)
# s: a_change_cost rises from the starting value to A_CHANGE_COST over this time after a standstill.
A_CHANGE_COST_RAMP_TIME = 0.75
LIMIT_COST = 1e6
ACADOS_SOLVER_TYPE = 'SQP_RTI'


# Fewer timestamps don't hurt performance and lead to
# much better convergence of the MPC with low iterations
N = 12
MAX_T = 10.0
T_IDXS_LST = [index_function(idx, max_val=MAX_T, max_idx=N) for idx in range(N+1)]

T_IDXS = np.array(T_IDXS_LST)
FCW_IDXS = T_IDXS < 5.0
PRED_DANGER_IDXS = (T_IDXS > 0.2) & (T_IDXS < 3.0)
T_DIFFS = np.diff(T_IDXS, prepend=[0.])
A_CHANGE_FACTORS = tuple(np.interp(t, [0.0, 1.0, 2.0], [1.0, 1.0, 0.0]) for t in T_IDXS[:N])
COMFORT_BRAKE = 2.5
STOP_DISTANCE = 6.0

def get_jerk_factor(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.moreRelaxed:
    return 1.0
  elif personality==log.LongitudinalPersonality.relaxed:
    return 1.0
  elif personality==log.LongitudinalPersonality.standard:
    return 1.0
  elif personality==log.LongitudinalPersonality.aggressive:
    return 0.5
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def get_a_change_cost(prev_accel_constraint: bool, a_change_cost_starting: float,
                      response_factor: float = 1.0, ramp: float = 1.0) -> float:
  if prev_accel_constraint:
    base_cost = a_change_cost_starting + (A_CHANGE_COST - a_change_cost_starting) * float(np.clip(ramp, 0.0, 1.0))
  else:
    base_cost = a_change_cost_starting
  return float(base_cost * np.clip(response_factor, 0.0, 1.0))


def binding_reference(x_targets, preferred, binding, ego_speeds):
  """Gap cost reference: from the binding obstacle's target to the closest preferred target."""
  return (x_targets[np.arange(len(binding)), binding] - np.min(preferred, axis=1)) / (np.maximum(ego_speeds, 0.0) + 10.0)


def bind_obstacles(x_targets, preferred):
  """Where cruise (column 2) is the closest obstacle, a lead (columns 0, 1) whose preferred target is the
  closest binds instead; any other closest obstacle (leadTwo, traffic or Tesla stop) keeps its constraint."""
  by_pref = np.argmin(preferred, axis=1)
  by_target = np.argmin(x_targets, axis=1)
  return np.where((by_pref < 2) & (by_target == 2), by_pref, by_target)


def get_T_FOLLOW(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.moreRelaxed:
    return 2.0
  elif personality==log.LongitudinalPersonality.relaxed:
    return 1.75
  elif personality==log.LongitudinalPersonality.standard:
    return 1.45
  elif personality==log.LongitudinalPersonality.aggressive:
    return 1.25
  else:
    raise NotImplementedError("Longitudinal personality not supported")

def get_stopped_equivalence_factor(v_lead):
  return (v_lead**2) / (2 * COMFORT_BRAKE)

def get_safe_obstacle_distance(v_ego, t_follow=None, comfort_brake=COMFORT_BRAKE, stop_distance=STOP_DISTANCE):
  if t_follow is None:
    t_follow = get_T_FOLLOW()
  return (v_ego**2) / (2 * comfort_brake) + t_follow * v_ego + stop_distance

def desired_follow_distance(v_ego, v_lead, comfort_brake, stop_distance, t_follow=None):
  if t_follow is None:
    t_follow = get_T_FOLLOW()
  return get_safe_obstacle_distance(v_ego, t_follow, comfort_brake, stop_distance) - get_stopped_equivalence_factor(v_lead)


def gen_long_model():
  model = AcadosModel()
  model.name = MODEL_NAME

  # set up states & controls
  x_ego = SX.sym('x_ego')
  v_ego = SX.sym('v_ego')
  a_ego = SX.sym('a_ego')
  model.x = vertcat(x_ego, v_ego, a_ego)

  # controls
  j_ego = SX.sym('j_ego')
  model.u = vertcat(j_ego)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  v_ego_dot = SX.sym('v_ego_dot')
  a_ego_dot = SX.sym('a_ego_dot')
  model.xdot = vertcat(x_ego_dot, v_ego_dot, a_ego_dot)

  # live parameters
  a_min = SX.sym('a_min')
  a_max = SX.sym('a_max')
  x_obstacle = SX.sym('x_obstacle')
  prev_a = SX.sym('prev_a')
  lead_t_follow = SX.sym('lead_t_follow')
  lead_danger_factor = SX.sym('lead_danger_factor')
  comfort_brake = SX.sym('comfort_brake')
  stop_distance = SX.sym('stop_distance')
  model.p = vertcat(a_min, a_max, x_obstacle, prev_a, lead_t_follow, lead_danger_factor, comfort_brake, stop_distance)

  # dynamics model
  f_expl = vertcat(v_ego, a_ego, j_ego)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model


def gen_long_ocp():
  ocp = AcadosOcp()
  ocp.model = gen_long_model()

  Tf = T_IDXS[-1]

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  QR = np.zeros((COST_DIM, COST_DIM))
  Q = np.zeros((COST_E_DIM, COST_E_DIM))

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  x_ego, v_ego, a_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]
  j_ego = ocp.model.u[0]

  a_min, a_max = ocp.model.p[0], ocp.model.p[1]
  x_obstacle = ocp.model.p[2]
  prev_a = ocp.model.p[3]
  lead_t_follow = ocp.model.p[4]
  lead_danger_factor = ocp.model.p[5]
  comfort_brake = ocp.model.p[6]
  stop_distance = ocp.model.p[7]

  ocp.cost.yref = np.zeros((COST_DIM, ))
  ocp.cost.yref_e = np.zeros((COST_E_DIM, ))

  desired_dist_comfort = get_safe_obstacle_distance(v_ego, lead_t_follow, comfort_brake, stop_distance)

  # The main cost in normal operation is how close you are to the "desired" distance
  # from an obstacle at every timestep. This obstacle can be a lead car
  # or other object. In e2e mode we can use x_position targets as a cost
  # instead.
  costs = [((x_obstacle - x_ego) - (desired_dist_comfort)) / (v_ego + 10.),
           x_ego,
           v_ego,
           a_ego,
           a_ego - prev_a,
           j_ego]
  ocp.model.cost_y_expr = vertcat(*costs)
  ocp.model.cost_y_expr_e = vertcat(*costs[:-1])

  # Constraints on speed, acceleration and desired distance to
  # the obstacle, which is treated as a slack constraint so it
  # behaves like an asymmetrical cost.
  constraints = vertcat(v_ego,
                        (a_ego - a_min),
                        (a_max - a_ego),
                        ((x_obstacle - x_ego) - lead_danger_factor * (desired_dist_comfort)) / (v_ego + 10.))
  ocp.model.con_h_expr = constraints

  x0 = np.zeros(X_DIM)
  ocp.constraints.x0 = x0
  ocp.parameter_values = np.array([-1.2, 1.2, 0.0, 0.0, lead_t_follow, LEAD_DANGER_FACTOR, comfort_brake, stop_distance])


  # We put all constraint cost weights to 0 and only set them at runtime
  cost_weights = np.zeros(CONSTR_DIM)
  ocp.cost.zl = cost_weights
  ocp.cost.Zl = cost_weights
  ocp.cost.Zu = cost_weights
  ocp.cost.zu = cost_weights

  ocp.constraints.lh = np.zeros(CONSTR_DIM)
  ocp.constraints.uh = 1e4*np.ones(CONSTR_DIM)
  ocp.constraints.idxsh = np.arange(CONSTR_DIM)

  # The HPIPM solver can give decent solutions even when it is stopped early
  # Which is critical for our purpose where compute time is strictly bounded
  # We use HPIPM in the SPEED_ABS mode, which ensures fastest runtime. This
  # does not cause issues since the problem is well bounded.
  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = ACADOS_SOLVER_TYPE
  ocp.solver_options.qp_solver_cond_N = 1

  # More iterations take too much time and less lead to inaccurate convergence in
  # some situations. Ideally we would run just 1 iteration to ensure fixed runtime.
  ocp.solver_options.qp_solver_iter_max = 10
  ocp.solver_options.qp_tol = 1e-3

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = T_IDXS

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LongitudinalMpc:
  def __init__(self, mode='acc', dt=DT_MDL):
    self.mode = mode
    self.dt = dt
    self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)

    self.a_change_cost = A_CHANGE_COST
    self.jerk_cost_factor = 1.0
    self.lead_accel_response_active = False
    self.lead_accel_response_level = 0
    self.obstacle_cost_factor = 1.0
    # Not part of reset(): a solver reset must not restart the safety-distance gap blend.
    self.gap_safe_state = gap_safe.GapSafeState()
    self.cutin_credit = gap_safe.CutinCredit()
    self.lead_gap = None
    self.cutin_shift = 0.0

    self.reset()
    self.source = SOURCES[2]

    self.t_follow = 1.0
    self.desired_distance = 0.0
    self.base_desired_distances = np.zeros(2)
    self.lead_danger_factor = LEAD_DANGER_FACTOR
    self.predicted_danger_margin = 1e3


  def reset(self):
    self._cost_key = None
    self._cost_solver = None
    # self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self.solver.reset()
    # self.solver.options_set('print_level', 2)
    self.v_solution = np.zeros(N+1)
    self.a_solution = np.zeros(N+1)
    self.prev_a = np.array(self.a_solution)
    self.j_solution = np.zeros(N)
    self.yref = np.zeros((N+1, COST_DIM))
    for i in range(N):
      self.solver.cost_set(i, "yref", self.yref[i])
    self.solver.cost_set(N, "yref", self.yref[N][:COST_E_DIM])
    self.x_sol = np.zeros((N+1, X_DIM))
    self.u_sol = np.zeros((N,1))
    self.params = np.zeros((N+1, PARAM_DIM))
    for i in range(N+1):
      self.solver.set(i, 'x', np.zeros(X_DIM))
    self.last_cloudlog_t = 0
    self.status = False
    self.crash_cnt = 0.0
    self.predicted_danger_margin = 1e3
    self.solution_status = 0
    self.lead_accel_response_active = False
    self.lead_accel_response_level = 0
    # timers
    self.lead_response_state = LeadAccelResponseState()
    self.lead_gap_states = (LeadGapState(), LeadGapState())
    self.lead_comfort_states = (LeadComfortState(), LeadComfortState())
    self.soft_brake_state = soft_brake.SoftBrakeState()
    self.stop_gap_state = StopGapWeightState()
    self.approach_state = ApproachState()
    self.obstacle_cost_factor = 1.0
    self.lead_gap_margins = np.zeros((N+1, 2))
    self.solve_time = 0.0
    self.time_qp_solution = 0.0
    self.time_linearization = 0.0
    self.time_integrator = 0.0
    self.x0 = np.zeros(X_DIM)
    self.set_weights()

  def set_cost_weights(self, cost_weights, constraint_cost_weights):
    # Compare the actual double values (including signed zero), not a rounded
    # tuning key. Reset and solver replacement must always resend the costs.
    values = (*cost_weights, *constraint_cost_weights)
    key = (self.mode, struct.pack('10d', *values)) if all(map(math.isfinite, values)) else None
    if key is not None and key == self._cost_key and self.solver is self._cost_solver:
      return
    self._cost_key = None
    W = np.asfortranarray(np.diag(cost_weights))
    for i in range(N):
      # TODO don't hardcode A_CHANGE_COST idx
      # reduce the cost on (a-a_prev) later in the horizon.
      W[4,4] = cost_weights[4] * A_CHANGE_FACTORS[i]
      self.solver.cost_set(i, 'W', W)
    # Setting the slice without the copy make the array not contiguous,
    # causing issues with the C interface.
    self.solver.cost_set(N, 'W', np.copy(W[:COST_E_DIM, :COST_E_DIM]))

    # Set L2 slack cost on lower bound constraints
    Zl = np.array(constraint_cost_weights)
    for i in range(N):
      self.solver.cost_set(i, 'Zl', Zl)
    self._cost_key = key
    self._cost_solver = self.solver

  def set_weights(self, prev_accel_constraint=True,
                  personality=log.LongitudinalPersonality.standard,
                  jerk_factor=1.0,
                  a_change_cost_starting=A_CHANGE_COST_STARTING,
                  a_change_cost_factor=1.0,
                  jerk_cost_factor=1.0,
                  obstacle_cost_factor=1.0,
                  a_change_cost_ramp=1.0):
    #jerk_factor = get_jerk_factor(personality)
    if self.mode == 'acc':
      a_change_cost = get_a_change_cost(
        prev_accel_constraint, a_change_cost_starting, a_change_cost_factor, a_change_cost_ramp,
      )
      applied_jerk_cost_factor = float(np.clip(jerk_cost_factor, 0.0, 1.0))
      cost_weights = [X_EGO_OBSTACLE_COST * max(1.0, float(obstacle_cost_factor)), X_EGO_COST, V_EGO_COST, A_EGO_COST,
                      a_change_cost, jerk_factor * applied_jerk_cost_factor * J_EGO_COST]
      constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]
    elif self.mode == 'blended':
      a_change_cost = 40.0 if prev_accel_constraint else 0
      cost_weights = [0., 0.1, 0.2, 5.0, a_change_cost, 1.0]
      constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]
    else:
      raise NotImplementedError(f'Planner mode {self.mode} not recognized in planner cost set')
    self.a_change_cost = float(a_change_cost)
    self.jerk_cost_factor = float(applied_jerk_cost_factor if self.mode == 'acc' else 1.0)
    self.set_cost_weights(cost_weights, constraint_cost_weights)

  def update_predicted_danger_margin(self, lead, lead_obstacle, t_follow, comfort_brake, stop_distance, lead_gap=None):
    self.predicted_danger_margin = 1e3
    if lead.status:
      if lead_gap is None:
        safe_distance = get_safe_obstacle_distance(self.x_sol[:,1], t_follow, comfort_brake, stop_distance)
        danger_margin = lead_obstacle - self.x_sol[:,0] - self.lead_danger_factor * safe_distance
      else:
        danger_margin = (lead_obstacle - self.x_sol[:,0]
                         - lead_gap.danger_factor * gap_safe.safe_distance(self.x_sol[:,1], lead_gap, stop_distance))
      self.predicted_danger_margin = float(np.min(danger_margin[PRED_DANGER_IDXS]))

  def set_cur_state(self, v, a):
    v_prev = self.x0[1]
    self.x0[1] = v
    self.x0[2] = a
    if abs(v_prev - v) > 2.:  # probably only helps if v < v_prev
      for i in range(N+1):
        self.solver.set(i, 'x', self.x0)

  @staticmethod
  def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau, positive_tau_factor=1.0):
    tau = a_lead_tau * (positive_tau_factor if a_lead > 0.0 else 1.0)
    a_lead_traj = a_lead * np.exp(-tau * (T_IDXS**2)/2.)
    v_lead_traj = np.clip(v_lead + np.cumsum(T_DIFFS * a_lead_traj), 0.0, 1e8)
    x_lead_traj = x_lead + np.cumsum(T_DIFFS * v_lead_traj)
    lead_xv = np.column_stack((x_lead_traj, v_lead_traj))
    return lead_xv

  def process_lead(self, lead):
    v_ego = self.x0[1]
    if lead is not None and lead.status:
      x_lead = lead.dRel
      v_lead = lead.vLead
      a_lead = lead.aLeadK
      a_lead_tau = lead.aLeadTau
    else:
      # Fake a fast lead car, so mpc can keep running in the same mode
      x_lead = 50.0
      v_lead = v_ego + 10.0
      a_lead = 0.0
      a_lead_tau = LEAD_ACCEL_TAU

    # MPC will not converge if immediate crash is expected
    # Clip lead distance to what is still possible to brake for
    min_x_lead = ((v_ego + v_lead)/2) * (v_ego - v_lead) / (-ACCEL_MIN * 2)
    x_lead = np.clip(x_lead, min_x_lead, 1e8)
    v_lead = np.clip(v_lead, 0.0, 1e8)
    a_lead = np.clip(a_lead, -10., 5.)

    lead_xv = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau)
    weak = float(np.interp(v_lead - v_ego, LEAD_ACCEL_COST_POS_REL_SPEEDS, (0.0, 1.0)))
    if LEAD_ACCEL_COST_POS_EGO_SPEEDS[1] > 0.0:
      weak *= float(np.interp(v_ego, LEAD_ACCEL_COST_POS_EGO_SPEEDS, (0.0, 1.0)))
    tau_factor = 1.0 + (LEAD_ACCEL_COST_POS_TAU_FACTOR - 1.0) * weak
    lead_xv_cost = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau, tau_factor)
    return lead_xv, v_lead, lead_xv_cost

  def set_accel_limits(self, min_a, max_a):
    # TODO this sets a max accel limit, but the minimum limit is only for cruise decel
    # needs refactor
    self.cruise_min_a = min_a
    self.max_a = max_a

  def update(self, carrot, reset_state, radarstate, v_cruise, x, v, a, j,
             personality=log.LongitudinalPersonality.standard,
             prev_accel_constraint=True,
             jerk_factor=1.0,
             a_change_cost_starting=A_CHANGE_COST_STARTING,
             lead_accel_response_enabled=False,
             lead_gap_enabled=False,
             lead_track_frames=(0, 0),
             measured_a_ego=0.0,
             cutout_relief_enabled=False,
             tesla_stop_distance=None,
             approach_soft_stop_enabled=False,
             a_change_cost_ramp=1.0):
    tesla_stop_active = (tesla_stop_distance is not None and not reset_state
                         and math.isfinite(tesla_stop_distance) and 0 <= tesla_stop_distance < 255)
    if tesla_stop_active:
      self.mode = 'acc'
    v_ego = self.x0[1]
    a_ego = self.x0[2]
    self.status = radarstate.leadOne.status or radarstate.leadTwo.status
    t_follow = carrot.get_T_FOLLOW(personality, v_ego, a_ego)
    jerk_factor = carrot.jerk_factor

    lead_xv_0, lead_v_0, lead_xv_cost_0 = self.process_lead(radarstate.leadOne)
    lead_xv_1, lead_v_1, lead_xv_cost_1 = self.process_lead(radarstate.leadTwo)

    mode = self.mode
    comfort_brake = carrot.comfort_brake
    stop_distance = carrot.stop_distance
    self.base_desired_distances = np.array([
      desired_follow_distance(v_ego, lead_v_0, comfort_brake, stop_distance, t_follow),
      desired_follow_distance(v_ego, lead_v_1, comfort_brake, stop_distance, t_follow),
    ])

    if mode == 'blended':
      stop_x = 1000.0
    else:
      v_cruise, stop_x = carrot.v_cruise, carrot.stop_dist
      mode = 'acc' if tesla_stop_active else carrot.mode

    # Gap level index as in CarrotPlanner._get_base_t_follow: aggressive (0) = level 1 ... moreRelaxed (3) = level 4.
    gap_level = int(getattr(personality, 'raw', personality))
    gap_safe_on = 0 <= gap_level < gap_safe.GAP_LEVELS
    self.gap_safe_state.update(
      enabled=gap_safe_on, snap=reset_state, dt=self.dt,
      lead_decel=gap_safe.LEAD_DECELS[gap_level] if gap_safe_on else 0.0,
    )
    # ACC and blended lead nodes use the safety-distance gap. Only an unknown gap level keeps the TF
    # computation for leads.
    lead_gap = None
    if mode in ('acc', 'blended') and self.gap_safe_state.active:
      lead_gap = self.gap_safe_state.lead_gap(comfort_brake=comfort_brake, v_ego=float(v_ego), a_ego=float(a_ego))
    self.lead_gap = lead_gap
    # Per lead: the safety-distance gap for a real lead, the TF computation for the fake far lead
    # process_lead() puts in when there is none.
    lead_gaps = tuple(lead_gap if lead.status else None for lead in (radarstate.leadOne, radarstate.leadTwo))
    for lead_index, (gap, lead_v) in enumerate(zip(lead_gaps, (lead_v_0, lead_v_1))):
      if gap is not None:
        self.base_desired_distances[lead_index] = gap_safe.desired_distance(v_ego, lead_v, gap, stop_distance)
    # TF seconds as read by lane-change credit, cutout relief and gap recovery.
    lead_t_follow = t_follow if lead_gap is None else gap_safe.equivalent_t_follow(v_ego, lead_gap)

    # To estimate a safe distance from a moving lead, we calculate how much stopping
    # distance that lead needs as a minimum. We can add that to the current distance
    # and then treat that as a stopped car/obstacle at this new distance.
    def obstacle(lead_xv, gap):
      return lead_xv[:,0] + (get_stopped_equivalence_factor(lead_xv[:,1]) if gap is None else gap_safe.stopped_equivalence(lead_xv[:,1], gap))
    lead_0_obstacle, lead_1_obstacle = obstacle(lead_xv_0, lead_gaps[0]), obstacle(lead_xv_1, lead_gaps[1])
    # How much closer the cost-side prediction (weaker positive lead accel) puts each lead obstacle.
    cost_shifts = np.column_stack([
      np.maximum(0.0, lead_0_obstacle - obstacle(lead_xv_cost_0, lead_gaps[0])),
      np.maximum(0.0, lead_1_obstacle - obstacle(lead_xv_cost_1, lead_gaps[1])),
    ])
    # Ego speeds of the previous plan, moved to the current speed (also the comfort-margin speeds below).
    gap_v = np.maximum(0.0, self.x_sol[:,1] + v_ego - self.x_sol[0,1])
    if mode == 'acc' and approach_soft_stop_enabled and not reset_state:
      approach_weight = self.approach_state.update(
        enabled=True, lead_status=bool(radarstate.leadOne.status), track_id=radarstate.leadOne.radarTrackId,
        distance=float(radarstate.leadOne.dRel), lead_speed=float(radarstate.leadOne.vLead),
        lead_end_speed=float(lead_xv_0[-1, 1]), lead_accel=float(radarstate.leadOne.aLeadK),
        v_ego=float(v_ego), stop_distance=float(stop_distance), dt=self.dt,
        lead_stop_decel=(1.0 / lead_gaps[0].inv_lead_decel) if lead_gaps[0] is not None else COMFORT_BRAKE,
      )
    else:
      approach_weight = self.approach_state.reset()
    # The closing terms keep the full lead prediction: a slower cost-side lead there raised the target and
    # added brake entries (replay level 2: 2.09 -> 2.31 /min). Only the lead position moves (cost_shifts).
    cost_lead_speeds = (lead_xv_0[:,1], lead_xv_1[:,1])
    lead_one = radarstate.leadOne
    if lead_gaps[0] is None:
      # Off (blended, unknown level): forget the lead. No leadOne: keep it, so it gets no credit on return.
      cutin_credit = self.cutin_credit.reset() if lead_gap is None else self.cutin_credit.hold(self.dt)
    else:
      cutin_credit = self.cutin_credit.update(
        enabled=True, track_id=lead_one.radarTrackId, distance=float(lead_one.dRel),
        desired=gap_safe.steady_distance(v_ego, float(lead_one.vLead), lead_gaps[0], stop_distance), closing_speed=-float(lead_one.vRel),
        lead_accel=float(lead_one.aLeadK), cushion=lead_gap.comfort_time * v_ego, dt=self.dt,
        grant=not reset_state and not getattr(carrot, 'lane_change_gap', LaneChangeGapPlan()).active,
      )
    cutin_shift = 0.0
    tf_desired_excess = 0.0

    self.params[:,0] = ACCEL_MIN if not reset_state else a_ego
    # negative accel constraint causes problems because negative speed is not allowed
    self.params[:,1] = max(0.0, self.max_a if not reset_state else a_ego)

    # Update in ACC mode or ACC/e2e blend
    if mode == 'acc':
      #self.params[:,5] = LEAD_DANGER_FACTOR
      # Fake an obstacle for cruise, this ensures smooth acceleration to set speed
      # when the leads are no factor.
      v_lower = v_ego + (T_IDXS * self.cruise_min_a * 1.05)
      # TODO does this make sense when max_a is negative?
      v_upper = v_ego + (T_IDXS * self.max_a * 1.05)
      v_cruise_clipped = np.clip(v_cruise * np.ones(N+1),
                                 v_lower,
                                 v_upper)
      cruise_obstacle = np.cumsum(T_DIFFS * v_cruise_clipped) + get_safe_obstacle_distance(v_cruise_clipped, t_follow, comfort_brake, stop_distance)

      adjust_dist = get_traffic_stop_distance_adjust(
        carrot.trafficStopDistanceAdjust,
        v_ego,
        getattr(carrot, "trafficStopModelLeadOffset", 0.0),
      )
      traffic_stop_obstacle = get_traffic_stop_obstacle_distance(stop_x, cruise_obstacle[0], adjust_dist)
      x2 = traffic_stop_obstacle * np.ones(N+1)

      lane_change = getattr(carrot, 'lane_change_gap', LaneChangeGapPlan())
      lead_0_relief = np.zeros(N+1)
      if cutout_relief_enabled and not reset_state:
        if lane_change.active:
          lead_0_relief = lane_change.credit(
            radarstate.leadOne, T_IDXS, v_ego, self.max_a, lead_t_follow, stop_distance, carrot.dynamicTFollowLC,
          )
        else:
          lead_0_relief = cutout_obstacle_relief(
            radarstate.leadOne, v_ego, T_IDXS, lead_t_follow, stop_distance,
          )
      if cutin_credit > 0.0:
        # The credit never makes the plan looser than the TF one (5250bdff). It moves leadOne's obstacle,
        # and with it the danger floor d_min, at most down to the TF danger floor
        # (LEAD_DANGER_FACTOR x TF distance - v_lead^2 / (2 x COMFORT_BRAKE)), and the cost reference
        # at most down to the TF following target (below).
        tf_distance = get_safe_obstacle_distance(gap_v, t_follow, comfort_brake, stop_distance)
        tf_floor = LEAD_DANGER_FACTOR * tf_distance - get_stopped_equivalence_factor(lead_xv_0[:,1])
        d_min = gap_safe.min_distance(gap_v, lead_xv_0[:,1], lead_gaps[0], stop_distance)
        cutin_shift = np.clip(d_min - tf_floor, 0.0, cutin_credit)
        # How far the TF following target lies above the moved danger floor.
        tf_desired_excess = np.maximum(0.0, tf_distance - get_stopped_equivalence_factor(lead_xv_0[:,1]) - (d_min - cutin_shift))
      # Cut-in credit and cut-out / lane-change relief both move leadOne back; they do not add up.
      lead_0_follow_obstacle = lead_0_obstacle + np.maximum(lead_0_relief, cutin_shift)
      x_obstacles = np.column_stack([lead_0_follow_obstacle, lead_1_obstacle, cruise_obstacle, x2])

      #elif self.source in ['cruise', 'e2e']:
      #  self.params[:,0] = - COMFORT_BRAKE

      # These are not used in ACC mode
      x[:], v[:], a[:], j[:] = 0.0, 0.0, 0.0, 0.0

      #safe_distance = lead_0_obstacle[0] - get_safe_obstacle_distance(v_ego, comfort_brake, stop_distance)
      self.lead_danger_factor = LEAD_DANGER_FACTOR
      self.params[:,5] = self.lead_danger_factor

    elif mode == 'blended':
      self.params[:,5] = 1.0

      # Blended keeps the lead as a constraint only (factor 1.0): a safety-distance lead constrains at its
      # steady comfort target (d_min + comfort time, the equal-speed gap ACC aims at), so a mode change keeps
      # the following gap. The closing terms stay out (BLENDED_CLOSING_SHARE): as a constraint they braked a
      # stopped lead far ahead at once (MPC synthetic: -2.75 to -2.95 m/s^2 in the first second).
      share = np.interp(gap_v, BLENDED_CLOSING_SPEEDS, (1.0, 0.0))
      blended_margins = [
        np.zeros(N+1) if gap is None else gap_safe.comfort_margin(gap_v, lead_xv[:,1], gap._replace(
          closing_time=gap.closing_time * share, closing_factor=gap.closing_factor * share))
        for gap, lead_xv in zip(lead_gaps, (lead_xv_0, lead_xv_1), strict=True)]
      lead_0_blended = lead_0_obstacle - blended_margins[0]
      if cutin_credit > 0.0:
        # A cut-in gets the credit here too, never looser than the TF constraint blended used before.
        tf_target = get_safe_obstacle_distance(gap_v, t_follow, comfort_brake, stop_distance) - get_stopped_equivalence_factor(lead_xv_0[:,1])
        target = gap_safe.min_distance(gap_v, lead_xv_0[:,1], lead_gaps[0], stop_distance) + blended_margins[0]
        lead_0_blended = lead_0_blended + np.clip(target - tf_target, 0.0, cutin_credit)
      x_obstacles = np.column_stack([lead_0_blended, lead_1_obstacle - blended_margins[1]])
      cruise_target = T_IDXS * np.clip(v_cruise, v_ego - 2.0, 1e3) + x[0]
      xforward = ((v[1:] + v[:-1]) / 2) * (T_IDXS[1:] - T_IDXS[:-1])
      x = np.cumsum(np.insert(xforward, 0, x[0]))

      x_and_cruise = np.column_stack([x, cruise_target])
      x = np.min(x_and_cruise, axis=1)

      self.source = 'e2e' if x_and_cruise[1,0] < x_and_cruise[1,1] else 'cruise'

    else:
      raise NotImplementedError(f'Planner mode {self.mode} not recognized in planner update')

    if tesla_stop_active:
      # This is an ego stop endpoint, not a model endpoint or a measured lead.
      # Cancel the standstill gap in the solver's obstacle representation.
      obstacle = tesla_stop_distance + stop_distance
      x_obstacles = np.column_stack((x_obstacles, np.full(N+1, obstacle)))
    # A gap-safe lead obstacle subtracts its own distance in the solver, so obstacles are compared
    # on the TF scale: the smallest of these is the obstacle with the closest target at each node.
    x_targets = x_obstacles.copy()
    for lead_index, gap in enumerate(lead_gaps):
      if gap is not None:
        x_targets[:, lead_index] -= gap_safe.target_offset(gap_v, gap, t_follow, comfort_brake, stop_distance)
    # Stopping behind a lead: the lead that sets the end of the horizon stops in the plan.
    stop_lead_index = int(np.argmin(x_targets[-1, :2])) if mode == 'acc' else 0
    stop_lead = (radarstate.leadOne, radarstate.leadTwo)[stop_lead_index]
    stop_factor = self.approach_state.obstacle_cost_factor() * self.stop_gap_state.update(
      enabled=mode == 'acc' and not reset_state,
      lead_status=bool(stop_lead.status),
      lead_end_speed=float((lead_xv_0, lead_xv_1)[stop_lead_index][-1, 1]),
      lead_sets_stop=mode == 'acc' and bool(x_targets[-1, stop_lead_index] <= np.min(x_targets[-1]) + 1e-6),
      v_ego=v_ego,
      gap_excess=(float(stop_lead.dRel - self.base_desired_distances[stop_lead_index])
                  if stop_lead.status else float('inf')),
      dt=self.dt,
    )
    self.obstacle_cost_factor = stop_factor

    # Extra TF is a comfort preference, not a change to physical lead obstacles,
    # base TF, cruise/map targets or braking constraints. Level 5 bypasses the
    # opening-gap memory. Braking onset never adds a remembered distance.
    gap_x = np.cumsum(T_DIFFS * np.concatenate(([gap_v[0]], (gap_v[1:] + gap_v[:-1]) * 0.5)))
    self.lead_gap_margins[:] = 0.0
    comfort_reliefs = np.zeros_like(self.lead_gap_margins)
    for lead_index, (lead, lead_xv) in enumerate(((radarstate.leadOne, lead_xv_0), (radarstate.leadTwo, lead_xv_1))):
      gap = lead_gaps[lead_index]
      if gap is None:
        gap_safe_distance = get_safe_obstacle_distance(gap_v, t_follow, comfort_brake, stop_distance)
        relief_base_tf = t_follow
        lead_danger_factor = self.params[0,5]
      else:
        gap_safe_distance = gap_safe.safe_distance(gap_v, gap, stop_distance)
        # Comfort relief may spend the comfort time above d_min, never d_min itself.
        relief_base_tf = COMFORT_RELIEF_MIN_TF + gap.comfort_time
        lead_danger_factor = gap.danger_factor
      eligible = (
        mode == 'acc' and lead_gap_enabled and not reset_state
        and not getattr(carrot, 'lane_change_active', False)
        and len(lead_track_frames) > lead_index and lead_track_frames[lead_index] >= LEAD_ACCEL_MIN_TRACK_FRAMES
        and lead.status and lead.radar
      )
      state = self.lead_gap_states[lead_index]
      physical_gaps = lead_xv[:,0] - gap_x
      if gap is None:
        desired_distances = desired_follow_distance(gap_v, lead_xv[:,1], comfort_brake, stop_distance, t_follow)
        lead_equivalence = get_stopped_equivalence_factor(lead_xv[:,1])
      else:
        desired_distances = gap_safe.desired_distance(gap_v, lead_xv[:,1], gap, stop_distance)
        lead_equivalence = gap_safe.stopped_equivalence(lead_xv[:,1], gap)
      state.update(level=carrot.leadAccelResponse, track_id=lead.radarTrackId, enabled=eligible, dt=self.dt,
                   ego_speed=v_ego, lead_speed=lead.vLead if eligible else 0.0, relative_speed=lead.vRel if eligible else 0.0,
                   distance=lead.dRel if eligible else 0.0, desired_distance=self.base_desired_distances[lead_index],
                   base_tf=lead_t_follow, stop_distance=stop_distance,
                   min_headroom=0.0 if gap is None else MIN_HEADROOM_TF)
      self.lead_gap_margins[:,lead_index] = state.margins(
        level=carrot.leadAccelResponse, times=T_IDXS, ego_speeds=gap_v, lead_speeds=lead_xv[:,1], base_tf=lead_t_follow,
        lead_distances=physical_gaps, desired_distances=desired_distances, stop_distance=stop_distance)
      comfort = self.lead_comfort_states[lead_index]
      comfort.update(enabled=eligible, track_id=lead.radarTrackId, dt=self.dt,
                     distance=lead.dRel if eligible else 0.0,
                     relative_speed=lead.vRel if eligible else 0.0)
      danger = lead_xv[:,0] + lead_equivalence - gap_x - lead_danger_factor * gap_safe_distance
      relief = comfort.comfort_relief(dt=self.dt, enabled=eligible and self.lead_gap_margins[0,lead_index] <= 0.01,
                                     base_tf=relief_base_tf, ego_speeds=gap_v, lead_speeds=lead_xv[:,1],
                                     lead_accel=lead.aLeadK if eligible else 0.0, physical_gaps=physical_gaps,
                                     danger_margins=danger, times=T_IDXS, stop_distance=stop_distance)
      self.lead_gap_margins[:,lead_index] -= relief
      comfort_reliefs[:,lead_index] = relief
    base_margins = self.lead_gap_margins.copy()
    def lead_margins(approach):
      """Margins on the lead targets. Without the approach curve they choose the binding obstacle, so the
      approach stays a reference-only change; with it they set the cost reference."""
      margins = base_margins.copy()
      if approach and approach_weight > 0.0:
        # Only where the curve asks for more than the comfort target (the two never stack).
        if lead_gaps[0] is None:
          following = get_safe_obstacle_distance(gap_v, t_follow, comfort_brake, stop_distance)
        else:
          following = gap_safe.safe_distance(gap_v, lead_gaps[0], stop_distance) + gap_safe.comfort_margin(gap_v, cost_lead_speeds[0], lead_gaps[0])
        margins[:,0] += self.approach_state.reference_margins(
          ego_speeds=gap_v, following_distances=following, stop_distance=stop_distance)
      # The safety-distance comfort margin rides on the cost reference like any margin; the
      # displayed target already contains it through base_desired_distances.
      comfort = np.zeros_like(margins)
      for lead_index, gap in enumerate(lead_gaps):
        if gap is not None:
          comfort[:, lead_index] = gap_safe.comfort_margin(gap_v, cost_lead_speeds[lead_index], gap)
      margins += comfort
      if mode == 'acc':
        # The weaker positive lead accel of the cost-side prediction.
        margins += cost_shifts
      # leadOne's cost reference with a cut-in credit: max(TF following target, comfort target - credit), and
      # never more than the comfort target. The moved obstacle already took cutin_shift of the credit.
      if cutin_credit > 0.0:
        margins[:, 0] = np.maximum(np.minimum(margins[:, 0] + cutin_shift, tf_desired_excess), margins[:, 0] - (cutin_credit - cutin_shift))
      return margins, comfort
    self.cutin_shift = cutin_shift
    self.lead_gap_margins, comfort_margins = lead_margins(True)

    # The binding obstacle per node: a lead binds where its comfort target (obstacle - margins) is the
    # closest target, or where its obstacle itself is; otherwise the closest obstacle. A lead looser
    # than cruise even with its margins never binds (no plan below the set speed without a lead).
    def preferred_targets(margins):
      preferred = x_targets.copy()
      if mode == 'acc':
        preferred[:, :2] -= margins
      return preferred
    bind_margins = lead_margins(False)[0] if approach_weight > 0.0 else self.lead_gap_margins
    preferred = preferred_targets(self.lead_gap_margins)
    binding = bind_obstacles(x_targets, preferred_targets(bind_margins))
    if mode == 'acc':
      # Only currently selected leadOne/leadTwo may constrain braking.
      # Entry snapshots and lane-change credit guards never add obstacles.
      bind_preferred = preferred_targets(bind_margins)
      self.source = SOURCES[int(bind_obstacles(x_targets[:1, :4], bind_preferred[:1, :4])[0])]
      if v_cruise == 0 and self.source == 'cruise':
        self.params[:,0] = - carrot.autoNaviSpeedDecelRate
    if tesla_stop_active and binding[0] == x_obstacles.shape[1] - 1:
      self.source = 'e2e'

    response_lead_index = 1 if self.source == 'lead1' else 0
    response_lead = (radarstate.leadOne, radarstate.leadTwo)[response_lead_index]
    response_gap_margin = (
      float(response_lead.dRel - self.base_desired_distances[response_lead_index])
      if response_lead.status else -1.0
    )
    response_track_stable = (
      len(lead_track_frames) > response_lead_index
      and lead_track_frames[response_lead_index] >= LEAD_ACCEL_MIN_TRACK_FRAMES
    )
    response_request = get_lead_accel_mpc_request(
      carrot.leadAccelResponse,
      enabled=(
        mode == 'acc'
        and lead_accel_response_enabled
        and response_track_stable
      ),
      source=self.source,
      lead_status=(
        response_lead.status
        and response_lead.radar
        and response_lead.radarTrackId >= 0
      ),
      a_lead=response_lead.aLeadK,
      a_ego=measured_a_ego,
      v_rel=response_lead.vRel,
      gap_margin=response_gap_margin,
      speed_error=v_cruise - v_ego,
    )
    response_request = self.lead_response_state.update(response_request, self.dt, response_lead.radarTrackId)
    self.lead_accel_response_active = response_request.active
    self.lead_accel_response_level = response_request.level if response_request.active else 0
    self.set_weights(
      prev_accel_constraint,
      personality=personality,
      jerk_factor=jerk_factor,
      a_change_cost_starting=a_change_cost_starting,
      a_change_cost_factor=response_request.a_change_cost_factor,
      jerk_cost_factor=response_request.jerk_cost_factor,
      obstacle_cost_factor=self.obstacle_cost_factor,
      a_change_cost_ramp=a_change_cost_ramp,
    )

    # Display the current following reference, including comfort headroom and
    # any already-authorized cutout/lane-change relief. This does not select
    # control leads or modify solver obstacles.
    display_margins = comfort_margins + (cost_shifts if mode == 'acc' else 0.0)
    self.desired_distance = displayed_follow_distance(
      self.base_desired_distances, np.array([lead_0_obstacle[0], lead_1_obstacle[0]]),
      x_obstacles[0, :2], self.lead_gap_margins[0] - display_margins[0],
      (radarstate.leadOne.status, radarstate.leadTwo.status),
    )
    self.yref[:,0] = binding_reference(x_targets, preferred, binding, gap_v)
    self.yref[:,1] = x
    self.yref[:,2] = v
    self.yref[:,3] = a
    # The acceleration-change residual is a_ego - prev_a. Its reference
    # spreads a moderate brake without changing the constraints or obstacle selection.
    lead_one_track = lead_one.radarTrackId if lead_one.status and lead_one.radar and lead_one.radarTrackId >= 0 else None
    closing = v_ego - lead_v_0
    self.soft_brake_state.update(
      lead_one_track, reset=reset_state or mode != 'acc', lead_accel=lead_one.aLeadK,
      ego_accel=min(a_ego, measured_a_ego), closing=closing,
      time_gap=lead_one.dRel / max(v_ego, 0.1) if lead_one_track is not None else math.inf,
    )
    self.yref[:,4] = 0.0
    accel_ref_delta = None
    if (mode == 'acc' and not reset_state and lead_gap_enabled and not tesla_stop_active
        and self.max_a >= 0.0 and approach_weight == 0.0 and lead_one_track is not None
        and not self.soft_brake_state.blocked and lead_gaps[0] is not None
        and lead_track_frames and lead_track_frames[0] * self.dt >= soft_brake.SOFT_BRAKE_TRACK_TIME):
      weight = soft_brake.soft_brake_weight(
        distance=lead_one.dRel, ego_speed=v_ego, lead_speed=lead_v_0, lead_accel=lead_one.aLeadK,
        lead_end_speed=lead_xv_0[-1, 1], planned_decel=-np.min(self.prev_a[T_IDXS <= 3.0]),
        gap=lead_gaps[0], stop_distance=stop_distance,
      )
      if weight > 0.0:
        accel_ref_delta = soft_brake.soft_brake_reference_delta(
          T_IDXS, self.prev_a, weight=weight, closing=closing, danger_margin=self.predicted_danger_margin,
        )
        self.yref[:,4] = np.where(binding == 0, accel_ref_delta, 0.0)
    self.yref[:,5] = j
    for i in range(N):
      self.solver.set(i, "yref", self.yref[i])
    self.solver.set(N, "yref", self.yref[N][:COST_E_DIM])

    danger_factor_default = float(self.params[0,5])
    def set_obstacle_params(binding):
      self.params[:,2] = x_obstacles[np.arange(N+1), binding]
      self.params[:,4] = t_follow
      self.params[:,5] = danger_factor_default
      self.params[:,6] = comfort_brake
      for lead_index, gap in enumerate(lead_gaps):
        if gap is not None:
          lead_nodes = binding == lead_index
          self.params[lead_nodes,4] = gap.t_follow
          self.params[lead_nodes,5] = gap.danger_factor
          self.params[lead_nodes,6] = gap.comfort_brake
    self.params[:,3] = np.copy(self.prev_a)
    self.params[:,7] = stop_distance
    set_obstacle_params(binding)

    self.t_follow = lead_t_follow

    self.run()
    if np.any(comfort_reliefs) and self.solution_status == 0 and any(state.relief > 0.0 for state in self.lead_comfort_states):
      # The previous trajectory only admits a trial. Validate the resulting
      # trajectory too, and resolve once without slack if it spends the reserve.
      if lead_gap is None:
        danger_floor = self.params[:,5] * get_safe_obstacle_distance(self.x_sol[:,1], t_follow, comfort_brake, stop_distance)
      else:
        danger_floor = lead_gap.danger_factor * gap_safe.safe_distance(self.x_sol[:,1], lead_gap, stop_distance)
      reserve_ok = all(
        not np.any(comfort_reliefs[:, i]) or comfort_path_has_reserve(
          self.x_sol[:,1], lead_xv[:,0] - self.x_sol[:,0],
          obstacle - self.x_sol[:,0] - danger_floor, T_IDXS, stop_distance)
        for i, (lead_xv, obstacle) in enumerate(((lead_xv_0, lead_0_obstacle), (lead_xv_1, lead_1_obstacle)))
      )
      if not reserve_ok:
        self.lead_gap_margins += comfort_reliefs
        for state in self.lead_comfort_states:
          state.relief = 0.0
        bind_margins += comfort_reliefs if bind_margins is not self.lead_gap_margins else 0.0
        preferred = preferred_targets(self.lead_gap_margins)
        binding = bind_obstacles(x_targets, preferred_targets(bind_margins))
        set_obstacle_params(binding)
        self.yref[:,0] = binding_reference(x_targets, preferred, binding, gap_v)
        if accel_ref_delta is not None:
          self.yref[:,4] = np.where(binding == 0, accel_ref_delta, 0.0)
        for i in range(N):
          self.solver.set(i, "yref", self.yref[i])
        self.solver.set(N, "yref", self.yref[N][:COST_E_DIM])
        self.desired_distance = displayed_follow_distance(
          self.base_desired_distances, np.array([lead_0_obstacle[0], lead_1_obstacle[0]]),
          x_obstacles[0, :2], self.lead_gap_margins[0] - display_margins[0], (radarstate.leadOne.status, radarstate.leadTwo.status))
        trial_times = (self.solve_time, self.time_qp_solution, self.time_linearization, self.time_integrator)
        self.run()
        self.solve_time += trial_times[0]
        self.time_qp_solution += trial_times[1]
        self.time_linearization += trial_times[2]
        self.time_integrator += trial_times[3]
    if mode == 'acc':
      self.update_predicted_danger_margin(radarstate.leadOne, lead_0_obstacle, t_follow, comfort_brake, stop_distance, lead_gap)

    if (np.any(lead_xv_0[FCW_IDXS,0] - self.x_sol[FCW_IDXS,0] < CRASH_DISTANCE) and
            radarstate.leadOne.modelProb > 0.9):
      self.crash_cnt += 1
    else:
      self.crash_cnt = 0

    # Check if it got within lead comfort range
    # TODO This should be done cleaner
    if self.mode == 'blended':
      def lead_constrains(i):
        gap = lead_gaps[i] if mode == 'blended' else None
        distance = (get_safe_obstacle_distance(self.x_sol[:,1], t_follow, comfort_brake, stop_distance) if gap is None
                    else gap_safe.safe_distance(self.x_sol[:,1], gap, stop_distance))
        obstacle = x_obstacles[:, i] if mode == 'blended' else (lead_0_obstacle, lead_1_obstacle)[i]
        return any(obstacle - distance - self.x_sol[:,0] < 0.0)
      if lead_constrains(0):
        self.source = 'lead0'
      if lead_constrains(1) and \
         (lead_1_obstacle[0] - lead_0_obstacle[0]):
        self.source = 'lead1'

  def run(self):
    # t0 = time.monotonic()
    # reset = 0
    for i in range(N+1):
      self.solver.set(i, 'p', self.params[i])
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)

    self.solution_status = self.solver.solve()
    self.solve_time = float(self.solver.get_stats('time_tot')[0])
    self.time_qp_solution = float(self.solver.get_stats('time_qp')[0])
    self.time_linearization = float(self.solver.get_stats('time_lin')[0])
    self.time_integrator = float(self.solver.get_stats('time_sim')[0])

    # qp_iter = self.solver.get_stats('statistics')[-1][-1] # SQP_RTI specific
    # print(f"long_mpc timings: tot {self.solve_time:.2e}, qp {self.time_qp_solution:.2e}, lin {self.time_linearization:.2e}, \
    # integrator {self.time_integrator:.2e}, qp_iter {qp_iter}")
    # res = self.solver.get_residuals()
    # print(f"long_mpc residuals: {res[0]:.2e}, {res[1]:.2e}, {res[2]:.2e}, {res[3]:.2e}")
    # self.solver.print_statistics()

    for i in range(N+1):
      self.x_sol[i] = self.solver.get(i, 'x')
    for i in range(N):
      self.u_sol[i] = self.solver.get(i, 'u')

    self.v_solution = self.x_sol[:,1]
    self.a_solution = self.x_sol[:,2]
    self.j_solution = self.u_sol[:,0]

    self.prev_a = np.interp(T_IDXS + self.dt, T_IDXS, self.a_solution)

    t = time.monotonic()
    if self.solution_status != 0:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning(f"Long mpc reset, solution_status: {self.solution_status}")
      self.reset()
      # reset = 1
    # print(f"long_mpc timings: total internal {self.solve_time:.2e}, external: {(time.monotonic() - t0):.2e} qp {self.time_qp_solution:.2e}, \
    # lin {self.time_linearization:.2e} qp_iter {qp_iter}, reset {reset}")


if __name__ == "__main__":
  ocp = gen_long_ocp()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
  # AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
