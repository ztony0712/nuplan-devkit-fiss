import math
import time
import logging
from typing import List, Type, Optional, Tuple
from queue import PriorityQueue

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.planner.project2.bfs_router import BFSRouter
from nuplan.planning.simulation.planner.project2.reference_line_provider import ReferenceLineProvider
from nuplan.planning.simulation.planner.project2.simple_predictor import SimplePredictor
from nuplan.planning.simulation.planner.project2.abstract_predictor import AbstractPredictor

from nuplan.planning.simulation.planner.project2.merge_path_speed import transform_path_planning, cal_dynamic_state, cal_pose
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects

from nuplan.planning.simulation.planner.fiss_plus_planner.planners.fiss_plus_planner import FissPlusPlanner, FissPlusPlannerSettings, Vehicle, nuPlan_Vehicle
from nuplan.planning.simulation.planner.fiss_plus_planner.planners.common.scenario.frenet import FrenetState, State, FrenetTrajectory
from nuplan.planning.simulation.planner.fiss_plus_planner.planners.frenet_optimal_planner import Stats

logger = logging.getLogger(__name__)


class MyFissPlusPlanner(AbstractPlanner, FissPlusPlanner):
    """
    Fiss Plus Planner.
    """

    def __init__(
            self,
            horizon_seconds: float,
            sampling_time: float,
            max_velocity: float = 5.0,
    ):
        super().__init__(FissPlusPlannerSettings(), nuPlan_Vehicle(get_pacifica_parameters())) # ?：如何使用这些settings
        """
        Constructor for SimplePlanner.
        :param horizon_seconds: [s] time horizon being run.
        :param sampling_time: [s] sampling timestep.
        :param max_velocity: [m/s] ego max velocity.
        """
        self.horizon_time = TimePoint(int(horizon_seconds * 1e6))
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.max_velocity = max_velocity
        self.vehicle = get_pacifica_parameters()

        self._router: Optional[BFSRouter] = None
        self._predictor: AbstractPredictor = None
        self._reference_path_provider: Optional[ReferenceLineProvider] = None
        self._routing_complete = False

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._router = BFSRouter(initialization.map_api)
        self._router._initialize_route_plan(initialization.route_roadblock_ids)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore



    # TODO: 2. Please implement your own trajectory planning.
    def planning(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list, time_step_now: int = 0) -> FrenetTrajectory:
        
        t_start = time.time()
        
        # Reset values for each planning cycle
        self.stats = Stats() 
        self.settings.highest_speed = max_target_speed
        self.start_state = frenet_state
        self.frontier_idxs = PriorityQueue()
        self.candidate_trajs = PriorityQueue()
        self.refined_trajs = PriorityQueue()
        self.best_traj = None
        
        # Sample all the end states in 3 dimension, [d, v, t] and form the 3d traj canndidate array
        self.trajs_3d = self.sample_end_frenet_states()
        self.sizes = np.array([len(self.trajs_3d), len(self.trajs_3d[0]), len(self.trajs_3d[0][0])])
        
        best_idx = None
        best_traj_found = False
        
        while not best_traj_found:
            self.stats.num_iter += 1
            
            ############################## Initial Guess ##############################
            
            if self.candidate_trajs.empty():
                best_idx = self.find_initial_guess()
                if best_idx is None:
                    break
                else:
                    best_idx = self.candidate_trajs.queue[0][1] # peek the index of the most likely candidate
                    
                i = 0
                converged = False
                while not converged:
                    i += 1
                    is_minimum, best_idx = self.explore_neighbors(best_idx)
                    if self.frontier_idxs.empty():
                        converged = True
                    else:
                        _, best_idx = self.frontier_idxs.get()
                        
            ############################## Validation ##############################
            
            if not self.candidate_trajs.empty():
                _, idx = self.candidate_trajs.get()
                candidate = self.trajs_3d[idx[0]][idx[1]][idx[2]]
                self.stats.num_trajs_validated += 1
                # Convert to global coordinates
                candidate = self.calc_global_paths(candidate)
                # Check for constraints
                passed_candidate = self.check_constraints(candidate)
                if passed_candidate:
                    # Check for collisions
                    safe_candidate = self.check_collisions(candidate, obstacles, time_step_now)
                    self.stats.num_collison_checks += 1
                    if safe_candidate:
                        best_traj_found = True
                        self.best_traj = safe_candidate[0] ## ?： 为什么是0: lowest cost ranks first
                        self.prev_best_idx = self.best_traj.idx
                        break
                    else:
                        continue
                else:
                    continue
            else:
                break
            
        if best_traj_found and self.settings.refine_trajectory:
            time_spent = time.time() - t_start
            time_left = self.settings.time_limit - time_spent
            
            if not self.settings.has_time_limit or time_left > 0.0:
                refined_traj = self.refine_solution(self.best_traj, time_left, obstacles, time_step_now)
                if refined_traj is not None:
                    self.best_traj = refined_traj
                         
        return self.best_traj
    
    def do_planning(self, ego_state, current_input, num_samples: tuple):
        
        min_speed = 0.0
        max_speed = 13.5
  
        # Obstacle settings: obtain the future positions of the obstacles
        obstacles_all = current_input.tracked_objects.tracked_objects 
        #current_input.history.observations[0].tracked_objects.tracked_objects[0].center: StateSE2(x=331366.1206990134, y=4690974.712417038, heading=2.565304996102283)
        obstacle_positions = []
        final_time_step = 100 # TODO： Simulation的时间步，nuplan中不需要，考虑去掉
        """
            fiss plus planner中并没有prediction，所以需要记录每个时间步下的障碍物位置，当做完美的prediction。
            另外由于其也没有nuplan中的simulation，所以需要自己设定一个final_time_step，记录每一个时间步下planner output，
            最后把这些结果一帧一帧地画出来，起到一个simulation的效果。
            
        """
        for t_step in range(final_time_step):
            frame_positions = []
            for obstacle in obstacles_all:
                if obstacle.state_at_time(t_step) is not None: # ?： t_step 是simulation的时间步
                    frame_positions.append(obstacle.state_at_time(t_step).position)
        obstacle_positions.append(frame_positions)
        
        # Initialize local planner
        vehicle = nuPlan_Vehicle(self.vehicle)   
        num_width, num_speed, num_t = num_samples
        planner_settings = FissPlusPlannerSettings(num_width, num_speed, num_t)       
        planner = FissPlusPlanner(planner_settings, vehicle)  
        _, ref_ego_lane_pts = planner.generate_frenet_frame(self._reference_path_provider) # ?：注意reference line 的格式
        
        # Initial state
        start_state = State(t=ego_state.time_point.time_s, x=ego_state.center.x, y=ego_state.center.y, yaw=ego_state.center.heading, v=ego_state.dynamic_car_state.speed, a=ego_state.dynamic_car_state.acceleration)
        current_frenet_state = FrenetState()
        current_frenet_state.from_state(start_state, ref_ego_lane_pts)
        
        processing_time = 0
        num_cycles = 0
        stats = Stats()
        
        for i in range(final_time_step):
            num_cycles += 1
            
            # Plan!
            start_time = time.time()
            best_traj_ego = self.planning(current_frenet_state, max_speed, obstacles_all, i)
            end_time = time.time()
            processing_time += (end_time - start_time)
            stats += planner.stats
            
            if best_traj_ego is None:
                break
            next_step_idx = 1
            current_state = best_traj_ego.state_at_time_step(next_step_idx) # ?：是否有这个方法,之后如何把它转成nuplan 需要的trajectory
            current_frenet_state = best_traj_ego.frenet_state_at_time_step(next_step_idx)
            
            avg_processing_time = processing_time / num_cycles
            stats.average(num_cycles)
        
            # TODO: convert the current state to nuplan trajectory
        return avg_processing_time, current_state, stats
                    
        
        
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Implement a trajectory that goes straight.
        Inherited, see superclass.
        """

        # 1. Routing
        ego_state, observations = current_input.history.current_state
        if not self._routing_complete: 
            self._router._initialize_ego_path(ego_state, self.max_velocity)
            self._routing_complete = True

        # 2. Generate reference line
        self._reference_path_provider = ReferenceLineProvider(self._router)
        self._reference_path_provider._reference_line_generate(ego_state)

        # 3. Objects prediction
        # self._predictor = SimplePredictor(ego_state, observations, self.horizon_time.time_s, self.sampling_time.time_s)
        # objects = self._predictor.predict()

        # 4. Planning
        trajectory: List[EgoState] = self.do_planning(ego_state, current_input, (5, 5, 5))

        return InterpolatedTrajectory(trajectory)
    
   