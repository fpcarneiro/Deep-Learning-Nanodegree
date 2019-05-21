import numpy as np
from scipy.spatial import distance
from physics_sim import PhysicsSim

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def percent_change(old, new):
    change = np.zeros(old.shape)
    change[old != 0] = (old[old != 0] - new[old != 0])/old[old != 0]
    return change

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        
        self.unique_state_size = np.concatenate(([self.sim.pose] + [self.sim.v] + [self.sim.angular_v])).shape[0]
        
        self.state_size = self.action_repeat * self.unique_state_size
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        # Origin
        self.init_pose = init_pose

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
    
    def get_reward(self, old_pose, old_angular_v, old_v):
        """Uses current pose of sim to return reward."""
        distances = self.sim.pose[:3] - self.target_pos
        distances_tanh = np.tanh(abs(distances))
        velocities_tanh = np.tanh(abs(self.sim.v))
#        distance_euclidean = distance.euclidean(self.sim.pose[:3], self.target_pos)
        right_direction = distances * self.sim.v < 0.0
        wrong_direction = distances * self.sim.v >= 0.0

        reward_right_direction = np.mean(right_direction * np.minimum(distances_tanh, velocities_tanh))
        penalize_wrong_direction = np.mean(wrong_direction * np.maximum(distances_tanh, velocities_tanh))
        
        # Percentual changes in angles and velocities
        euler_change = percent_change(old_angular_v, self.sim.angular_v)
        velocity_change = percent_change(old_v, self.sim.v)
        
        # Abrupt percentual changes in angles? In velocities?
        abrupt_euler_change = euler_change >= 0.2
        abrupt_velocity_change = velocity_change >= 0.2
        
        # Penalty for big changes in angles and velocity in order to encourage smoother movements
        abrupt_euler_change_penalty = np.mean(abrupt_euler_change * np.tanh(abs(euler_change)))
        
        # Apply penalty only after some timesteps
        if self.sim.time >= 1.2:
            abrupt_velocity_change_penalty = np.mean(abrupt_velocity_change * np.tanh(abs(velocity_change)))
        else:
            abrupt_velocity_change_penalty = 0.0
        
        # During the first timesteps give extra points for velocities in axis Z in the right direction
        if self.sim.time <= 1.2:
            extra_z = right_direction[2]
        else:
            extra_z = 0.0
        
        abrupt_penalty = abrupt_euler_change_penalty + abrupt_velocity_change_penalty
        
        extra_point = (np.sum(right_direction) == 3)
        extra_penalty = (np.sum(distances * self.sim.v > 0.0) == 3)
        
        rewards = (reward_right_direction * 0.95) + (extra_point * 0.05) + (extra_z * 0.05)
        penalties = (penalize_wrong_direction * 0.85) + (abrupt_penalty * 0.1) + (extra_penalty * 0.05)
        
        final_reward = 0.01 + rewards - penalties
        final_reward = final_reward / self.action_repeat
        
        return final_reward
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            old_pose = np.copy(self.sim.pose)
            old_angular_v = np.copy(self.sim.angular_v)
            old_v = np.copy(self.sim.v)
            
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(old_pose, old_angular_v, old_v)
            pose_all.append(self.sim.pose)
            pose_all.append(self.sim.v)
            pose_all.append(self.sim.angular_v)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done
    
    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate(([self.sim.pose] + [self.sim.v] + [self.sim.angular_v]) * self.action_repeat) 
        return state
    
    def get_pose_from_state(self, state, i = -1):
        i = (self.action_repeat - 1) if (i >= self.action_repeat or i == -1) else i
        from_idx = self.unique_state_size * i
        to_idx = from_idx + self.sim.pose.shape[0]
        pose = state[from_idx : to_idx]
        return pose