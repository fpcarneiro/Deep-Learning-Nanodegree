# TODO: your agent here!
import numpy as np
from tqdm import tqdm, trange
from scipy.spatial import distance
import sys
from task import Task
from helper import Log, ReplayBuffer, Actor, Critic, OUNoise
        
class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task, episodes = 1000, buffer_size = 100000, batch_size = 64, gamma = 0.99, tau = 0.01, exploration_mu = 0, exploration_theta = 0.15, exploration_sigma = 0.2, batch_normalization = False):
        self.episodes = episodes
        self.log = Log()
        
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.batch_normalization = batch_normalization

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.batch_normalization)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, self.batch_normalization)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, self.batch_normalization)
        self.critic_target = Critic(self.state_size, self.action_size, self.batch_normalization)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = exploration_mu
        self.exploration_theta = exploration_theta
        self.exploration_sigma = exploration_sigma
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # for soft update of target parameters
        
        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.shortest_dist_target_at_end_episode = np.inf
        self.longest_trip = -np.inf
        self.radius = 1.0

    def reset_episode(self):
        self.total_reward = 0.0
        self.total_distance = 0.0
        self.count = 0
        
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.distance_to_target = distance.euclidean(self.task.sim.pose[:3], self.task.target_pos)
        
        self.closest_pose = self.task.sim.pose[:3]
        self.closest_distance = self.distance_to_target
        self.reached_target_at_all = (self.distance_to_target <= self.radius)
        return state
    
    def step(self, action, reward, next_state, done):
        self.total_reward += reward
        self.count += 1
        
        previous_pose = self.task.get_pose_from_state(self.last_state)
        self.distance_traveled = distance.euclidean(previous_pose[:3], self.task.sim.pose[:3])
        self.distance_to_target = distance.euclidean(self.task.sim.pose[:3], self.task.target_pos)
        self.total_distance += self.distance_traveled
        
        if self.distance_to_target < self.closest_distance:
            self.closest_distance = self.distance_to_target
            self.closest_pose = self.task.sim.pose[:3]
        
        self.reached_target = (self.distance_to_target <= self.radius)
        
        if not self.reached_target_at_all:
            self.reached_target_at_all = self.reached_target
        
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state
        
        if done:
            if self.total_reward > self.best_score:
                self.best_score = self.total_reward
            if self.distance_to_target < self.shortest_dist_target_at_end_episode:
                self.shortest_dist_target_at_end_episode = self.distance_to_target
            if self.total_distance > self.longest_trip:
                self.longest_trip = self.total_distance

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        
        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        
    def train(self, verbose = False):
        my_range = trange(1, self.episodes+1, desc='Episodes', leave=True)
        for i_episode in my_range:
            state = self.reset_episode()
            self.log.add_timestep(i_episode, self.task.sim.time, self.task.sim.pose, 
                                  self.distance_to_target, 0.0, 0.0, False, self.task.sim.v, 
                                  self.task.sim.angular_v, [0.,0.,0.,0.], reward=0, cum_reward=0, done=False)
            
            while True:                
                action = self.act(state)
                next_state, reward, done = self.task.step(action)
                self.step(action, reward, next_state, done)
                self.log.add_timestep(i_episode, self.task.sim.time, self.task.sim.pose, self.distance_to_target, 
                                      self.distance_traveled, self.total_distance, self.reached_target, 
                                      self.task.sim.v, self.task.sim.angular_v, action, reward, self.total_reward, done)
                state = next_state
                if done:
                    self.log.add_episode(i_episode, self.count, self.total_reward, 
                                         self.total_distance, self.reached_target_at_all, 
                                         self.closest_distance, self.closest_pose)
                    
                    description = "{:4d}: Steps= {:3d}; score= {:.2f}(best= {:.2f}); Dist target (final={:.2f}, best={:.2f}); Trip= {:.2f}".format(i_episode, self.count, self.total_reward, self.best_score, self.distance_to_target, self.closest_distance, self.total_distance)
                    my_range.set_description(description)
                    my_range.refresh()
                    
                    if verbose:
                        print("\nEp = {:4d}, Score = {:.4f} (Best = {:.4f}) \nMem Size = {:4d} Timesteps = {:4d} \nFinal Distance Target = {:.2f} Trip Distance = {:.2f} Closest = {:.2f}".format(i_episode, self.total_reward, 
                        self.best_score, len(self.memory), self.count, self.distance_to_target, 
                        self.total_distance, self.closest_distance), end="")
                    break
            
            sys.stdout.flush()
    
    def plotflight3d(self, episode = None):
        self.log.plotflight3d(episode = episode, target = self.task.target_pos)
      
    def plotflight3d_2(self, episodes = None, save = True, filename = "my_plot.html"):
        self.log.plotflight3d_2(episodes = episodes, origin = self.task.init_pose, target = self.task.target_pos, save=save, filename=filename)
    
    def get_nsmallest(self, column="closest_distance", n = 10):
        return self.log.episodes_as_pandas().nsmallest(n, column)
    
    def get_nlargest(self, column="cumulative_reward", n = 10):
        return self.log.episodes_as_pandas().nlargest(n, column)
        
    def plot_nsmallest_3d(self, column="closest_distance", n = 10, save = True, filename = "plot_nsmallest_3d.html"):
        episodes = self.get_nsmallest(column=column, n = n).episode.tolist()
        self.log.plotflight3d_2(episodes = episodes, origin = self.task.init_pose, 
                                target = self.task.target_pos, save=save, filename=filename)
        
    def plot_nlargest_3d(self, column="cumulative_reward", n = 10, save = True, filename = "plot_nlargest_3d.html"):
        episodes = self.get_nlargest(column=column, n = n).episode.tolist()
        self.log.plotflight3d_2(episodes = episodes, origin = self.task.init_pose, 
                                target = self.task.target_pos, save=save, filename=filename)
        
    def plot_rewards(self):
        self.log.plot_rewards()