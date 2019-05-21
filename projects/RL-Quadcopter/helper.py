import random
from collections import namedtuple, deque
from keras import layers, models, optimizers, regularizers
from keras import backend as K
import numpy as np
import copy
import pandas as pd

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

import plotly.offline as py
import plotly.graph_objs as go

class Log:
    """Class to store the logs of the episodes for later analysis."""
    
    def __init__(self):
        """Initialize a Log object.
        Params
        ======

        """
        self.labels_timestep = ['episode', 'time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'distance_to_target', 'distance_traveled', 'total_distance_traveled', 'reached_target', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4', 'reward', 'cumulative_reward', 'done']
        
        self.labels_episodes = ['episode', 'timesteps', 'cumulative_reward', 'total_distance_traveled', 'reached_target', 'closest_distance', 'closest_x', 'closest_y', 'closest_z']
        
        self.time_steps = {x : [] for x in self.labels_timestep}
        self.episodes = {x : [] for x in self.labels_episodes}
    
    def add_timestep(self, episode, time, pose, distance_to_target, distance_traveled, total_distance_traveled, reached_target, v, angular_v, rotor_speeds, reward, cum_reward, done):
        to_write = [episode] + [time] + list(pose) + [distance_to_target] + [distance_traveled] + [total_distance_traveled] + [reached_target] + list(v) + list(angular_v) + list(rotor_speeds) + [reward] + [cum_reward] + [done]
        for ii in range(len(self.labels_timestep)):
            self.time_steps[self.labels_timestep[ii]].append(to_write[ii])
    
    def add_episode(self, episode, timesteps, cum_reward, total_distance_traveled, reached_target, closest_distance, closest_pose):
        to_write = [episode] + [timesteps] + [cum_reward] + [total_distance_traveled] + [reached_target] + [closest_distance] + list(closest_pose)
        for ii in range(len(self.labels_episodes)):
            self.episodes[self.labels_episodes[ii]].append(to_write[ii])
    
    def time_steps_as_pandas(self, episode = None):
        df = pd.DataFrame.from_dict(self.time_steps)
        return df if episode is None else df[df.episode == episode]
    
    def episodes_as_pandas(self):
        return(pd.DataFrame.from_dict(self.episodes))
    
    def best_episode(self):
        df = self.episodes_as_pandas()
        return df[df.cumulative_reward == max(df.cumulative_reward)].episode.values[0]
    
    def get_longest(self):
        df = self.episodes_as_pandas()
        return df[df.timesteps == max(df.timesteps)]

    def plotflight3d(self, episodes = None, origin = (0,0,0), target = (0,0,0)):        
        mpl.rcParams['legend.fontsize'] = 10

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        palette = plt.get_cmap('Set1')
        plt.style.use('seaborn-darkgrid')
        # Prepare arrays x, y, z
        df = self.time_steps_as_pandas()
        df = df if episodes is None else df[df.episode.isin(episodes)]
        
        for e in df.episode.unique():
            z = df[df.episode == e].z.values
            x = df[df.episode == e].x.values
            y = df[df.episode == e].y.values
            ax.plot(x, y, z, label=e, marker='.')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Annotate flight's origin and final points as well as the target
        #origin = df.head(1).loc[:,["x","y","z"]].values[0]
        #destination = df.tail(1).loc[:,["x","y","z"]].values[0]
        #ax.text(*origin, "origin", None)
        #ax.text(*destination, "destination", None)
        ax.text(*target, "target", None)
        
        ax.legend()
        plt.show()
    
    def plotflight3d_2(self, episodes = None, origin = (0,0,0), target = (0,0,0), save = True, filename = "plot.html"):
        def mark_target(radius = 5.0, target = (0,0,0)):
            trace_target = go.Scatter3d(x=target[0], y=target[1], z=target[2],
                text = "target",
                mode = 'markers',
                marker = dict(
                    sizemode = 'diameter',
                    size=radius,
                    colorscale = 'Viridis',
                    line=dict(color='rgb(140, 140, 170)')
                )
            )
            return trace_target
        
        py.init_notebook_mode(connected=True)
        
        df = self.time_steps_as_pandas()
        df = df if episodes is None else df[df.episode.isin(episodes)]
        
        data = []

        for e in df.episode.unique():
            name = "{}".format(e)
            color = e
            z = np.round(df[df.episode == e].z.values, 2)
            x = np.round(df[df.episode == e].x.values, 2)
            y = np.round(df[df.episode == e].y.values, 2)
            dt = ["\u0394d: {:.2f}\n\u03A3: {:.2f}".format(d, r) for d, r in zip(np.round(df[df.episode == e].distance_to_target.values, 2), np.round(df[df.episode == e].cumulative_reward.values, 2))]
            
            trace = dict(
                name = name,
                x = x, y = y, z = z,
                type = "scatter3d",    
                mode = 'lines+markers',
                text = dt,
                marker = dict( size=2, opacity = 0.7) )
        
            data.append( trace )
        
        trace_target = mark_target(radius = 50.0, target = target)
        data.append( trace_target )
        
        origin_annotation = dict(showarrow = False, x=origin[0], y=origin[1], z=origin[2], text = "origin",)
        target_annotation = dict(showarrow = False, x=target[0], y=target[1], z=target[2], text = "target",)
        
        layout = go.Layout(
            title='Agent Flights', 
            scene = dict(aspectratio = dict(x = 1, y = 1, z = 1),
                         camera = dict(center = dict(x = 0, y = 0, z = 0), eye = dict(x = 1.96903462608, y = -1.09022831971, z = 0.405345349304), up = dict(x = 0, y = 0, z = 1)),
                         annotations = [origin_annotation, target_annotation]),)
        
        fig = go.Figure(data=data, layout=layout)
        py.iplot(fig)
        if save:
            py.plot(fig, filename=filename)
    
    def plot2d(self, episode = None, what = "positions"):
        episode = self.best_episode() if episode is None else episode
        
        options = {"positions": ['x', 'y', 'z'], "velocity": ["x_velocity", "y_velocity", "z_velocity"], 
                   "angles": ["phi", "theta", "psi"], "angular_velocity": ["phi_velocity", "theta_velocity", "psi_velocity"],
                  "rotor_speed": ["rotor_speed1", "rotor_speed2", "rotor_speed3", "rotor_speed4"]}
        
        df = self.time_steps_as_pandas(episode).loc[:, ['time'] + options[what]]
        df.plot(x='time', y = options[what], marker='.', title = what)
   
    def plot_rewards(self):
        def running_mean(x, N):
            cumsum = np.cumsum(np.insert(x, 0, 0)) 
            return (cumsum[N:] - cumsum[:-N]) / N
        
        df = self.episodes_as_pandas().loc[:, ["episode", "cumulative_reward"]]
        eps, rews = np.array(df.values).T
        smoothed_rews = running_mean(rews, 10)
        plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
        plt.plot(eps, rews, color='grey', alpha=0.3)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, batch_normalization = False, lr=0.0001):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.batch_normalization = batch_normalization
        self.lr = lr

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')
       
        # Add hidden layers
        net = layers.Dense(units=64)(states)
        if self.batch_normalization:
            net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        
        net = layers.Dense(units=512)(net)
        if self.batch_normalization:
            net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        
        net = layers.Dense(units=384)(net)
        if self.batch_normalization:
            net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=self.lr)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, batch_normalization = False, lr=0.00001):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.batch_normalization = batch_normalization
        self.lr = lr

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        
        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=64)(states)
        if self.batch_normalization:
            net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation("relu")(net_states)
        
        net_states = layers.Dense(units=384)(net_states)
        if self.batch_normalization:
            net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation("relu")(net_states)
        
        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=64)(actions)
        if self.batch_normalization:
            net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation("relu")(net_actions)
        
        net_actions = layers.Dense(units=384)(net_actions)
        if self.batch_normalization:
            net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation("relu")(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.lr)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
        
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state