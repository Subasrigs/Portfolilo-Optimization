import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Assume market_env.py is in the same directory
from market_env import MarketEnvironment

# Configuration dictionary for DDPG
DDPG_CONFIG = {
    # RL Parameters
    'gamma': 0.99,
    'tau': 0.001,
    'actor_lr': 1e-4,
    'critic_lr': 1e-3,

    # Replay Buffer
    'buffer_size': 1000000,
    'batch_size': 64,

    # Training Run
    'total_episodes': 2000,
    'evaluation_interval': 20,

    # L2 Regularization
    'critic_weight_decay': 1e-5,

    # Ornstein-Uhlenbeck Noise
    'ou_mu': 0.0,
    'ou_theta': 0.15,
    'ou_sigma': 0.2,
}

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu.copy()

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

class Actor(nn.Module):
    """
    Actor Network (Policy)
    Maps states to actions.
    """
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    """
    Critic Network (Q-Value)
    Maps state-action pairs to Q-values.
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU()
        )
        self.combined_layer = nn.Sequential(
            nn.Linear(256 + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        state_out = self.state_layer(state)
        combined = torch.cat([state_out, action], 1)
        q_value = self.combined_layer(combined)
        return q_value

class ReplayBuffer:
    """
    Replay Buffer for off-policy learning.
    Stores transitions and provides random mini-batches.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    """
    DDPG Agent that manages the Actor, Critic, and their target networks.
    """
    def __init__(self, state_dim, action_dim, action_high):
        self.actor = Actor(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        # Copy initial weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=DDPG_CONFIG['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=DDPG_CONFIG['critic_lr'], weight_decay=DDPG_CONFIG['critic_weight_decay'])

        self.action_high = action_high
        self.noise_process = OUNoise(action_dim, seed=0, mu=DDPG_CONFIG['ou_mu'], theta=DDPG_CONFIG['ou_theta'], sigma=DDPG_CONFIG['ou_sigma'])

    def select_action(self, state, training=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]

        # Scale tanh output from [-1, 1] to [0, 1.5]
        scaled_action = (self.action_high / 2.0) * (action + 1.0)

        if training:
            noise = self.noise_process.sample()
            scaled_action = np.clip(scaled_action + noise, 0, self.action_high)

        return scaled_action

    def update(self, replay_buffer):
        if len(replay_buffer) < DDPG_CONFIG['batch_size']:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(DDPG_CONFIG['batch_size'])

        # --- Update Critic ---
        # Get next actions from target actor
        next_actions = self.target_actor(next_states)
        # Get Q-values from target critic
        target_q_values = self.target_critic(next_states, next_actions)
        # Calculate target Q-value
        y_i = rewards + DDPG_CONFIG['gamma'] * (1 - dones) * target_q_values

        # Get current Q-values
        current_q_values = self.critic(states, actions)

        # Critic loss
        critic_loss = nn.MSELoss()(current_q_values, y_i.detach())

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft update target networks ---
        self.soft_update(self.target_actor, self.actor, DDPG_CONFIG['tau'])
        self.soft_update(self.target_critic, self.critic, DDPG_CONFIG['tau'])

    def soft_update(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

if __name__ == '__main__':
    # Initialize environment and agent
    env = MarketEnvironment()
    agent = DDPGAgent(env.state_dim, env.action_dim, env.action_space_high)
    replay_buffer = ReplayBuffer(DDPG_CONFIG['buffer_size'])

    evaluation_rewards = []

    for episode in range(DDPG_CONFIG['total_episodes']):
        state = env.reset()
        agent.noise_process.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            agent.update(replay_buffer)

            state = next_state
            episode_reward += reward

        print(f"Episode {episode + 1}/{DDPG_CONFIG['total_episodes']}: Reward = {episode_reward}")

        # Periodic evaluation
        if (episode + 1) % DDPG_CONFIG['evaluation_interval'] == 0:
            total_eval_reward = 0
            for _ in range(100):
                state = env.reset(seed=42) # Fixed seed for evaluation
                eval_done = False
                eval_episode_reward = 0
                while not eval_done:
                    action = agent.select_action(state, training=False)
                    next_state, reward, eval_done, _ = env.step(action)
                    eval_episode_reward += reward
                    state = next_state
                total_eval_reward += eval_episode_reward

            avg_eval_reward = total_eval_reward / 100
            evaluation_rewards.append(avg_eval_reward)
            print(f"Evaluation at Episode {episode + 1}: Average Reward = {avg_eval_reward}")

    env.close()

    # --- Generate and Save Plots ---

    # 1. Convergence Plot
    plt.figure(figsize=(10, 6))
    episodes = np.arange(DDPG_CONFIG['evaluation_interval'], DDPG_CONFIG['total_episodes'] + 1, DDPG_CONFIG['evaluation_interval'])
    plt.plot(episodes, evaluation_rewards, label='Smoothed Average Terminal Utility')
    plt.title('DDPG Convergence Analysis')
    plt.xlabel('Training Episode')
    plt.ylabel('Average Terminal Utility (U(V_T))')
    plt.legend()
    plt.grid(True)
    plt.savefig('ddpg_convergence_plot.png')
    plt.close()

    # 2. Learned Policy Plot
    policy_actions = []
    state = env.reset(seed=42) # Use a fixed seed for consistency
    done = False
    while not done:
        action = agent.select_action(state, training=False)
        policy_actions.append(action[0])
        state, _, done, _ = env.step(action)

    plt.figure(figsize=(10, 6))
    plt.plot(policy_actions, label='Learned Investment Proportion ($\\pi_t$)')
    plt.title('Learned Policy Profile')
    plt.xlabel('Time Step (t)')
    plt.ylabel('Investment Proportion ($\\pi_t$)')
    plt.legend()
    plt.grid(True)
    plt.savefig('ddpg_policy_profile.png')
    plt.close()

    print("Training complete. Plots saved as ddpg_convergence_plot.png and ddpg_policy_profile.png")
