import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

class MarketEnvironment:
    """
    A simulated financial market environment for portfolio optimization.

    This class implements the Heston-Nandi GARCH(1,1) model for a risky asset
    and a risk-free asset, as described in Gollart & Okhrin (2025).
    It is designed to be used with a reinforcement learning agent.
    """
    def __init__(self, config):
        """
        Initializes the environment with all necessary parameters.
        """
        # Market Parameters (HN-GARCH)
        self.lambda_ = config['lambda']
        self.omega = config['omega']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma_hn = config['gamma_hn']

        # Asset Parameters
        self.r = config['r']

        # Investor Preferences
        self.gamma_investor = config['gamma_investor']

        # Simulation Horizon
        self.T = config['T']
        self.N = config['N']
        self.dt = self.T / self.N

        # Initial Conditions
        self.V_0 = config.get('V_0', 1.0)
        self.S_0 = config.get('S_0', 1.0)

        # Pre-calculate unconditional variance
        # h_uncond = (ω + α) / (1 - α * γ_HN^2 - β) - This is for the original HN model with z^2.
        # The paper uses a slightly different model where the variance equation is:
        # h_{t+2} = ω + β*h_{t+1} + α(z_t - γ_HN * sqrt(h_{t+1}))^2
        # The unconditional variance for this is E[h] = (ω + α) / (1 - β - α * γ_HN^2)
        # However, the prompt specifies the formula as h_uncond = (ω + α) / (1 - α*γ_HN^2 - β)
        # Let's stick to the prompt's formula for now, but note the potential discrepancy.
        # After re-reading the user feedback, the formula for h_t+2 is indeed
        # h_{t+2} = ω + β*h_{t+1} + α(z_t - γ_HN * sqrt(h_{t+1}))^2
        # The unconditional variance E[h] is found by setting E[h_t+2]=E[h_t+1]=h and E[(z_t - ...)^2] = 1 + γ_HN^2*h
        # h = ω + β*h + α(1 + γ_HN^2*h) => h(1 - β - α*γ_HN^2) = ω + α => h = (ω + α) / (1 - β - α*γ_HN^2)
        self.h_uncond = (self.omega + self.alpha) / (1 - self.beta - self.alpha * self.gamma_hn**2)

        # Initialize state variables
        self.t = 0
        self.S_t = self.S_0
        self.V_t = self.V_0
        self.h_t_plus_1 = self.h_uncond

    def reset(self, *, seed=None):
        """
        Resets the environment to the initial state for a new episode.
        Returns the initial state observation.
        """
        if seed is not None:
            np.random.seed(seed)

        self.t = 0
        self.S_t = self.S_0
        self.V_t = self.V_0
        self.h_t_plus_1 = self.h_uncond

        return np.array([self.V_t, self.h_t_plus_1, self.t])

    def step(self, action):
        """
        Executes one time step within the environment.
        - Takes an action (pi_t) from the agent.
        - Calculates the next state, reward, and done flag.
        Returns (next_state, reward, done, info).
        """
        # 1. Clip the action (portfolio allocation) to the allowed range [0, 1.5]
        pi_t = np.clip(action, 0, 1.5)

        # 2. Generate the stochastic shock from a standard normal distribution
        z_t = np.random.randn()

        # 3. Calculate the next stock price (S_{t+1}) using Equation 4
        # ln(S_{t+1}/S_t) = r*dt + (λ - 0.5)*h_{t+1}*dt + sqrt(h_{t+1}*dt)*z_t
        log_return = (self.r * self.dt +
                      (self.lambda_ - 0.5) * self.h_t_plus_1 * self.dt +
                      np.sqrt(self.h_t_plus_1 * self.dt) * z_t)
        S_t_plus_1 = self.S_t * np.exp(log_return)

        # 4. Calculate the next portfolio value (V_{t+1}) using Equation 3
        # V_{t+1} = V_t * [π_t * (S_{t+1}/S_t) + (1 - π_t) * exp(r*dt)]
        portfolio_return = pi_t * (S_t_plus_1 / self.S_t) + (1 - pi_t) * np.exp(self.r * self.dt)
        V_t_plus_1 = self.V_t * portfolio_return

        # 5. Calculate the next conditional variance (h_{t+2}) using Equation 5 (Heston-Nandi GARCH)
        # h_{t+2} = ω + β*h_{t+1} + α(z_t - γ_HN * sqrt(h_{t+1}))^2
        h_t_plus_2 = (self.omega + self.beta * self.h_t_plus_1 +
                      self.alpha * (z_t - self.gamma_hn * np.sqrt(self.h_t_plus_1))**2)

        # 6. Increment the internal time step
        self.t += 1

        # 7. Update the state variables for the next step
        self.S_t = S_t_plus_1
        self.V_t = V_t_plus_1
        self.h_t_plus_1 = h_t_plus_2

        # 8. Determine the reward and done flag
        done = self.t == self.N
        reward = 0.0
        if done:
            # If at the terminal step, calculate the CRRA utility as the reward
            if self.gamma_investor == 1.0:
                reward = np.log(self.V_t)
            else:
                reward = (self.V_t**(1 - self.gamma_investor)) / (1 - self.gamma_investor)

        # 9. Construct the next state and return
        next_state = np.array([self.V_t, self.h_t_plus_1, self.t])
        info = {}

        return next_state, reward, done, info

    def run_validation_suite(self):
        """
        Includes a suite of tests to verify the environment's correctness.
        """
        print("--- Running Validation Suite ---")

        # Create a directory for validation plots
        plot_dir = "validation_plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # 1. Numerical Stability Test
        print("\n[1/3] Running Numerical Stability Test...")
        self.reset(seed=42)
        for i in range(10000):
            action = np.random.uniform(0, 1.5)
            state, _, _, _ = self.step(action)
            if np.isnan(state).any() or np.isinf(state).any():
                print(f"!!! Stability test failed at step {i}. State: {state}")
                return
        print(">>> Stability Test Passed: No NaN or Inf values detected in 10,000 steps.")

        # 2. Stochastic Process Validation (Monte Carlo)
        print("\n[2/3] Running Stochastic Process Validation (Monte Carlo)...")
        num_episodes = 10000
        terminal_values = []
        log_returns = []

        for i in range(num_episodes):
            state = self.reset(seed=i)
            done = False
            while not done:
                s_t_before = self.S_t
                state, _, done, _ = self.step(0.6) # Constant policy pi_t = 0.6
                log_returns.append(np.log(self.S_t / s_t_before))

            terminal_values.append(state[0]) # state[0] is V_T

        terminal_values = np.array(terminal_values)

        # Report statistics
        print(">>> Terminal Portfolio Value (V_T) Statistics:")
        print(f"    Mean:      {np.mean(terminal_values):.4f}")
        print(f"    Std. Dev.: {np.std(terminal_values):.4f}")
        print(f"    Skewness:  {skew(terminal_values):.4f}")
        print(f"    Kurtosis:  {kurtosis(terminal_values):.4f}")

        # Plot histogram of log-returns
        plt.figure(figsize=(10, 6))
        plt.hist(log_returns, bins=100, density=True, alpha=0.7, label='Log-Returns')
        plt.title('Distribution of Stock Log-Returns')
        plt.xlabel('Log-Return ln(S_{t+1}/S_t)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        log_returns_path = os.path.join(plot_dir, "stock_log_returns_distribution.png")
        plt.savefig(log_returns_path)
        plt.close()
        print(f">>> Saved stock log-returns plot to {log_returns_path}")

        # Plot histogram of terminal portfolio values
        plt.figure(figsize=(10, 6))
        plt.hist(terminal_values, bins=100, density=True, alpha=0.7)
        plt.title('Distribution of Terminal Portfolio Values (V_T)')
        plt.xlabel('Terminal Value')
        plt.ylabel('Density')
        plt.grid(True)
        terminal_wealth_path = os.path.join(plot_dir, "terminal_wealth_distribution.png")
        plt.savefig(terminal_wealth_path)
        plt.close()
        print(f">>> Saved terminal wealth plot to {terminal_wealth_path}")

        # 3. Path-Level Sanity Check
        print("\n[3/3] Running Path-Level Sanity Check...")
        self.reset(seed=123)
        path_S = [self.S_0]
        path_V = [self.V_0]
        path_h = [self.h_uncond]

        done = False
        while not done:
            state, _, done, _ = self.step(0.6) # Use a constant policy for a clear path
            path_V.append(state[0])
            path_h.append(state[1])
            path_S.append(self.S_t) # self.S_t was updated in the step

        # We need to plot h_t, not h_{t+1}. The state contains h_{t+2}, so we need to be careful.
        # The h path should be h_1, h_2, ..., h_{N+1}
        # h_1 is h_uncond. After step 0, we calculate h_2. The state is [V_1, h_2, 1].
        # So path_h correctly stores h_1, h_2, ...

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        time_steps = np.arange(self.N + 1)

        ax1.plot(time_steps, path_S)
        ax1.set_title('Stock Price (S_t)')
        ax1.set_ylabel('Price')
        ax1.grid(True)

        ax2.plot(time_steps, path_V)
        ax2.set_title('Portfolio Value (V_t)')
        ax2.set_ylabel('Value')
        ax2.grid(True)

        ax3.plot(time_steps, path_h)
        ax3.set_title('Conditional Variance (h_t)')
        ax3.set_xlabel('Time Step (t)')
        ax3.set_ylabel('Variance')
        ax3.grid(True)

        plt.suptitle('Example Simulation Path (pi_t = 0.6)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        simulation_path = os.path.join(plot_dir, "simulation_path_example.png")
        plt.savefig(simulation_path)
        plt.close()
        print(f">>> Saved simulation path plot to {simulation_path}")

        print("\n--- Validation Suite Finished ---")

if __name__ == '__main__':
    # Configuration dictionary with parameters from the paper
    config = {
        # Market Parameters
        'r': 0.02,          # Annual risk-free rate
        'lambda': 0.3,      # Market price of risk
        'omega': 1.5e-6,    # GARCH constant
        'alpha': 1.5e-6,    # GARCH ARCH term
        'beta': 0.1,        # GARCH GARCH term
        'gamma_hn': 300,    # GARCH leverage term

        # Investor/Horizon
        'gamma_investor': 5.0, # Investor's CRRA coefficient
        'T': 1.0,           # Total investment period in years
        'N': 52,            # Number of rebalancing periods (weekly)

        # Initial Conditions
        'V_0': 1.0,         # Initial portfolio value
        'S_0': 1.0          # Initial stock price
    }

    # Instantiate the environment
    env = MarketEnvironment(config)

    # Run the validation suite
    env.run_validation_suite()