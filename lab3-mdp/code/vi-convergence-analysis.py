import numpy as np
import matplotlib.pyplot as plt
import gym
import time
import heapq
from tqdm import tqdm
import random
import seaborn as sns
import collections

# Set seeds for reproducibility
# RANDOM_SEEDS = [42, 123, 256, 789, 101]
RANDOM_SEEDS = [int(time.time()) % 1000, int(time.time()) % 1000 + 1, int(time.time()) % 1000 + 2,
                  int(time.time()) % 1000 + 3, int(time.time()) % 1000 + 4]

def compute_value_iteration(env, gamma=0.9, epsilon=1e-6, max_iterations=50000, verbose=True):
    """
    Standard Value Iteration.
    Returns:
    - optimal value function
    - convergence history
    - number of iterations
    """
    nS = env.observation_space.n
    nA = env.action_space.n
    
    # Initialize value function
    V = np.zeros(nS)
    
    iteration = 0
    converged = False
    
    if verbose:
        print("Starting Standard Value Iteration...")
    
    while iteration < max_iterations and not converged:
        # Create a copy of the value function to use for updates
        # This is key for standard VI - all updates based on previous iteration
        V_prev = V.copy()
        
        delta = 0
        
        # Update each state
        for s in range(nS):
            # Calculate Q values for all actions using previous value function
            q_values = np.zeros(nA)
            for a in range(nA):
                for prob, next_state, reward, terminated in env.P[s][a]:
                    if terminated:
                        q_values[a] += prob * reward
                    else:
                        q_values[a] += prob * (reward + gamma * V_prev[next_state])
            
            # Update value function with best action
            V[s] = np.max(q_values)
            
            # Track maximum change
            delta = max(delta, abs(V_prev[s] - V[s]))
            
            # Count this as one iteration
            iteration += 1
        
        # Check for convergence
        if delta < epsilon:
            converged = True
    
    if verbose:
        if converged:
            print(f"Value Iteration converged after {iteration} updates.")
        else:
            print(f"Value Iteration reached max iterations ({max_iterations}).")
    
    return V, iteration

def compute_gauss_seidel_vi(env, V_star, gamma=0.9, epsilon=1e-6, max_iterations=50000, verbose=True):
    """
    Gauss-Seidel Value Iteration.
    Returns:
    - value function
    - convergence history (V - V*)
    - number of iterations
    """
    nS = env.observation_space.n
    nA = env.action_space.n
    
    # Initialize value function
    V = np.zeros(nS)
    V_diff_history = []  # Track ||V - V*||
    
    # Record initial error
    v_diff_norm = np.linalg.norm(V - V_star)
    V_diff_history.append((0, v_diff_norm))
    
    iteration = 0
    converged = False
    
    if verbose:
        print("Starting Gauss-Seidel Value Iteration...")
    
    while iteration < max_iterations and not converged:
        delta = 0
        
        # Update each state using the most recent values
        for s in range(nS):
            old_v = V[s]
            
            # Calculate Q values for all actions
            # KEY DIFFERENCE: Uses current V directly (in-place updates)
            # rather than using V_prev like in standard VI
            q_values = np.zeros(nA)
            for a in range(nA):
                for prob, next_state, reward, terminated in env.P[s][a]:
                    if terminated:
                        q_values[a] += prob * reward
                    else:
                        q_values[a] += prob * (reward + gamma * V[next_state])
            
            # Update value function with best action (in-place update)
            V[s] = np.max(q_values)
            
            # Track maximum change
            delta = max(delta, abs(old_v - V[s]))
            
            # Count this as one iteration
            iteration += 1
            
            # Record error norm at EVERY iteration
            v_diff_norm = np.linalg.norm(V - V_star)
            V_diff_history.append((iteration, v_diff_norm))
        
        # Check for convergence
        if delta < epsilon:
            converged = True
    
    if verbose:
        if converged:
            print(f"Gauss-Seidel VI converged after {iteration} updates.")
        else:
            print(f"Gauss-Seidel VI reached max iterations ({max_iterations}).")
    
    return V, V_diff_history, iteration

def compute_prioritized_sweeping_vi(env, V_star, gamma=0.9, epsilon=1e-6, max_iterations=50000, verbose=True):
    """
    Prioritized Sweeping Value Iteration.
    Returns:
    - value function
    - convergence history (V - V*)
    - number of iterations
    """
    nS = env.observation_space.n
    nA = env.action_space.n
    
    # Initialize value function
    V = np.zeros(nS)
    
    # For tracking state priorities more efficiently
    # We'll use a dictionary to track which states are in the queue
    state_priorities = {}  # Maps state -> priority
    pq = []  # Priority queue [(priority, counter, state)]
    
    # Use a counter to ensure unique entries in heapq when priorities are equal
    counter = 0
    
    # Predecessors cache for efficient backups
    predecessors = collections.defaultdict(set)
    for s in range(nS):
        for a in range(nA):
            for prob, next_state, _, _ in env.P[s][a]:
                if prob > 0:
                    predecessors[next_state].add(s)
    
    # Initialize priority queue with all states
    for s in range(nS):
        q_values = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, terminated in env.P[s][a]:
                if terminated:
                    q_values[a] += prob * reward
                else:
                    q_values[a] += prob * (reward + gamma * V[next_state])
        
        priority = abs(np.max(q_values) - V[s])
        if priority > epsilon:
            # Add to priority queue with unique counter
            state_priorities[s] = priority
            heapq.heappush(pq, (-priority, counter, s))  # Negative for max heap
            counter += 1
    
    V_diff_history = []  # Track ||V - V*||
    
    # Record initial error
    v_diff_norm = np.linalg.norm(V - V_star)
    V_diff_history.append((0, v_diff_norm))
    
    iteration = 0
    
    if verbose:
        print("Starting Prioritized Sweeping Value Iteration...")
    
    # Continue until convergence or max iterations
    while pq and iteration < max_iterations:
        # Get highest priority state
        _, _, s = heapq.heappop(pq)
        
        # Remove from tracking dictionary
        if s in state_priorities:
            del state_priorities[s]
        
        # Update value function for the state
        old_v = V[s]
        q_values = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, terminated in env.P[s][a]:
                if terminated:
                    q_values[a] += prob * reward
                else:
                    q_values[a] += prob * (reward + gamma * V[next_state])
        V[s] = np.max(q_values)
        
        # Count this as one iteration
        iteration += 1
        
        # Record error norm at EVERY iteration
        v_diff_norm = np.linalg.norm(V - V_star)
        V_diff_history.append((iteration, v_diff_norm))
        
        # If value changed significantly, update predecessors
        if abs(old_v - V[s]) > epsilon:
            for pred in predecessors[s]:
                # Calculate priority for predecessor
                old_v_pred = V[pred]
                q_values = np.zeros(nA)
                for a in range(nA):
                    for prob, next_state, reward, terminated in env.P[pred][a]:
                        if terminated:
                            q_values[a] += prob * reward
                        else:
                            q_values[a] += prob * (reward + gamma * V[next_state])
                priority = abs(np.max(q_values) - old_v_pred)
                
                # Add to priority queue if priority is high enough
                if priority > epsilon:
                    # If state already in queue, compare priorities
                    if pred in state_priorities:
                        # If new priority is higher, update it
                        if priority > state_priorities[pred]:
                            state_priorities[pred] = priority
                            # Add new entry (old one remains but will be ignored)
                            heapq.heappush(pq, (-priority, counter, pred))
                            counter += 1
                    else:
                        # New entry
                        state_priorities[pred] = priority
                        heapq.heappush(pq, (-priority, counter, pred))
                        counter += 1
    
    if verbose:
        if not pq:
            print(f"Prioritized Sweeping VI converged after {iteration} updates.")
        else:
            print(f"Prioritized Sweeping VI reached max iterations ({max_iterations}).")
    
    return V, V_diff_history, iteration

def policy_evaluation(policy, env, V, V_star, gamma=0.9, epsilon=1e-6, max_iterations=50000, 
                      V_diff_history=None, total_iterations=0, verbose=True, record=True):
    """
    Evaluate a policy and return the value function and number of iterations.
    Also track V - V* at each iteration.
    """
    nS = env.observation_space.n
    iteration = 0
    converged = False
    
    while iteration < max_iterations and not converged:
        delta = 0

        for s in range(nS):
            v_old = V[s]
            
            # Get action based on the policy
            a = policy[s]
            
            # Calculate new value
            new_v = 0
            for prob, next_state, reward, terminated in env.P[s][a]:
                if terminated:
                    new_v += prob * reward
                else:
                    new_v += prob * (reward + gamma * V[next_state])
            
            V[s] = new_v
            delta = max(delta, abs(v_old - V[s]))
            
            # Count this as one iteration
            iteration += 1
            
            # Record error norm at EVERY iteration
            # we are only interested in iterations after the first one
            if record:
                v_diff_norm = np.linalg.norm(V - V_star)
                V_diff_history.append((iteration + total_iterations, v_diff_norm))
        
        if delta < epsilon:
            converged = True

    if verbose:
        if converged:
            print(f"Policy Evaluation converged after {iteration} updates.")
        else:
            print(f"Policy Evaluation reached max iterations ({max_iterations}).")
    
    # only increase the total iterations if this is not the first evaluation - the first one tends to be very bad
    if not record:
        return V, total_iterations
    else:
        return V, iteration + total_iterations

def policy_improvement(V, old_policy, env, gamma=0.9):
    """
    Improve policy based on value function.
    Returns:
    - new policy
    - boolean indicating if policy changed
    """
    nS = env.observation_space.n
    nA = env.action_space.n
    
    new_policy = np.zeros(nS, dtype=int)
    policy_stable = True
    
    for s in range(nS): 
        # Calculate Q values for all actions
        q_values = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, terminated in env.P[s][a]:
                if terminated:
                    q_values[a] += prob * reward
                else:
                    q_values[a] += prob * (reward + gamma * V[next_state])
        
        # Choose best action
        new_policy[s] = np.argmax(q_values)
        
        # Check if policy changed
        if old_policy[s] != new_policy[s]:
            policy_stable = False
    
    return new_policy, policy_stable

def compute_policy_iteration(env, V_star, gamma=0.9, epsilon=1e-6, max_iterations=50000, seed=42, verbose=True):
    """
    Policy Iteration.
    Returns:
    - value function
    - convergence history (V - V*)
    - number of iterations
    """
    np.random.seed(seed)
    random.seed(seed)
    
    nS = env.observation_space.n
    nA = env.action_space.n
    
    # Initialize with a random policy and zero value function
    policy = np.random.randint(0, nA, size=nS)
    V = np.zeros(nS)
    
    V_diff_history = []  # Track ||V - V*||
    
    # Record initial error
    v_diff_norm = np.linalg.norm(V - V_star)
    V_diff_history.append((0, v_diff_norm))
    
    total_iterations = 0
    policy_iteration = 0
    
    if verbose:
        print(f"Starting Policy Iteration (seed {seed})...")
    
    policy_stable = False
    record = False
    start_recording_after = 2

    while not policy_stable and policy_iteration < 100:  # Limit policy iterations
        policy_iteration += 1
        
        # Policy Evaluation
        if  policy_iteration > start_recording_after:
            record = True
        
        V, total_iterations = policy_evaluation(policy, env, V.copy(), V_star, gamma, epsilon, max_iterations, 
                                                V_diff_history, total_iterations, verbose=verbose, record=record)

        # Policy Improvement
        policy, policy_stable = policy_improvement(V, policy, env, gamma)
    
    if verbose:
        print(f"Policy Iteration (seed {seed}) completed after {policy_iteration} policy updates and {total_iterations} value iterations.")
    
    return V, V_diff_history, total_iterations

def analyze_environment(env_name, gamma=0.9, epsilon=1e-6, max_iterations=50000):
    """
    Analyze convergence for the given environment
    """
    print(f"\n{'='*50}")
    print(f"Analyzing {env_name}")
    print(f"{'='*50}")
    
    # Create environment
    env = gym.make(env_name)
    
    # 1. Compute V* using standard value iteration
    print(f"\nStep 1: Computing optimal value function (V*)")
    V_star, vi_iterations = compute_value_iteration(env, gamma, epsilon, max_iterations)
    
    # 2. Run variants of Value Iteration
    print(f"\nStep 2: Running VI variants")
    
    # Gauss-Seidel VI
    _, gs_history, gs_iterations = compute_gauss_seidel_vi(env, V_star, gamma, epsilon, max_iterations)
    
    # Prioritized Sweeping VI
    _, ps_history, ps_iterations = compute_prioritized_sweeping_vi(env, V_star, gamma, epsilon, max_iterations)
    
    # 3. Run Policy Iteration with different seeds
    print(f"\nStep 3: Running Policy Iteration with multiple seeds")
    pi_histories = []
    
    for seed in RANDOM_SEEDS:
        _, pi_history, pi_iterations = compute_policy_iteration(
            env, V_star, gamma, epsilon, max_iterations, seed=seed, verbose=True
        )
        pi_histories.append((seed, pi_history))
        print(f"  - Seed {seed}: {pi_iterations} iterations")
    
    # average the history of policy iterations, noting that their lengths may differ
    max_length = max(len(pi_history) for _, pi_history in pi_histories)
    pi_history_avg = np.zeros(max_length)
    for _, pi_history in pi_histories:
        pi_history_len = len(pi_history)
        pi_history_avg[:pi_history_len] += np.array(pi_history)[:, 1]
    
    pi_history_avg /= len(pi_histories)
    
    # Convert to list of tuples for plotting
    pi_history_avg = [(i, v) for i, v in enumerate(pi_history_avg)]
    
    pi_for_plot = []
    pi_for_plot.append(("Average", pi_history_avg))

    # Plot convergence
    plot_convergence(env_name, gs_history, ps_history, pi_for_plot)
    
    print(f"\nAnalysis for {env_name} completed.")
    print(f"{'='*50}")
    
    return V_star

def plot_convergence(env_name, gs_history, ps_history, pi_histories):
    """
    Plot convergence curves for all algorithms
    """
    plt.figure(figsize=(12, 8))
    
    # Plot Gauss-Seidel VI
    iterations, errors = zip(*gs_history)
    plt.plot(iterations, errors, 'b-', label='Gauss-Seidel VI')
    
    # Plot Prioritized Sweeping VI
    iterations, errors = zip(*ps_history)
    plt.plot(iterations, errors, 'g-', label='Prioritized Sweeping VI')
    
    # Plot Policy Iteration for each seed
    colors = ['r', 'm', 'c', 'y', 'k']
    for i, (seed, pi_history) in enumerate(pi_histories):
        iterations, errors = zip(*pi_history)
        plt.plot(iterations, errors, f'{colors[i]}--', label=f'Policy Iteration (seed {seed})')
    
    plt.xlabel('Number of Updates')
    plt.ylabel('||V - V*|| (L2 Norm)')
    plt.title(f'Convergence Analysis for {env_name}')
    plt.yscale('linear')  # Normal scale for better visualization
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    # Save figure
    plt.savefig(f'{env_name}_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Set parameters
    gamma = 0.9
    epsilon = 1e-4
    max_iterations = 50000
    
    # Analyze Taxi environment
    analyze_environment("Taxi-v3", gamma, epsilon, max_iterations)
    
    # Analyze FrozenLake environment
    analyze_environment("FrozenLake-v1", gamma, epsilon, max_iterations)
