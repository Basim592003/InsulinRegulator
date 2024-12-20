import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement, CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from enviorment import BloodGlucoseEnvironment  # Assuming your environment is saved as environment.py

# --- Environment Setup ---
env = BloodGlucoseEnvironment()
env = Monitor(env)  # Monitor to track rewards and episodes

# --- Action Noise for Exploration ---
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

# --- Callback
# s ---

# Checkpoint callback to save the model periodically
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./logs/checkpoints/",
    name_prefix="rl_model"
)

# Stop training if there's no improvement
stop_training_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=10,
    min_evals=20,
    verbose=1
)

# Evaluation callback to periodically evaluate performance
eval_callback = EvalCallback(
    eval_env=env,
    callback_on_new_best=stop_training_callback,
    best_model_save_path="/logs/best_model/",
    log_path="./logs/results/",
    eval_freq=5000,
    deterministic=True,
    render=False,
    n_eval_episodes=10
)

# Combine callbacks into a single list
callback = CallbackList([checkpoint_callback, eval_callback])

# --- PPO Model Setup ---
model = PPO(
    policy="MultiInputPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./logs/ppo_blood_glucose_tensorboard/",
    batch_size=512,         # Balanced for stability
    n_steps=4096,           # Moderate to reduce variance
    learning_rate=3e-4,     # Lowered for stable learning
    gamma=0.99,             # Standard discount factor
    gae_lambda=0.95,        # Generalized Advantage Estimation
    ent_coef=0.01,          # Entropy coefficient for exploration
    vf_coef=0.5,            # Value function coefficient
    clip_range=0.2,         # PPO clipping
    n_epochs=10,            # Number of epochs for policy updates
    policy_kwargs=dict(
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]  # Simplified network
    )
)

# --- Training the Agent ---
model.learn(
    total_timesteps=100000,  
    callback=callback
)

# Save the final trained model
model.save("ppo_blood_glucose_agent")

# --- Evaluation ---
mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=20, deterministic=True
)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# --- Testing  ---
model = PPO.load("ppo_blood_glucose_agent")

# Test for one simulated week (7 days)
obs, info = env.reset()
for step in range(24 * 7):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

# Close the environment
env.close()
