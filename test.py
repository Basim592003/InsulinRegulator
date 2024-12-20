import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from enviorment import BloodGlucoseEnvironment  # Import your environment
from stable_baselines3 import PPO  # Import the PPO model

def run_inference(model_path="ppo_blood_glucose_agent", simulation_hours=24, output_csv=None):
    """
    Runs inference using a trained PPO model on the Blood Glucose Environment.
    
    Parameters:
    - model_path (str): Path to the saved PPO model.
    - simulation_hours (int): Number of hours to simulate.
    - output_csv (str): Optional path to save results as a CSV file.

    Returns:
    - dict: Results containing blood glucose levels, insulin doses, and rewards.
    """
    # Load the trained PPO model
    model = PPO.load(model_path)

    # Initialize the environment
    env = BloodGlucoseEnvironment()
    obs, info = env.reset()  # Reset environment and get initial state

    # Initialize storage for metrics
    bg_levels = []  # Blood glucose levels
    insulin_doses = []  # Insulin doses
    rewards = []  # Rewards
    hours = []  # Time in hours

    # Simulation loop for the specified number of hours
    for _ in range(simulation_hours):
        # Get the action from the model
        action, _ = model.predict(obs, deterministic=True)

        # Perform the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Record metrics
        bg_levels.append(obs["blood_glucose"][0])  # Blood glucose level
        insulin_doses.append(action[0])           # Raw insulin dose
        rewards.append(reward)                    # Reward
        hours.append(info.get("hour", len(hours)))  # Track hour progression

        # Reset the environment if the episode ends
        if terminated or truncated:
            obs, info = env.reset()

    # Save results to a CSV file if output path is provided
    if output_csv:
        results = pd.DataFrame({
            "Hour": hours,
            "Blood Glucose": bg_levels,
            "Insulin Dose": insulin_doses,
            "Reward": rewards
        })
        results.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    # Return results
    return {
        "hours": hours,
        "blood_glucose": bg_levels,
        "insulin_doses": insulin_doses,
        "rewards": rewards
    }

def plot_results(results):
    """
    Plots the results from the inference run.

    Parameters:
    - results (dict): Dictionary containing blood glucose levels, insulin doses, and rewards.
    """
    hours = np.arange(len(results["blood_glucose"]))

    plt.figure(figsize=(15, 8))

    # Plot blood glucose levels
    plt.subplot(3, 1, 1)
    plt.plot(hours, results["blood_glucose"], label="Blood Glucose Level", color='blue')
    plt.axhline(y=80, color='green', linestyle='--', label="Target Min")  # Example thresholds
    plt.axhline(y=180, color='green', linestyle='--', label="Target Max")
    plt.axhline(y=70, color='red', linestyle='--', label="Hypo Threshold")
    plt.axhline(y=250, color='orange', linestyle='--', label="Hyper Threshold")
    plt.ylabel("Blood Glucose (mg/dL)")
    plt.legend()
    plt.title("Blood Glucose Levels Over Time")

    # Plot insulin doses
    plt.subplot(3, 1, 2)
    plt.step(hours, results["insulin_doses"], label="Insulin Dose", color='purple', where='post')
    plt.ylabel("Insulin Dose (units)")
    plt.legend()
    plt.title("Insulin Doses Over Time")

    # Plot rewards
    plt.subplot(3, 1, 3)
    plt.plot(hours, results["rewards"], label="Reward", color='black')
    plt.axhline(y=0, color='gray', linestyle='--', label="Zero Reward")
    plt.ylabel("Reward")
    plt.xlabel("Time (hours)")
    plt.legend()
    plt.title("Rewards Over Time")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run inference
    simulation_hours = 168  # Simulate for one week
    results = run_inference(model_path="ppo_blood_glucose_agent", simulation_hours=simulation_hours, output_csv="inference_results.csv")

    # Plot the results
    plot_results(results)

    # Print summary metrics
    print(f"Average Blood Glucose Level: {np.mean(results['blood_glucose']):.2f} mg/dL")
    print(f"Average Insulin Dose: {np.mean(results['insulin_doses']):.2f} units")
    print(f"Average Reward: {np.mean(results['rewards']):.2f}")
