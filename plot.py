import plotly.subplots as sp
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import make_interp_spline
from enviorment import BloodGlucoseEnvironment  
from stable_baselines3 import PPO

# Initialize the environment and load the trained model
env = BloodGlucoseEnvironment()
model = PPO.load(r"D:\UNIVERSITY\SEM7\Regulator\Regulator\ppo_blood_glucose_agent.zip")
# Reset the environment
obs, info = env.reset()

# Initialize variables to store results
bg_levels = []
insulin_doses = []
rewards = []
hours = []

# Simulate for a smaller time period (e.g., 24 * 3 hours)
for _ in range(24 * 3):
    # Get action from the trained model
    action, _ = model.predict(obs, deterministic=True)
    
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Store metrics
    bg_levels.append(obs["blood_glucose"][0])  # Blood glucose level
    insulin_doses.append(action[0])           # Insulin dose
    rewards.append(reward)                    # Reward
    hours.append(info.get("hour", len(hours)))  # Track hour progression
    
    # Reset environment if episode ends
    if terminated or truncated:
        obs, info = env.reset()

# Smooth the data using make_interp_spline
def smooth_curve(data, n_points=500):
    x = np.arange(len(data))
    spline = make_interp_spline(x, data)  # Cubic spline interpolation
    x_smooth = np.linspace(x.min(), x.max(), n_points)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

# Generate smoothed curves
bg_x, bg_y = smooth_curve(bg_levels)
insulin_x, insulin_y = smooth_curve(insulin_doses)
reward_x, reward_y = smooth_curve(rewards)
fig = sp.make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    vertical_spacing=0.05,  # Reduced spacing between subplots
    subplot_titles=("Blood Glucose Levels", "Insulin Doses", "Rewards")
)
# Extract meal times
mealtimes = env.meal_times  
total_hours = len(bg_levels)  # Total length of the time series
days = total_hours // 24      # Calculate number of days in the data
extended_meal_times = []      # List to store meal times across all days

for day in range(days):
    extended_meal_times += [t + (24 * day) for t in env.meal_times]

extended_meal_times = [t for t in extended_meal_times if t < total_hours]

# Extended meal BG values
extended_meal_bg_values = [bg_levels[t] for t in extended_meal_times]


# Plot blood glucose levels
fig.add_trace(go.Scatter(x=np.arange(len(bg_levels)), y=bg_levels,
                         mode='lines+markers', name="Original BG Level",
                         line=dict(color='lightblue', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=bg_x, y=bg_y, mode='lines', name="Smoothed BG Level",
                         line=dict(color='blue', width=3)), row=1, col=1)

# Highlight meal times on blood glucose plot
fig.add_trace(go.Scatter(x=extended_meal_times, y=extended_meal_bg_values, mode='markers',
                         name="Meal Times", marker=dict(color='red', size=10, symbol='x')), row=1, col=1)

# Add threshold lines
fig.add_hline(y=env.target_min, line_dash="dash", line_color="green", row=1, col=1)
fig.add_hline(y=env.target_max, line_dash="dash", line_color="green", row=1, col=1)
fig.add_hline(y=env.hypo_threshold, line_dash="dash", line_color="red", row=1, col=1)
fig.add_hline(y=env.hyper_threshold, line_dash="dash", line_color="orange", row=1, col=1)

# Plot insulin doses
fig.add_trace(go.Scatter(x=np.arange(len(insulin_doses)), y=insulin_doses,
                         mode='lines+markers', name="Original Insulin Dose",
                         line=dict(color='red', width=2)), row=2, col=1)
fig.add_trace(go.Scatter(x=insulin_x, y=insulin_y, mode='lines', name="Smoothed Insulin Dose",
                         line=dict(color='darkred', width=3)), row=2, col=1)

# Highlight meal times on insulin dose plot
fig.add_trace(go.Scatter(x=extended_meal_times, y=[insulin_doses[t] for t in extended_meal_times], mode='markers',
                         name="Meal Times", marker=dict(color='red', size=10, symbol='x')), row=2, col=1)

# Plot rewards
fig.add_trace(go.Scatter(x=np.arange(len(rewards)), y=rewards,
                         mode='lines+markers', name="Original Reward",
                         line=dict(color='gray', width=2)), row=3, col=1)
fig.add_trace(go.Scatter(x=reward_x, y=reward_y, mode='lines', name="Smoothed Reward",
                         line=dict(color='black', width=3)), row=3, col=1)

# Highlight meal times on reward plot
fig.add_trace(go.Scatter(x=extended_meal_times, y=[rewards[t] for t in extended_meal_times], mode='markers',
                         name="Meal Times", marker=dict(color='red', size=10, symbol='x')), row=3, col=1)


# Layout updates
fig.update_layout(
    height=850,  # Increase overall figure height
    width=1500,   # Adjust width to keep proportions
    title="Blood Glucose, Insulin Doses, and Rewards Over Time",
    legend_title="Legend",
    showlegend=True
)

# Update axis titles
fig.update_yaxes(title_text="Blood Glucose (mg/dL)", row=1, col=1)
fig.update_yaxes(title_text="Insulin Dose (units)", row=2, col=1)
fig.update_yaxes(title_text="Reward", row=3, col=1)
fig.update_xaxes(title_text="Time (Hours)", row=3, col=1)  # Add x-axis label only to the last plot

# Show plot
fig.show()
