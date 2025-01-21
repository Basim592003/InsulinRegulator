import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from scipy.interpolate import make_interp_spline  

model_path =r"D:\UNIVERSITY\SEM7\Regulator\logs\checkpoints\rl_model_3600000_steps.zip"
model = PPO.load(model_path)

csv_path = r"D:\UNIVERSITY\SEM7\Regulator\simulated_type2_diabetic.csv"
data = pd.read_csv(csv_path)

required_columns = ["blood_glucose", "iob", "hour", "is_sleep", "carbs", "bg_history_1", "bg_history_2", "bg_history_3",
                    "insulin_history_1", "insulin_history_2", "insulin_history_3"]
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing columns in CSV: {missing_columns}")
###HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
def prepare_observation(row):
    return {
        "blood_glucose": np.array([row["blood_glucose"]], dtype=np.float32),
        "iob": np.array([row["iob"]], dtype=np.float32),
        "hour": int(row["hour"]),
        "is_sleep": int(row["is_sleep"]),
        "carbs": np.array([row["carbs"]], dtype=np.float32),
        "bg_history": np.array([row["bg_history_1"], row["bg_history_2"], row["bg_history_3"]], dtype=np.float32),
        "insulin_history": np.array([row["insulin_history_1"], row["insulin_history_2"], row["insulin_history_3"]], dtype=np.float32),
    }

predictions = []
for index, row in data.iterrows():
    obs = prepare_observation(row)
    action, _ = model.predict(obs, deterministic=True)
    predictions.append(action[0])  

data["predicted_insulin_dose"] = predictions

def smooth_data_with_spline(data):
    x = np.arange(len(data))  
    spline = make_interp_spline(x, data) 
    x_new = np.linspace(x.min(), x.max(), 500)  
    smooth_data = spline(x_new)
    return x_new, smooth_data

x_new_bg, smoothed_bg = smooth_data_with_spline(data["blood_glucose"].values)
x_new_predicted, smoothed_predicted_insulin_dose = smooth_data_with_spline(data["predicted_insulin_dose"].values)

fig, ax1 = plt.subplots(figsize=(10, 4))  

color_glucose = 'blue'
ax1.plot(x_new_bg, smoothed_bg, label="Blood Glucose", color=color_glucose)
ax1.axhline(y=110, color="green", linestyle="--", label="Target Min (Glucose)")
ax1.axhline(y=150, color="red", linestyle="--", label="Target Max (Glucose)")
ax1.set_ylabel("Blood Glucose (mg/dL)", color=color_glucose)
ax1.tick_params(axis='y', labelcolor=color_glucose)
ax1.set_title("Blood Glucose and Predicted Insulin Dose Over the Day")
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  
color_insulin = 'purple'
ax2.plot(x_new_predicted, smoothed_predicted_insulin_dose, label="Predicted Insulin Dose", color=color_insulin, alpha=0.7)
ax2.set_ylabel("Insulin Dose (units)", color=color_insulin)
ax2.tick_params(axis='y', labelcolor=color_insulin)
ax2.legend(loc='upper right')

ax1.set_xlabel("Hour of the Day")
fig.tight_layout()

plt.show()
