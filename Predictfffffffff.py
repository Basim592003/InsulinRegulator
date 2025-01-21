import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from scipy.interpolate import make_interp_spline
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load PPO model
model_path = r"logs\best_model\best_model.zip"
model = PPO.load(model_path)

# Function to collect patient information
def collect_patient_info():
    print("Please enter the following patient details:")
    age = int(input("Age (years): "))
    weight = float(input("Weight (kg): "))
    insulin_sensitivity = float(input("Insulin Sensitivity (unit/kg): "))
    activity_level = input("Activity Level (low/medium/high): ").lower()
    
    return {
        "age": age,
        "weight": weight,
        "insulin_sensitivity": insulin_sensitivity,
        "activity_level": activity_level
    }

# Collect patient information
patient_info = collect_patient_info()
print(f"Patient Information: {patient_info}")

# CGM data generator with specific meal and snack times
def generate_cgm_data(num_entries=24, seed=42, patient_info=None):
    np.random.seed(seed)
    
    blood_glucose = np.clip(140 + np.random.normal(0, 20, size=num_entries), 70, 200)
    iob = np.clip(np.random.normal(1, 0.5, size=num_entries), 0, None)
    hour = np.arange(num_entries) % 24
    is_sleep = ((hour >= 22) | (hour <= 6)).astype(int)
    
    meal_schedule = {
        "breakfast": [8],
        "lunch": [12],
        "snack": [15],
        "dinner": [19],
    }
    
    carbs = np.zeros(num_entries)
    for meal, times in meal_schedule.items():
        for t in times:
            if meal in ["breakfast", "lunch", "dinner"]:
                carbs[hour == t] = np.random.choice([30, 45, 60])
            elif meal == "snack":
                carbs[hour == t] = np.random.choice([15, 30])

    if patient_info:
        if patient_info["activity_level"] == "high":
            carbs += 10
        elif patient_info["activity_level"] == "low":
            carbs -= 5

    bg_history = blood_glucose.copy()
    bg_history_1 = np.roll(bg_history, 1)
    bg_history_2 = np.roll(bg_history, 2)
    bg_history_3 = np.roll(bg_history, 3)

    bg_history_1[0] = bg_history_2[0] = bg_history_3[0] = blood_glucose[0]
    insulin_history = np.random.uniform(0, 5, size=(num_entries, 3))

    data = pd.DataFrame({
        "blood_glucose": blood_glucose,
        "iob": iob,
        "hour": hour,
        "is_sleep": is_sleep,
        "carbs": carbs,
        "bg_history_1": bg_history_1,
        "bg_history_2": bg_history_2,
        "bg_history_3": bg_history_3,
        "insulin_history_1": insulin_history[:, 0],
        "insulin_history_2": insulin_history[:, 1],
        "insulin_history_3": insulin_history[:, 2],
    })
    return data

data = generate_cgm_data(patient_info=patient_info)

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

def calculate_insulin_effect(iob, patient_info):
    base_isf = 50
    
    if patient_info["age"] > 50:
        base_isf -= 10
    if patient_info["weight"] > 80:
        base_isf -= 5
    if patient_info["activity_level"] == "high":
        base_isf += 10
    elif patient_info["activity_level"] == "low":
        base_isf -= 5

    effective_isf = np.random.uniform(base_isf - 10, base_isf + 10)
    return iob * effective_isf

def apply_insulin_effect_on_glucose(data, patient_info):
    glucose_after_insulin = data["blood_glucose"].copy()
    for i, dose in enumerate(data["predicted_insulin_dose"]):
        iob = data["iob"][i]
        insulin_effect = calculate_insulin_effect(iob, patient_info)
        glucose_after_insulin[i] = max(70, glucose_after_insulin[i] - insulin_effect)
    return glucose_after_insulin

insulin_effect_on_glucose = apply_insulin_effect_on_glucose(data, patient_info)

def smooth_data_with_spline(data):
    x = np.arange(len(data))  
    spline = make_interp_spline(x, data) 
    x_new = np.linspace(x.min(), x.max(), 500)  
    smooth_data = spline(x_new)
    return x_new, smooth_data

x_new_bg, smoothed_bg = smooth_data_with_spline(data["blood_glucose"].values)
x_new_effect, smoothed_bg_after_insulin = smooth_data_with_spline(insulin_effect_on_glucose)
x_new_insulin, smoothed_insulin_dose = smooth_data_with_spline(data["predicted_insulin_dose"].values)

meal_times = data[data["carbs"] > 0].index
meal_carbs = data.loc[meal_times, "carbs"]

fig, axs = plt.subplots(3, 1, figsize=(10, 12))

axs[0].plot(x_new_bg, smoothed_bg, label="Blood Glucose (CGM)", color="blue")
axs[0].axhline(y=80, color="green", linestyle="--", label="Target Min (Glucose)")
axs[0].axhline(y=120, color="red", linestyle="--", label="Target Max (Glucose)")
axs[0].set_ylabel("Blood Glucose (mg/dL)")
axs[0].set_title("Blood Glucose from CGM")

axs[0].legend()

axs[1].plot(x_new_effect, smoothed_bg_after_insulin, label="Blood Glucose After Insulin", color="orange")
axs[1].axhline(y=80, color="green", linestyle="--")
axs[1].axhline(y=120, color="red", linestyle="--")
axs[1].set_ylabel("Blood Glucose After Insulin (mg/dL)")
axs[1].set_title("Blood Glucose After Insulin Effect")
for meal_time, carb in zip(meal_times, meal_carbs):
    if carb > 20:  
        axs[1].text(meal_time, smoothed_bg_after_insulin[meal_time], f"{carb}g", color="orange", fontsize=8, ha='right', va='bottom')
axs[1].legend()

axs[2].plot(x_new_insulin, smoothed_insulin_dose, label="Predicted Insulin Dose", color="purple")
axs[2].set_ylabel("Insulin Dose (units)")
axs[2].set_title("Predicted Insulin Dose")

axs[2].legend()

fig.tight_layout()
plt.show()
