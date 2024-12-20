import gymnasium as gym
import numpy as np
from gymnasium import spaces

class BloodGlucoseEnvironment(gym.Env):
    """
    Blood Glucose Environment for Reinforcement Learning - Enhanced for Type 2 Diabetes
    """
    def __init__(self):
        super(BloodGlucoseEnvironment, self).__init__()

        # --- ENVIRONMENT PARAMETERS ---
        self.bg_min = 70
        self.bg_max = 300
        self.hypo_threshold = 80
        self.hyper_threshold = 180
        self.target_min = 110
        self.target_max = 150
        self.meal_times = [8, 13, 18, 21]
        self.sleep_start = 23
        self.sleep_end = 7
        self.pending_glucose_rise = 0 
        self.pending_insulin_effect = 0  

        # Action space: insulin dosage
        self.action_space = spaces.Box(low=0.0, high=20.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "blood_glucose": spaces.Box(low=self.bg_min, high=self.bg_max, shape=(1,), dtype=np.float32),
            "iob": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "hour": spaces.Discrete(24),
            "is_sleep": spaces.Discrete(2),
            "carbs": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "bg_history": spaces.Box(low=self.bg_min, high=self.bg_max, shape=(3,), dtype=np.float32),
            "insulin_history": spaces.Box(low=0.0, high=20.0, shape=(3,), dtype=np.float32)
        })
        
        self.reset()

    def step(self, action):
        insulin_units = action[0]

        # --- Insulin on Board (IOB) ---
        self.iob = self._update_iob(insulin_units)

        if self.pending_glucose_rise > 0:
            if self.bg_history[-1] is not None: 
                self.current_bg = self.bg_history[-1] + self.pending_glucose_rise  
            else:
                self.current_bg += self.pending_glucose_rise  
            self.pending_glucose_rise = 0  

        carbs = self._get_meal_carbs() if self.current_hour in self.meal_times else 0
        glucose_rise = self._calculate_glucose_impact(carbs)
        
        self.pending_glucose_rise = glucose_rise

        # --- Apply Insulin Effect Immediately ---
        insulin_effect = self._calculate_insulin_effect(self.iob)
        self.current_bg -= insulin_effect  

        # --- Basal Glucose Production ---
        basal_rate = self._get_basal_rate()
        self.current_bg += basal_rate

        # --- Normalize Blood Glucose ---
        self.current_bg = np.clip(self.current_bg, self.bg_min, self.bg_max)

        # --- Update Time and Sleep Status ---
        self.current_hour = (self.current_hour + 1) % 24
        self.is_sleep = 1 if self.current_hour >= self.sleep_start or self.current_hour < self.sleep_end else 0

        # --- Calculate Reward ---
        reward = self._calculate_reward()
        self.cumulative_reward += reward        

        # --- Update Observations ---
        self.bg_history = np.roll(self.bg_history, -1)
        self.bg_history[-1] = self.current_bg
        self.insulin_history = np.roll(self.insulin_history, -1)
        self.insulin_history[-1] = insulin_units
        observation = self._get_observation()

        # --- Check for Termination ---
        terminated = self.current_hour == 0  # End of day
        return observation, reward, terminated, False, {}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_bg = np.random.uniform(140, 180)
        self.iob = 0
        self.current_hour = 0
        self.is_sleep = 1
        self.cumulative_reward = 0
        self.bg_history = np.full(3, self.current_bg)
        self.insulin_history = np.zeros(3)
        self.pending_glucose_rise = 0  # Tracks pending glucose rise from meal
        self.pending_insulin_effect = 0  # Tracks pending insulin effect
        return self._get_observation(), {}

    def render(self, mode='human'):
        print(f"Hour: {self.current_hour}, BG: {self.current_bg:.2f}, IOB: {self.iob:.2f}")

    def _calculate_glucose_impact(self,carbs):

    
        return carbs * np.random.uniform(2.0, 4.0)

    def _calculate_insulin_effect(self, iob):
        # Effective insulin sensitivity factor (ISF): 25-50 mg/dL per unit of insulin
        effective_isf = np.random.uniform(25, 50)
        return iob * effective_isf

    def _get_basal_rate(self):
        # Reduced and more physiological basal rates
        return np.random.uniform(0.5, 1.5) if self.is_sleep else np.random.uniform(1.0, 2.0)

    def _calculate_reward(self):
        if self.target_min <= self.current_bg <= self.target_max:
            reward = 3.0
        elif self.hypo_threshold <= self.current_bg < self.target_min:
            reward = -1.0 - (self.target_min - self.current_bg) / 10
        elif self.target_max < self.current_bg <= self.hyper_threshold:
            reward = -1.0 - (self.current_bg - self.target_max) / 20
        elif self.current_bg < self.hypo_threshold:
            reward = -5.0 - (self.hypo_threshold - self.current_bg) / 5
        else:
            reward = -3.0 - (self.current_bg - self.hyper_threshold) / 10

        # Penalty for rapid BG changes
        bg_change = self.bg_history[-1] - self.bg_history[-2]
        if abs(bg_change) > 20:
            reward -= abs(bg_change) / 20

        return reward

    def _get_meal_carbs(self):

        if self.current_hour == 8:
            return np.random.uniform(40, 60)
        elif self.current_hour == 13:
            return np.random.uniform(50, 70)
        elif self.current_hour == 18:
            return np.random.uniform(60, 80)
        elif self.current_hour == 21:
            return np.random.uniform(20, 40)
        return 0

    def _update_iob(self, insulin_units):
    # Simple IOB model: exponential decay
        decay_rate = 0.05  # Adjust as needed
        new_iob = self.iob * (1 - decay_rate) + insulin_units

        
        # Fix dimension mismatch by adjusting insulin history or decay_rates
        decay_rates = np.array([0.7, 0.5, 0.3])  # Example decay rates matching history length
        decay_effect = np.dot(decay_rates, self.insulin_history)
    
        # Update IOB to include decay effect
        self.iob = max(0, new_iob - decay_effect)  # Ensure IOB remains non-negative
        return self.iob


    def _get_observation(self):
        return {
            "blood_glucose": np.array([self.current_bg], dtype=np.float32),
            "iob": np.array([self.iob], dtype=np.float32),
            "hour": self.current_hour,
            "is_sleep": self.is_sleep,
            "carbs": np.array([0], dtype=np.float32),
            "bg_history": np.array(self.bg_history, dtype=np.float32),
            "insulin_history": np.array(self.insulin_history, dtype=np.float32)
        }
