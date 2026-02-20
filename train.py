from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from drone_env import DroneEnv
import matplotlib.pyplot as plt

class RewardCallback(BaseCallback):

    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self):
        self.rewards.append(self.locals["rewards"][0])
        return True


env = DroneEnv()
callback = RewardCallback()

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=150000, callback=callback)

model.save("drone_model")

plt.plot(callback.rewards)
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("Training Reward Curve")
plt.savefig("training_curve.png")
plt.show()
