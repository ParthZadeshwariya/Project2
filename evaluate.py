from stable_baselines3 import PPO
from drone_env import DroneEnv

env = DroneEnv()
model = PPO.load("drone_model")

success = 0
episodes = 50

for _ in range(episodes):
    obs, _ = env.reset()

    for _ in range(250):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

        if done:
            if reward == 100:
                success += 1
            break

print("Success rate:", success/episodes*100, "%")
