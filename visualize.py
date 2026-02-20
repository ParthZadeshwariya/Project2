import pybullet as p
import pybullet_data
import time
import numpy as np

from stable_baselines3 import PPO
from drone_env import DroneEnv


# ===============================
# CONNECT
# ===============================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,0)

p.resetDebugVisualizerCamera(
    cameraDistance=15,
    cameraYaw=45,
    cameraPitch=-35,
    cameraTargetPosition=[5,5,2]
)

plane = p.loadURDF("plane.urdf")


# ===============================
# LOAD ENV + MODEL
# ===============================
env = DroneEnv()
model = PPO.load("drone_model", device="cpu")

obs, _ = env.reset()

episode_count = 1


# ===============================
# CREATE DRONE
# ===============================
drone_visual = p.createVisualShape(
    p.GEOM_SPHERE,
    radius=0.25,
    rgbaColor=[0,0,1,1]
)

drone = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=drone_visual,
    basePosition=[0,0,1]
)


# ===============================
# CREATE GOAL
# ===============================
goal_visual = p.createVisualShape(
    p.GEOM_SPHERE,
    radius=0.3,
    rgbaColor=[0,1,0,1]
)

goal = p.createMultiBody(
    baseMass=0,
    baseVisualShapeIndex=goal_visual,
    basePosition=[0,0,1]
)


# ===============================
# OBSTACLE CREATION FUNCTION
# ===============================
obstacles = []

def create_obstacles():
    global obstacles

    for obs in obstacles:
        p.removeBody(obs)

    obstacles.clear()

    box_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.4,0.4,0.4],
        rgbaColor=[1,0,0,1]
    )

    for obs_pos in env.obstacles:
        box = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=box_visual,
            basePosition=obs_pos.tolist()
        )
        obstacles.append(box)


create_obstacles()


# ===============================
# TEXT HANDLES
# ===============================
text_id = None


# ===============================
# MAIN LOOP
# ===============================
while p.isConnected():

    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)

    # update drone position
    x,y,z = env.drone_pos
    p.resetBasePositionAndOrientation(drone,[x,y,z],[0,0,0,1])

    # update goal
    gx,gy,gz = env.goal
    p.resetBasePositionAndOrientation(goal,[gx,gy,gz],[0,0,0,1])

    # remove old text
    if text_id is not None:
        p.removeUserDebugItem(text_id)

    text_id = p.addUserDebugText(
        f"Episode: {episode_count}",
        [0,0,8],
        textSize=1.5,
        textColorRGB=[1,1,1]
    )

    p.stepSimulation()
    time.sleep(0.12)

    # ===========================
    # EPISODE END
    # ===========================
    if done:

        if reward == 100:
            print("GOAL REACHED")

            # goal turns blue
            p.changeVisualShape(goal,-1,rgbaColor=[0,0,1,1])

            status = "GOAL REACHED"

        else:
            print("FAILED")

            # drone turns red
            p.changeVisualShape(drone,-1,rgbaColor=[1,0,0,1])

            status = "FAILED"

        p.addUserDebugText(
            status,
            [0,0,9],
            textSize=2,
            textColorRGB=[1,1,0],
            lifeTime=2
        )

        time.sleep(1.5)

        # reset colors
        p.changeVisualShape(goal,-1,rgbaColor=[0,1,0,1])
        p.changeVisualShape(drone,-1,rgbaColor=[0,0,1,1])

        # reset environment
        obs, _ = env.reset()
        episode_count += 1

        create_obstacles()
