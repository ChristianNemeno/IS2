import gymnasium as gym

import ale_py # Required for Gymnasium 1.0+

from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_atari_env

from stable_baselines3.common.vec_env import VecFrameStack

from stable_baselines3.common.callbacks import CheckpointCallback


# 1. Create the Atari environment with standard preprocessing

# This handles the resizing to 84x84 and grayscale conversion

env = make_atari_env("ALE/Pong-v5", n_envs=8, seed=0)


# 2. Stack 4 frames so the model can see movement (trajectory)

# This changes the shape to the (4, 84, 84) the model expects

env = VecFrameStack(env, n_stack=4)


# 3. Initialize the model with CnnPolicy for visual processing

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    device="cuda",
    n_steps=512,
    ent_coef=0.01,       
    learning_rate=1e-4,  
)


# 4. Train the agent

total_timesteps = 10_000_000
print("=" * 50)
print("TRAINING CONFIG")
print("=" * 50)
print(f"  Algorithm      : {type(model).__name__}")
print(f"  Policy         : {model.policy.__class__.__name__}")
print(f"  Device         : {model.device}")
print(f"  Total timesteps: {total_timesteps:,}")
print(f"  Learning rate  : {model.learning_rate}")
print(f"  Optimizer      : {type(model.policy.optimizer).__name__}")
print(f"  n_steps        : {model.n_steps}")
print(f"  batch_size     : {model.batch_size}")
print(f"  n_epochs       : {model.n_epochs}")
print(f"  gamma          : {model.gamma}")
print(f"  Environment    : ALE/Pong-v5 (n_envs=8, frame_stack=4)")
print("=" * 50)

checkpoint_callback = CheckpointCallback(
    save_freq=1_000_000,
    save_path="./checkpoints/",
    name_prefix="ppo_pong_v2",
)

model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)


# 5. Save the model

model.save("ppo_pong_model_v2_10m")


# 6. Test/Render the trained model

# Note: For rendering, we need to use a single environment with human mode

print("Testing the model...")

obs = env.reset()

for _ in range(500000):

    action, _states = model.predict(obs)

    obs, rewards, dones, info = env.step(action)

    env.render("human")
