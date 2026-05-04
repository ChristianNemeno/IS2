import torch
import gymnasium as gym
import ale_py
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
import os

torch.backends.cudnn.benchmark = True

os.makedirs('./checkpoints', exist_ok=True)

n_envs = 8
env = make_atari_env("ALE/Pong-v5", n_envs=n_envs, seed=0)
env = VecFrameStack(env, n_stack=4)

CHECKPOINT_PATH = "./checkpoints/a2c_pong_24000000_steps"
STEPS_ALREADY_DONE = 24_000_000
total_timesteps = 26_000_000
remaining_timesteps = total_timesteps - STEPS_ALREADY_DONE  
# ─────────────────────────────────────────────────────────────────────────────

# model = A2C(
#     "CnnPolicy",
#     env,
#     verbose=1,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     n_steps=5,
#     learning_rate=7e-4,
#     ent_coef=0.01,
#     vf_coef=0.25,
#     max_grad_norm=0.5,
#     gamma=0.99,
#     gae_lambda=1.0,
#     normalize_advantage=False,
#     policy_kwargs=dict(
#         optimizer_class=RMSpropTFLike,
#         optimizer_kwargs=dict(eps=1e-5),
#     ),
# )

model = A2C.load(
    CHECKPOINT_PATH,
    env=env,
    device="cuda",
)
model.num_timesteps = STEPS_ALREADY_DONE

#saved_params = {k: v.clone() for k, v in model.policy.state_dict().items()}

model.n_steps = 16
model.normalize_advantage = True
model.vf_coef = 0.5
model.ent_coef = 0.01
model.max_grad_norm = 0.5
model.learning_rate = 1e-4
model.lr_schedule = lambda _: 1e-4

#model._setup_model()

#model.policy.load_state_dict(saved_params)

assert model.normalize_advantage == True
assert model.learning_rate == 1e-4
print(f"✓ normalize_advantage: {model.normalize_advantage}")
print(f"✓ learning_rate: {model.learning_rate}")

print("=" * 55)
print("TRAINING CONFIG  (resumed from checkpoint)")
print("=" * 55)
print(f"  Checkpoint       : {CHECKPOINT_PATH}")
print(f"  Steps done       : {STEPS_ALREADY_DONE:,}")
print(f"  Remaining steps  : {remaining_timesteps:,}")
print(f"  Algorithm        : {type(model).__name__}")
print(f"  Policy           : {model.policy.__class__.__name__}")
print(f"  Device           : {model.device}")
print(f"  Total timesteps  : {total_timesteps:,}")
print(f"  Learning rate    : {model.learning_rate}")
print(f"  Optimizer        : RMSpropTFLike (eps=1e-5)")
print(f"  n_steps          : {model.n_steps}")
print(f"  Effective batch  : {model.n_steps * n_envs} samples/update")
print(f"  ent_coef         : {model.ent_coef}")
print(f"  vf_coef          : {model.vf_coef}")
print(f"  max_grad_norm    : {model.max_grad_norm}")
print(f"  gamma            : {model.gamma}")
print(f"  gae_lambda       : {model.gae_lambda}")
print(f"  Environment      : ALE/Pong-v5 (n_envs={n_envs}, frame_stack=4)")
print("=" * 55)

checkpoint_callback = CheckpointCallback(
    save_freq=1_000_000 // n_envs,
    save_path="./checkpoints/",
    name_prefix="a2c_pong",
)

# model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
model.learn(total_timesteps=remaining_timesteps, 
            callback=checkpoint_callback,
            reset_num_timesteps=False,
            )
model.save("a2c_pong_model_final")
print("Testing the model...")
obs = env.reset()
total_reward = 0
for _ in range(500_000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    total_reward += rewards.sum()
    env.render("human")

print(f"Total reward over test run: {total_reward:.1f}")