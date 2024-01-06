import gymnasium as gym
from gymnasium.utils.play import play

# 创建CartPole环境
env = gym.make("CartPole-v1", render_mode="rgb_array")

# 使用 play 函数运行环境
play(env, keys_to_action={"a": 0,"d": 1})