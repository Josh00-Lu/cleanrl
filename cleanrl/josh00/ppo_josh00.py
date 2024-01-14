'''
trick1: batch norm for advantage
trick2: value function clipping
trick3: entropy loss
trick4: gradient clip
trick5: orthogonal Initialization
trick6: trick6: Adam epsilon
'''

num_iterations = int(5e5) ## 总迭代次数
enroll_length = 128 ## 环境交互步数
num_env = 4 ## 环境个数
gamma = 0.99
gae_lambda = 0.95
batch_size = 128
num_epoches = 4
clip_coef = 0.2 ## epsilon for PPO
entropy_coef = 0.01
max_grad_norm = 0.5

learning_rate = 1e-2

# trick6: Adam epsilon
Adam_eps = 1e-5

import gymnasium as gym
import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
import torch.optim as opt

def make_env(env_id, idx, capture_video, run_name):
    if capture_video and idx == 0:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"myvideos/{run_name}")
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

envs = gym.vector.SyncVectorEnv(
    [lambda: make_env("CartPole-v1", i, True, "Demo") for i in range(num_env)],
)

# trick5: Orthogonal Initialization
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, envs) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.prod(envs.single_observation_space.shape), 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1))
        )
        self.policy = nn.Sequential(
            layer_init(nn.Linear(np.prod(envs.single_observation_space.shape), 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, envs.single_action_space.n))
        )
    
    def forward(self, obs, action=None):
        ## 值函数
        value = self.critic(obs)
        ## 输出概率
        logits = self.policy(obs)
        
        ## 从Policy中进行动作采样
        props = Categorical(logits=logits)
        
        if action is None:
            action = props.sample()
        
        return value.view(-1), action, props.log_prob(action), props.entropy()
    

states = torch.zeros((enroll_length, num_env, )+envs.single_observation_space.shape) ## s_t
actions = torch.zeros((enroll_length, num_env)) ## a_t
log_props = torch.zeros((enroll_length, num_env)) ## p(a_t|s_t)
rewards = torch.zeros((enroll_length, num_env)) ## r_t
values = torch.zeros((enroll_length, num_env)) ## V(s_t)
next_dones = torch.zeros((enroll_length, num_env)) 

advantages = torch.zeros((enroll_length, num_env)) ## A_t
TD_targets = torch.zeros((enroll_length, num_env))

Agent = ActorCritic(envs)
state, _ = envs.reset(seed=0)
state = torch.Tensor(state)

## 定义优化器
optimizer = opt.Adam(Agent.parameters(), lr=learning_rate, eps=Adam_eps)

for iteration in range(num_iterations//batch_size//num_env):
    for step in range(enroll_length):
        ## s_t
        states[step] = state
        ## a_t
        with torch.no_grad():
            value, action, log_prop, _ = Agent(state)
        ## next state
        state, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
        state = torch.Tensor(state)
        ## collect traj
        actions[step], log_props[step], rewards[step], values[step] = map(lambda x: torch.Tensor(x), [action, log_prop, reward, value])
        
        next_dones[step] = torch.Tensor(np.logical_or(terminated, truncated))
        
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"episodic_return={info['episode']['r']}")
        
    for t in reversed(range(enroll_length)):
        ## V(s_(t+1))
        with torch.no_grad():
            if t == enroll_length - 1:
                next_value, _ , _ , _ = Agent(state)
            else:
                next_value = values[t+1]
        
            delta = rewards[t] + (1 - next_dones[t]) * gamma * next_value - values[t]
            if t == enroll_length - 1:
                advantages[t] = delta
            else:
                advantages[t] = delta + (1 - next_dones[t]) * gamma * gae_lambda * advantages[t+1]
        
        TD_targets = advantages + values
            
    b_states = states.view((-1,)+envs.single_observation_space.shape)
    b_actions = actions.view(-1)
    b_log_props = log_props.view(-1)
    b_advantages = advantages.view(-1)
    b_TD_targets = TD_targets.view(-1)
    b_values = values.view(-1)
    
    ## trick1: batch norm for advantage
    b_advantages_norm = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
    
    b_index = np.arange(enroll_length * num_env)
    
    for epoch in range(num_epoches): 
        np.random.shuffle(b_index)
        for start in range(0, b_actions.shape[0], batch_size):
            end = start + batch_size
            b_range = b_index[start:end]
            
            ## policy loss
            value, _, current_log_prop, entropy = Agent(b_states[b_range], b_actions[b_range])
            ratio = torch.exp(current_log_prop - b_log_props[b_range]) ## pi_new / pi_old
            
            pg_loss1 = ratio * b_advantages_norm[b_range]
            pg_loss2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * b_advantages_norm[b_range]
            pg_loss = -torch.min(pg_loss1, pg_loss2).mean()
            
            ## value loss
            ## trick2: value function clipping
            value_loss1 = (value - b_TD_targets[b_range]) ** 2
            value_loss2 = (torch.clamp(value, b_values[b_range] - clip_coef, b_values[b_range] + clip_coef) - b_TD_targets[b_range]) ** 2
            value_loss = torch.max(value_loss1, value_loss2).mean()
            
            ## trick3: entropy loss
            entropy_loss = entropy.mean()
            
            ## total loss
            loss = pg_loss + value_loss - entropy_coef * entropy_loss
                        
            optimizer.zero_grad()
            loss.backward()
            # trick 4: gradient clip
            nn.utils.clip_grad_norm_(Agent.parameters(), max_grad_norm)
            optimizer.step()
        