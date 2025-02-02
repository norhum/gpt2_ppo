import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
from huggingface_hub import login
from ppo import LLMEnvironment, PPOAgent

hf_token = os.getenv('HF_TOKEN')
login(hf_token)

# Load environment tokenizer and model
# env_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
# env_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
env_tokenizer = AutoTokenizer.from_pretrained("gpt2")
env_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load agent tokenizer 
agent_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_ppo_agent(episodes=1000):
    env = LLMEnvironment(env_model, device=device)
    agent = PPOAgent(device=device)

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_states, episode_actions, episode_rewards = [], [], []
        total_reward = 0

        while not done:
            action = agent.get_action(state) #(1, seq_len) -> (1,)

            next_state, reward, done = env.step(state, action)

            episode_states.append(state) #(B, seq_len + k), where k is increasing (e.g., 0, 1, 2, ...)
            episode_actions.append(action) #(B,) index for the next token
            episode_rewards.append(reward) #(B,)

            state = next_state
            total_reward += reward

        # Convert lists to tensors with desired shapes
        episode_states = torch.stack(episode_states).to(device)  # (B, seq_len + k), where k is increasing (e.g., 0, 1, 2, ...)
        episode_actions = torch.stack(episode_actions).to(device)  # (B,)
        episode_rewards = torch.tensor(episode_rewards).to(device)  # (B,)

        agent.train(episode_states, episode_actions, episode_rewards)
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}")

    return agent

# --------------------------------------------------------------------------------

if __name__ == '__main__':
    device = device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training Proximal Policy Optimization Agent:")
    trained_agent = train_ppo_agent(episodes=500)
