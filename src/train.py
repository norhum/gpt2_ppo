import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
from huggingface_hub import login
from ppo import LLMEnvironment, PPOAgent

hf_token = os.getenv('HF_TOKEN')
login(hf_token)

# Load environment tokenizer and model
env_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
env_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
# Load agent tokenizer 
agent_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_ppo_agent(episodes=1000):
    env = LLMEnvironment(env_model)
    agent = PPOAgent(state_size=env.size, device=device)

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_states, episode_actions, episode_rewards = [], [], []
        total_reward = 0

        while not done:
            action = agent.get_action(state) #(1, seq_len) -> (seq_len)
            # action -> tokens -> text -> tokens using the env tokenizer
            pre_env_action = agent_tokenizer.decode(action, skip_special_tokens=True)
            env_action = env_tokenizer.encode(pre_env_action, return_tensors="pt")

            next_state, reward, done = env.step(env_action)

            #next_state_tokens -> text -> tokens using the agent tokenizer
            pre_next_state = env_tokenizer.decode(next_state, skip_special_tokens=True)
            next_state = agent_tokenizer.encode(pre_next_state, return_tensors="pt") #(1, seq_len)
            
            episode_states.append(state) #(B, seq_len) tokens
            episode_actions.append(action) #(B, seq_len) index for the next token
            episode_rewards.append(reward) #(B,)

            # Convert lists to tensors with desired shapes
            episode_states = torch.stack(episode_states).to(device)  # (B, seq_len)
            episode_actions = torch.stack(episode_actions).to(device)  # (B, seq_len)
            episode_rewards = torch.tensor(episode_rewards).to(device)  # (B,)
        
            state = next_state
            total_reward += reward

        agent.train(episode_states, episode_actions, episode_rewards)
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}")

    return agent

# --------------------------------------------------------------------------------

if __name__ == '__main__':
    device = device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training Proximal Policy Optimization Agent:")
    trained_agent = train_ppo_agent(episodes=500)
