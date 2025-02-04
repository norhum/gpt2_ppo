import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from ppo import LLMEnvironment, PPOAgent
from hellaswag import iterate_examples, render_example, get_most_likely_row
import json

# Load environment model
# env_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
env_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load tokenizer 
tokenizer = AutoTokenizer.from_pretrained('gpt2')

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_ppo_agent(episodes=1000):
    env = LLMEnvironment(env_model, device=device)
    agent = PPOAgent(device=device)

    for episode in range(episodes):
        start_time = time.time()
        state = env.reset(episode)
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

        max_len = max(state.size(1) for state in episode_states)  # Find the max sequence length (seq_len)
        tokenizer.pad_token = tokenizer.eos_token

        padded_episode_states = []
        for state in episode_states:
            padding_size = max_len - state.size(1)  
            if padding_size > 0:
                # Pad with eos_token 
                padded_state = F.pad(state, (0, padding_size), value=tokenizer.pad_token_id)
                padded_episode_states.append(padded_state)
            else:
                padded_episode_states.append(state)

        # Step 3: Stack the tensors (now they all have the same length)
        episode_states = torch.stack(padded_episode_states).squeeze().to(device)  # (B, max_len)
        episode_actions = torch.stack(episode_actions).to(device)  # (B,)
        episode_rewards = torch.tensor(episode_rewards).to(device)  # (B,)

        # Create padding mask based on eos_token
        eos_token_id = tokenizer.eos_token_id
        attention_mask = (episode_states != eos_token_id).long().squeeze()  # (B, max_len) 1 for non-padding tokens, 0 for padding tokens 

        data = agent.train(episode_states, attention_mask, episode_actions, episode_rewards) # this is where it takes the most time and resource

        if episode % 50 == 0:
            print(f"episode {episode}: Total Reward = {total_reward:.6f}")

        if episode % 10 == 0:
            torch.save(agent.policy_network.state_dict(), f"models/policy_network_weights_{episode}.pth")
            torch.save(agent.policy_optimizer.state_dict(), f"models/policy_optimizer_{episode}.pth")  
            torch.save(agent.value_network.state_dict(), f"models/value_network_weights_{episode}.pth")
            torch.save(agent.value_optimizer.state_dict(), f"models/value_optimizer_{episode}.pth")  
            with open(f"logs/data_{episode}.json", "w") as f:
                json.dump(data, f, indent=4)

        end_time = time.time()
        print(f"episode: {episode}, took {end_time - start_time:.4f} seconds")
        print()

        # once in a while evaluate hellaswag and log the data instead of returning it at the end. both to the logs folder
        if episode % 50 == 0:
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits = agent.policy_network(tokens)["logits"]
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)

            # reduce the stats across all processes
            num_total = int(num_total)  # Just ensure it's an integer
            num_correct_norm = int(num_correct_norm)  # Convert to int if needed

            acc_norm = num_correct_norm / num_total
    
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(r"logs/hellaswag", "a") as f:
                f.write(f"{episode} hellaswag {acc_norm:.4f}\n")

        # once in a while generate from the model (except step 0, which is noise)
        
    return agent

# --------------------------------------------------------------------------------

if __name__ == '__main__':
    device = device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training Proximal Policy Optimization Agent:")
    print(device)
    trained_agent = train_ppo_agent(episodes=2)
