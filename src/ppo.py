import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import GPT, GPTConfig

# todo's : implement attention mask in gpt and use gpt that i coded instead / LLMEnvironment reset is linked to adding new data / ppo features in paper yet uncovered / training and validation / hellaswag
# possible error : reward might not be correctly implemented

class ValueNetwork(nn.Module):
    def __init__(self, pretrained_model):
        super(ValueNetwork, self).__init__()
        self.model = pretrained_model  
        self.value_head = nn.Linear(self.model.config.n_embd, 1)  

    def forward(self, input_ids): #(Batch, seq_len)
        outputs = self.model(input_ids) 
        hidden_states = outputs["last_hidden_state"] #(Batch, seq_len, n_embd)
        state_value = self.value_head(hidden_states[:, -1, :])  # Use the last token's hidden state
        
        return state_value.squeeze() #(Batch)

class LLMEnvironment:
    def __init__(self, better_model, device, max_tokens=50):
        """
        Initialize the LLM environment
        
        Args:
            better_model: The stronger LLM model used to generate next tokens and probabilities
            max_tokens: Maximum sequence length before episode terminates
        """
        self.better_model = better_model.to(device)
        self.max_tokens = max_tokens
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state with a new starting token"""
        from transformers import AutoTokenizer
        import torch
        enc = AutoTokenizer.from_pretrained("gpt2")
        tokens = enc.encode("Hello, I'm a language model,", add_special_tokens=True)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to("cuda")  # (1, seq_len)
        self.num_steps = 0
  
        return tokens
    
    def step(self, state, action_token):
        """
        Take a step in the environment by adding a token
        
        Args:
            state: Current state (1, seq_len) 
            action_tokens: The token chosen by the agent (1,)
            
        Returns:
            next_state: Updated token sequence
            reward: probability of the chosen token according to better model
            done: Whether episode has terminated
        """
        self.num_steps += 1

        # Calculate reward as probability
        next_token_probs = F.softmax(self.better_model(state).logits, dim=-1) #(1, 9, 50257)
        next_token_probs_last_pos = next_token_probs[0, -1, :]  # Shape: (50257,)
        next_token = torch.multinomial(next_token_probs_last_pos, 1)  # Returns a tensor with shape (1,)
        next_token = next_token.unsqueeze(0) #(1, 1)
        next_state = torch.cat((state, next_token), dim=1) #(1, 8)

        reward = next_token_probs_last_pos[action_token]
        
        # Check if we're done
        done = self.num_steps >= self.max_tokens 
        
        # Can add additional done conditions like:
        # - Better model gives very low probability
        # - Generated invalid sequence
        # Can't stop and eos_token since we use that for padding
        
        return next_state, reward, done

class PPOAgent:
    def __init__(self, device, learning_rate=0.0003, discount_factor=0.99, clip_epsilon=0.2, update_epochs=10, batch_size=64):      
        self.device = device
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        # Policy Network
        self.policy_network = GPT.from_pretrained('gpt2').to(self.device)

        # Value Network
        self.value_network = ValueNetwork(GPT.from_pretrained('gpt2')).to(self.device)

        self.policy_optimizer = optim.AdamW(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.AdamW(self.value_network.parameters(), lr=learning_rate)
        
    def get_action(self, state):  # (1, seq_len)
        with torch.no_grad():
            action_probs = self.policy_network(state)["probs"]  # (1, seq_len, vocab_size)
            action_probs = action_probs.squeeze(0)  # (seq_len, vocab_size)
            action = torch.multinomial(action_probs, num_samples=1)  # (seq_len, 1)
            return action.squeeze(-1)[-1]  # (1,)
        
    def get_value(self, state):
         with torch.no_grad():
            state_tensor = torch.FloatTensor(self.one_hot_encode(state)).to(self.device)
            value = self.value_network(state_tensor)
            return value.item()
    
    def train(self, state_tensors, attention_mask, action_tensors, reward_tensors):
        
        # Calculate discounted rewards for the entire episode
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(reward_tensors):
            cumulative_reward = r + self.discount_factor * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        discounted_rewards = torch.tensor(discounted_rewards).to(self.device) #(B,)

        # Precompute old probabilities and values (detached from computation graph)
        with torch.no_grad():
            old_action_probs = self.policy_network(state_tensors, attention_mask=attention_mask)["probs"].gather(2, action_tensors.unsqueeze(-1)).squeeze(-1) #(B, seq_len)

            old_values = self.value_network(state_tensors, attention_mask=attention_mask) #(B,)
            advantages = discounted_rewards - old_values #(B,)
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) #(B,)

        for _ in range(self.update_epochs):
            # Shuffle data
            indices = np.arange(len(state_tensors))
            np.random.shuffle(indices)

            for start in range(0, len(state_tensors), self.batch_size):
                end = min(start + self.batch_size, len(state_tensors))
                batch_indices = indices[start:end]
                
                batch_state_tensors = state_tensors[batch_indices] #(b,)
                batch_attention_mask = attention_mask[batch_indices]
                batch_action_tensors = action_tensors[batch_indices] #(b, seq_len)
                batch_advantages = advantages[batch_indices] #(b,)
                batch_discounted_rewards = discounted_rewards[batch_indices] #(b,)
                batch_old_action_probs = old_action_probs[batch_indices] #(b, seq_len)

                # Policy loss
                new_action_probs = self.policy_network(batch_state_tensors, attention_mask=batch_attention_mask).gather(2, batch_action_tensors.unsqueeze(-1)).squeeze(-1) #(b, seq_len)
                ratios = new_action_probs / (batch_old_action_probs + 1e-8) #(b, seq_len)
                
                clipped_ratios = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) #(b, seq_len)
                policy_loss = -torch.min(ratios * batch_advantages, clipped_ratios * batch_advantages).mean()
                #old_action_probs and advantages aren't being updated, only the new_action_probs are
                #advantages>0 means an action was better than expected
                #ratios>0 means the new policy is increasing the probability of that action
                #thus we need to maximize ratios * advantages 

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # Value loss
                values = self.value_network(batch_state_tensors, attention_mask=batch_attention_mask) #(b,)
                value_loss = F.mse_loss(values, batch_discounted_rewards)

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
