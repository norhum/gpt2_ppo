import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
from model import GPT

class ValueNetwork(nn.Module):
    def __init__(self, pretrained_model):
        super(ValueNetwork, self).__init__()
        self.model = pretrained_model  
        self.value_head = nn.Linear(self.model.config.n_embd, 1)  

    def forward(self, input_ids, attention_mask=None): #(Batch, seq_len)
        outputs = self.model(input_ids, attention_mask) 
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

    def reset(self, i):
        """Reset the environment to initial state with a new starting token"""
        enc = AutoTokenizer.from_pretrained("gpt2")
        with open(r"data/sentences.txt", "r") as f:
            sentences = f.readlines()
        tokens = enc.encode(sentences[i][:-1], add_special_tokens=True)
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
        # Can't stop for eos_token since we use that for padding
        
        return next_state, reward, done

class PPOAgent:
    def __init__(self, device, learning_rate=0.0003, discount_factor=0.99, clip_epsilon=0.2, update_epochs=5, batch_size=64, kl_beta=0.1, kl_epsilon=1e-8):      
        self.device = device
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.kl_beta = kl_beta
        self.kl_epsilon = kl_epsilon

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
            last_token_indices = (attention_mask != 1).to(torch.long).argmax(dim=1)
            last_token_indices_list = last_token_indices.tolist()
            last_token_indices_list = [last_token_indices_list[0] - 1] + last_token_indices_list[:-1]
            last_token_indices = torch.tensor(last_token_indices_list) #([B])
 
            inter = self.policy_network(state_tensors, attention_mask=attention_mask)["probs"] #([B, seq_len, vocab_size])
            inter = inter[torch.arange(inter.size(0)), last_token_indices] #([B, vocab_size])

            old_action_probs = inter.gather(1, action_tensors.unsqueeze(-1)).squeeze(-1) #(B,)

            old_values = self.value_network(state_tensors, attention_mask=attention_mask) #(B,)
            advantages = discounted_rewards - old_values #(B,)
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) #(B,)

        kl_losses = []
        preclipped_ratios = []
        policy_losses = []
        value_losses = []

        for epoch in range(self.update_epochs):

            # Shuffle data
            indices = np.arange(len(state_tensors))
            np.random.shuffle(indices)

            kl_loss_epoch = 0
            preclipped_ratio_epoch = 0
            policy_loss_epoch = 0
            value_loss_epoch = 0
            n = 0

            for start in range(0, len(state_tensors), self.batch_size):
        
                end = min(start + self.batch_size, len(state_tensors))
                batch_indices = indices[start:end]
                
                batch_state_tensors = state_tensors[batch_indices] #(b, seq_len)
                batch_attention_mask = attention_mask[batch_indices] #(b, seq_len)
                batch_last_token_indices = last_token_indices[batch_indices] #(b,)
                batch_action_tensors = action_tensors[batch_indices] #(b,)
                batch_advantages = advantages[batch_indices] #(b,)
                batch_discounted_rewards = discounted_rewards[batch_indices] #(b,)
                batch_old_action_probs = old_action_probs[batch_indices] #(b,)

                # Policy loss
                inter = self.policy_network(batch_state_tensors, attention_mask=batch_attention_mask)["probs"] #([b, seq_len, vocab_size])
                inter = inter[torch.arange(inter.size(0)), batch_last_token_indices] #([b, vocab_size])
                new_action_probs = inter.gather(1, batch_action_tensors.unsqueeze(-1)).squeeze(-1) #(b,)
                ratios = new_action_probs / (batch_old_action_probs + 1e-8) #(b,)

                # Compute the KL divergence element-wise for each action in the batch
                batch_old_action_probs = torch.clamp(batch_old_action_probs, min=self.kl_epsilon, max=1.0)
                new_action_probs = torch.clamp(new_action_probs, min=self.kl_epsilon, max=1.0)
                kl_loss = torch.sum(batch_old_action_probs * torch.log(batch_old_action_probs / new_action_probs), dim=0).mean()
                kl_loss_epoch += kl_loss

                clipped_ratios = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) #(b,)
                preclipped_ratio = ratios.mean()
                preclipped_ratio_epoch += preclipped_ratio
                policy_loss = -torch.min(ratios * batch_advantages, clipped_ratios * batch_advantages).mean() + self.kl_beta * kl_loss
                policy_loss_epoch += policy_loss
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
                value_loss_epoch += value_loss

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

                n += 1
            
            print(f"epoch: {epoch}, kl_loss: {kl_loss_epoch/n:.4f}, preclipped_ratio: {preclipped_ratio_epoch/n:.4f}, policy_loss: {policy_loss_epoch/n:.4f}, value_loss: {value_loss_epoch/n:.4f}")
 
            kl_losses.append(kl_loss_epoch/n)
            preclipped_ratios.append(preclipped_ratio_epoch/n)
            policy_losses.append(policy_loss_epoch/n)
            value_losses.append(value_loss_epoch/n)
        
        return {
            "kl_losses" : [loss.item() for loss in kl_losses],
            "preclipped_ratios" : [loss.item() for loss in preclipped_ratios],
            "policy_losses" : [loss.item() for loss in policy_losses],
            "value_losses" : [loss.item() for loss in value_losses]
        }
