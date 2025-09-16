import torch
from torch.nn.utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, StoppingCriteriaList, StoppingCriteria, BitsAndBytesConfig
from accelerate import Accelerator, DistributedDataParallelKwargs
import wandb
import re
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from utils import ReplayBuffer, get_newline_token_id
from skywork.model_utils.prm_model import PRM_MODEL
from skywork.model_utils.io_utils import prepare_input, prepare_batch_input_for_model, derive_step_rewards


# Note: Accelerator is provided by caller to avoid duplicate process groups and extra memory.
class GFlowNet:
    def __init__(self,
                 model,
                 tokenizer,
                 dataloader,
                 reward_model,
                 reward_tokenizer,
                 accelerator,
                 learning_rate=3e-6,
                 number_generation=2,
                 subTB_lambda=1.0,
                 temperature=0.6,
                 top_p=0.95,
                 top_k=999,
                 max_new_tokens=200
                ):
        self.ReplayBuffer = ReplayBuffer(1000)
        self.subTB_lambda = subTB_lambda
        self.number_generation = number_generation
        self.model = model
        self.dataloader = dataloader
        self.model.train()
        self.model.bfloat16()
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.learning_rate = learning_rate
        self.newline_token_id = get_newline_token_id(self.tokenizer)
        self.eos_token_id = tokenizer.eos_token_id
        self.alpha = torch.zeros((10000,1), requires_grad=True)
        self.optimizer = torch.optim.AdamW([
                    {'params': self.model.parameters()},
                    {'params': self.alpha, 'lr':1e-1}
                    
                ], lr=self.learning_rate) 
        
        self.num_steps = len(self.dataloader) #// 2
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_steps, eta_min=1e-8)
        self.accelerator = accelerator
        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.dataloader, self.scheduler)
        self.model._set_static_graph()
        self.temperature = temperature
        self.top_p = 1.0 #top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.compteur = 1
        self.avg_loss = 0
        self.avg_reward = 0
        self.stop_words_ids = [torch.tensor(newline_id) for newline_id in get_newline_token_id(self.tokenizer)]
        
        # Only initialize wandb on the main process (rank 0) to avoid duplicate runs
        if self.accelerator.is_main_process:
            self.run = wandb.init(project="GFlowNetSTEP", group="gfn0")
        else:
            self.run = None

    def calculate_reward(self, query, answer):
        processed_data = [prepare_input(query, answer, tokenizer=self.reward_tokenizer, step_token="\n")]
        input_ids, steps, reward_flags = zip(*processed_data)

        input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(input_ids, reward_flags, self.reward_tokenizer.pad_token_id)
        # Move to PRM device
        prm_device = next(self.reward_model.parameters()).device
        input_ids = input_ids.to(prm_device)
        attention_mask = attention_mask.to(prm_device)
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            _, _, rewards = self.reward_model(input_ids=input_ids, attention_mask=attention_mask, return_probs=True)
        # _, _, rewards = self.reward_model(input_ids=input_ids, attention_mask=attention_mask, return_probs=True)
        step_rewards = derive_step_rewards(rewards, reward_flags)
        # put random rewards between 0 and 1 
        # step_rewards = torch.rand(len(reward_flags)).unsqueeze(0)
        step_rewards = torch.tensor(step_rewards)

        # Only log to wandb on the main process
        if self.accelerator.is_main_process and self.run is not None:
            text_table = wandb.Table(columns=["prompt", "answer", "reward"])
            text_table.add_data(query, answer, torch.mean(step_rewards).item())
            self.run.log({"training_samples": text_table})
        
        del input_ids, reward_flags, steps, rewards
        torch.cuda.empty_cache()
        return step_rewards, True

    def sample(self):
        return self.ReplayBuffer.sample_weighted_and_remove(self.batch_size)
        #return self.ReplayBuffer.sample_weighted(self.batch_size)

    def save_model(self, path):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        #unwrapped_model = unwrapped_model.merge_and_unload()
        unwrapped_model.save_pretrained(path, is_main_process=self.accelerator.is_main_process, save_function=self.accelerator.save)
        self.tokenizer.save_pretrained(path)
        if self.accelerator.is_main_process and self.run is not None:
            self.run.finish()
        self.accelerator.wait_for_everyone()

    def calculate_loss_from_replay(self):
        sum_lambda = torch.tensor(0.0, device=self.accelerator.device)
        sum_rewards = torch.tensor(0.0, device=self.accelerator.device)
        loss = torch.tensor(0.0, requires_grad=True, device=self.accelerator.device)
        sum_mean_reward = torch.tensor(0.0, device=self.accelerator.device)
        sampled = self.sample()
        sum_confidence = 0
        sum_prop = 0
        
        

        for element in sampled:
            seq = element['generated_tokens'].to(self.accelerator.device)
            seq = seq[:-1]
            question = element['question'].to(self.accelerator.device)
            step_rewards = torch.tensor([element[0] for element in element['step_rewards']]).to(self.accelerator.device)
            mean_reward = element['mean_reward']
            input_ids = torch.cat((question, seq[:-1])).unsqueeze(0)

            outputs = self.model(input_ids=input_ids, return_dict=True, output_hidden_states=False) 
            tempered_logits = outputs.logits / self.temperature
            log_probs = tempered_logits.log_softmax(dim=-1)[:, question.size(0):, :]
            transition_probs = [
                log_probs[0, i, token].exp() for i, token in enumerate(seq[1:])
            ]
            transition_probs = torch.stack(transition_probs)
            
            eos_probs = torch.tensor([
                log_probs[0, i, self.eos_token_id] for i in range(len(seq[1:]))
            ], device=self.accelerator.device)
            
            matches = (seq.unsqueeze(1) == torch.tensor(self.newline_token_id, device=seq.device)).nonzero(as_tuple=True)[0]
            step_indices = matches[torch.cat([torch.tensor([True], device=matches.device), matches[1:] != matches[:-1] + 1])]

            list_indices=[(0, step_indices[0])]+[(step_indices[i-1]+1,step_indices[i]) for i,indice in enumerate(step_indices) if i>0]
            if step_indices[-1]+1<len(transition_probs):
                list_indices+=[(step_indices[-1]+1,len(transition_probs)+1)]
            
            
            step_rewards = step_rewards[:min(len(step_rewards), len(list_indices))]
            list_indices = list_indices[:min(len(step_rewards), len(list_indices))]
            

            L = [torch.mean(element) for element in transition_probs]
            sum_confidence += torch.mean(torch.tensor(L))
            
            
            cumulative_rewards = torch.zeros_like(transition_probs)
            cumulative_probs_F = torch.cumsum(torch.log(transition_probs), dim=0)
            mean_probs = []
            for k, (start, end) in enumerate(list_indices):
                cumulative_rewards[start:end + 1] = torch.log(step_rewards[k])
                mean_probs.append(torch.mean(transition_probs[start:end + 1]))
            cumulative_rewards = torch.cumsum(cumulative_rewards, dim=0)
            ABS = []
            for prob, rew in zip(mean_probs, step_rewards):
                ABS.append(abs(prob.item()-rew.item()))
                
            sum_prop += torch.mean(torch.tensor(ABS, device= self.accelerator.device))
            
            diff_probs_F = cumulative_probs_F.unsqueeze(1) - cumulative_probs_F.unsqueeze(0)
            diff_rewards = cumulative_rewards.unsqueeze(1) - cumulative_rewards.unsqueeze(0)
            diff_eos = eos_probs.unsqueeze(1) - eos_probs.unsqueeze(0)

            mask = torch.triu(torch.ones_like(diff_probs_F), diagonal=1)

            loss_terms = (- diff_rewards + diff_probs_F + diff_eos) ** 2 #torch.log(alpha) 
            weighted_loss = mask * (self.subTB_lambda ** torch.abs(torch.arange(len(loss_terms)).unsqueeze(1).to(self.accelerator.device) - torch.arange(len(loss_terms)).to(self.accelerator.device)))
            loss = loss + (weighted_loss * loss_terms).sum() / weighted_loss.sum()
            
            sum_rewards += torch.mean(step_rewards)
            sum_mean_reward += mean_reward
 
        if self.accelerator.is_main_process and self.run is not None:
            self.run.log({"Confidence": sum_confidence/ self.batch_size})
            self.run.log({"Prop": sum_prop/ self.batch_size})
        
        avg_rewards = sum_rewards / self.batch_size
        avg_mean_reward = sum_mean_reward / self.batch_size
        loss = loss / self.batch_size
        return loss, avg_rewards, avg_mean_reward, sampled

    
        
        
    def step(self):
        torch.cuda.empty_cache()
        loss, reward, mean_reward, sampled = self.calculate_loss_from_replay()
      
        loss = loss.to(self.accelerator.device)
        self.accelerator.backward(loss)
        loss_gathered = self.accelerator.gather([loss])
        reward_gathered = self.accelerator.gather([torch.tensor(reward).clone().detach()])
        mean_reward_gathered = self.accelerator.gather([torch.tensor(mean_reward).clone().detach()])
        grad_norm = torch.mean(torch.tensor([torch.norm(param.grad).item() for param in self.model.parameters() if param.grad is not None]))
        if self.accelerator.is_main_process and self.run is not None:
            self.run.log({"loss": torch.mean(loss_gathered[0])})
            self.run.log({"average reward": torch.mean(reward_gathered[0])})
            self.run.log({f"grad_norm": grad_norm})

        max_grad_norm = 5.0
        self.accelerator.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()
       
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.accelerator.print(f"Training loss: {loss.item()} ; Reward: {reward:.2f} ; Grad norm: {grad_norm} ; Replay buffer size: {len(self.ReplayBuffer.buffer)} ; alpha : {self.alpha.sum()} learning_rate : {self.scheduler.get_last_lr()[0]}") # ; alpha : {self.alpha.sum()}
        self.accelerator.wait_for_everyone()
        return loss.item(), reward

    def generate(self, inputs, ground_truth_answers):
        mean_rewards = torch.tensor([], device=self.accelerator.device)
        self.batch_size = len(inputs['input_ids']) // self.number_generation
        inputs = inputs.to(self.accelerator.device)
        generated_sequences_chunks = []
        micro_bs = 2
        with torch.inference_mode():
            for start in range(0, inputs['input_ids'].size(0), micro_bs):
                end = min(start + micro_bs, inputs['input_ids'].size(0))
                out = self.model.module.generate(
                    inputs['input_ids'][start:end],
                    attention_mask=inputs['attention_mask'][start:end],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=False,
                )
                generated_sequences_chunks.append(out)
        generated_sequences = torch.cat(generated_sequences_chunks, dim=0)

        for i in range(inputs['input_ids'].shape[0]):
            seq = generated_sequences[i].squeeze()
            prompt = seq[:len(inputs['input_ids'][i])]
            seq = seq[len(inputs['input_ids'][i]):]
            for j in range(len(seq)):
                if seq[j] == self.tokenizer.eos_token_id:
                    seq = seq[:j+1]
                    break
            for j in range(len(prompt)):
                if prompt[j] != self.tokenizer.eos_token_id:
                    prompt = prompt[j:]
                    break
            finished = False
            if self.tokenizer.eos_token_id in seq:
                finished = True
            question = self.tokenizer.decode(
                inputs['input_ids'][i], skip_special_tokens=True
            )
            newline_token_ids_tensor = torch.tensor(self.newline_token_id,device=self.accelerator.device)

            valid = torch.any(torch.isin(seq, newline_token_ids_tensor))
            corr = "I cannot generate a question" #avoid corrupted data
            answer = self.tokenizer.decode(seq, skip_special_tokens=True)
            
            if len(answer) > 10 and corr not in question and valid:               
                step_rewards,validity = self.calculate_reward(
                    question, answer
                )
                if validity:
                    mean_reward = torch.mean(step_rewards).unsqueeze(0)
                    mean_rewards = torch.cat((mean_rewards, mean_reward.to(self.accelerator.device)))
                    self.ReplayBuffer.add(
                        prompt, seq, step_rewards, mean_reward.item(), 10
                    )
                else:
                    print(answer)
        del seq, prompt, inputs, generated_sequences, generated_sequences_chunks
        torch.cuda.empty_cache()
            
        mean_rewards = torch.mean(mean_rewards)
        mean_rewards_gathered = self.accelerator.gather([mean_rewards])
        if self.accelerator.is_main_process and self.run is not None:
            self.run.log({"mean reward current": torch.mean(mean_rewards_gathered[0]).item()})
        return mean_rewards
