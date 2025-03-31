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
torch.autograd.set_detect_anomaly=True
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
os.environ["HF_HOME"] = "/home/ec2-user/SageMaker/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/ec2-user/SageMaker/huggingface_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/home/ec2-user/SageMaker/huggingface_cache/datasets"
os.environ["TORCH_HOME"] = "/home/ec2-user/SageMaker/huggingface_cache/torch"
os.environ["TMPDIR"] = "/home/ec2-user/SageMaker/tmp"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
os.environ["TORCH_NCCL_TIMEOUT_MS"] = "12000000"
os.environ["TORCH_TIMEOUT_MS"] = "12000000"
os.environ["NCCL_TIMEOUT"] = "12000000"
os.environ["TORCH_NCCL_TIMEOUT"] = "12000000"
os.environ["NCCL_P2P_LEVEL"] = "NVL"
os.environ["NCCL_P2P_DISABLE"] = "0"
accelerator = Accelerator(mixed_precision='bf16', kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])#, dataloader_config=dataloader_config) #,gradient_accumulation_steps=2,

epoch = 1
model_name = "path/to/your/model"


lora_config = LoraConfig(
    r=128,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head"]
)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,low_cpu_mem_usage=True, attn_implementation="flash_attention_2") 


tokenizer = AutoTokenizer.from_pretrained(model_name) 
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side='left'
#model = get_peft_model(model, peft_config=lora_config) #if you plan to use LoRA

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
)

reward_model = AutoModelForSequenceClassification.from_pretrained(
    'path/to/your/reward/model',
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config, 
    attn_implementation="flash_attention_2",
)
reward_tokenizer = AutoTokenizer.from_pretrained('path/to/your/reward/model', torch_dtype=torch.bfloat16)

#torch._dynamo.config.suppress_errors = True

    
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
                 max_new_tokens=800
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
        #self.alpha = torch.zeros((100,100), requires_grad=True)
        self.alpha = torch.zeros((10000,1), requires_grad=True)
        self.optimizer = torch.optim.AdamW([
                    {'params': self.model.parameters()},
                    {'params': self.alpha, 'lr':1e-1}
                    
                ], lr=self.learning_rate) #{'params': self.alpha, 'lr':1e-2}
        
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
        self.run = wandb.init(project="GFlowNetSTEP", group="gfn0")

    def calculate_reward(self, query, answer):
        text_table = wandb.Table(columns=["prompt", "answer", "reward"])
        list_steps = [step for step in answer.split('\n') if len(step) > 3]
        to_evaluate = [query + ''.join(list_steps[:i]) + '<next>' + list_steps[i].strip() for i in range(len(list_steps))]
        
        if len(to_evaluate) <=1:
            return -99, False
        tokenized_inputs = self.reward_tokenizer.batch_encode_plus(to_evaluate, return_tensors="pt", padding='longest').to(self.reward_model.device)
        with torch.inference_mode():
                outputs = self.reward_model(**tokenized_inputs)
                rewards = torch.sigmoid(outputs.logits).cpu()
                text_table.add_data(query, answer, torch.mean(rewards).item())
        self.run.log({"training_samples": text_table})
        del tokenized_inputs, outputs
        torch.cuda.empty_cache()
        return rewards, True

    def sample(self):
        return self.ReplayBuffer.sample_weighted_and_remove(self.batch_size)
        #return self.ReplayBuffer.sample_weighted(self.batch_size)

    def save_model(self, path):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        #unwrapped_model = unwrapped_model.merge_and_unload()
        unwrapped_model.save_pretrained(path, is_main_process=self.accelerator.is_main_process, save_function=self.accelerator.save)
        self.tokenizer.save_pretrained(path)
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
            if torch.isnan(sum_prop):
                print(ABS)
                print(mean_probs)
                print(list_indices)
                print('=========================================')
            
            diff_probs_F = cumulative_probs_F.unsqueeze(1) - cumulative_probs_F.unsqueeze(0)
            diff_rewards = cumulative_rewards.unsqueeze(1) - cumulative_rewards.unsqueeze(0)
            diff_eos = eos_probs.unsqueeze(1) - eos_probs.unsqueeze(0)

            mask = torch.triu(torch.ones_like(diff_probs_F), diagonal=1)

            loss_terms = (- diff_rewards + diff_probs_F + diff_eos) ** 2 #torch.log(alpha) 
            weighted_loss = mask * (self.subTB_lambda ** torch.abs(torch.arange(len(loss_terms)).unsqueeze(1).to(self.accelerator.device) - torch.arange(len(loss_terms)).to(self.accelerator.device)))
            loss = loss + (weighted_loss * loss_terms).sum() / weighted_loss.sum()
            
            sum_rewards += torch.mean(step_rewards)
            sum_mean_reward += mean_reward
 
        if self.accelerator.is_main_process:
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
        if self.accelerator.is_main_process:
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
        with torch.inference_mode():
            generated_sequences = self.model.module.generate(inputs['input_ids'],
                                                              attention_mask=inputs['attention_mask'],
                                                              max_new_tokens=self.max_new_tokens,
                                                              do_sample=True,
                                                              #temperature=self.temperature,
                                                              #top_p=self.top_p,
                                                              #top_k=self.top_k,
                                                              eos_token_id=self.eos_token_id,
                                                              pad_token_id=self.tokenizer.pad_token_id,
                                                              #length_penalty=2.0,
                                                              )
        
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
            ).split('step.')[5].split('assistant')[0].strip()
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
        del seq, prompt,inputs
        torch.cuda.empty_cache()
            
        mean_rewards = torch.mean(mean_rewards)
        mean_rewards_gathered = self.accelerator.gather([mean_rewards])
        if self.accelerator.is_main_process:
            self.run.log({"mean reward current": torch.mean(mean_rewards_gathered[0]).item()})
        return mean_rewards

    
    
    

def extract_answer_gsm8k(answer):
    return float(answer.split('####')[1].replace(',',''))

def extract_answer_nvidia(answer):
    match = re.search(r"\boxed\{(\d+)\}", answer)
    return match.group(1)

def mini_train(choice,batch_size):
    if choice == 0:
        dataset = load_dataset('openai/gsm8k', 'main', split='train')#.select(i for i in range(5000))
        column = 'question'
        response = 'answer'
    else:
        column = 'problem'
        response = 'expected_answer'
        dataset = load_dataset('nvidia/OpenMathInstruct-2',split='train_1M', cache_dir="~/.cache/huggingface/datasets/").select([i for i in range(700000)])
        def filter_dataset(example):
            messages = [
                    {"role": "user", "content": "You will answer this question and start a new line after each single step. Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}"},
                    {"role": "assistant", "content": "The expressions inside each square root must be non-negative. \n Therefore, $x-2 \\ge 0$, so $x\\ge2$. \n And $5 - x \\ge 0$, so $x \\le 5$. \n Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. \n Therefore, the domain of the expression is $\\boxed{[2,5)}$. \n Final Answer: The final answer is $[2,5)$."},
                    {"role": "user", "content": "You will answer this question and start a new line after each single step. If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$"},
                    {"role": "assistant", "content": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B})$. \n So, $(2)(12) = \\boxed{24}$. \n Final Answer: The final answer is $24$."},
                    {"role": "user", "content": "You will answer this question and start a new line after each single step. Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?"},
                    {"role": "assistant", "content": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. \n If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. \n Equating this to 480 pounds, we can solve for $n$: \n \\begin{align*} \n 30n&=480 \n \\\Rightarrow\\qquad n&=480/30=\\boxed{16} \n \\end{align*} \n Final Answer: The final answer is $16$."},
                    {"role": "user", "content": "You will answer this question and start a new line after each single step. If the system of equations \n \\begin{align*} \n 6x-4y&=a, \n 6y-9x &=b. \n \\end{align*} has a solution $(x, y)$ where $x$ and $y$ are both nonzero, \n find $\\frac{a}{b},$ assuming $b$ is nonzero."},
                    {"role": "assistant", "content": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain \n \\begin{align*} \n 6y-9x&=-\\frac{3}{2}a. \n \\end{align*} \n Since we also know that $6y-9x=b$, we have \n \\begin{align*} \n -\\frac{3}{2}a&=b \n \\\Rightarrow\\frac{a}{b}&=\\boxed{-\\frac{2}{3}}. \n \\end{align*} \n Final Answer: The final answer is $-\\frac{2}{3}$."},
                    {'role': 'user', 'content': 'You will answer this question and start a new line after each single step. '+example[column]}]

            tokenized_inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            return tokenized_inputs.shape[1] < 1000 and example["problem_source"] ==  "augmented_math"
        dataset = dataset.filter(filter_dataset)
        dataset=dataset.select([i for i in range(100000)])
       

    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True) 

    gfn = GFlowNet(model=model, tokenizer=tokenizer, reward_model=reward_model, dataloader=dataloader, reward_tokenizer=reward_tokenizer, accelerator=accelerator)


    for iteration in range(epoch):
        k=0
        for batch in tqdm(gfn.dataloader): =
            with accelerator.accumulate(gfn.model): #for grad acc
                #{"role": "system", "content": "You MUST answer questions and start a new line after each single step."}, #if you want to use system prompt
                
                messages = [[   
            
            {"role": "user", "content": "You will answer this question and start a new line after each single step. Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}"},
            {"role": "assistant", "content": "The expressions inside each square root must be non-negative. \n Therefore, $x-2 \\ge 0$, so $x\\ge2$. \n And $5 - x \\ge 0$, so $x \\le 5$. \n Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. \n Therefore, the domain of the expression is $\\boxed{[2,5)}$. \n Final Answer: The final answer is $[2,5)$."},
            {"role": "user", "content": "You will answer this question and start a new line after each single step. If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$"},
            {"role": "assistant", "content": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B})$. \n So, $(2)(12) = \\boxed{24}$. \n Final Answer: The final answer is $24$."},
            {"role": "user", "content": "You will answer this question and start a new line after each single step. Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?"},
            {"role": "assistant", "content": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. \n If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. \n Equating this to 480 pounds, we can solve for $n$: \n \\begin{align*} \n 30n&=480 \n \\\Rightarrow\\qquad n&=480/30=\\boxed{16} \n \\end{align*} \n Final Answer: The final answer is $16$."},
            {"role": "user", "content": "You will answer this question and start a new line after each single step. If the system of equations \n \\begin{align*} \n 6x-4y&=a, \n 6y-9x &=b. \n \\end{align*} has a solution $(x, y)$ where $x$ and $y$ are both nonzero, \n find $\\frac{a}{b},$ assuming $b$ is nonzero."},
            {"role": "assistant", "content": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain \n \\begin{align*} \n 6y-9x&=-\\frac{3}{2}a. \n \\end{align*} \n Since we also know that $6y-9x=b$, we have \n \\begin{align*} \n -\\frac{3}{2}a&=b \n \\\Rightarrow\\frac{a}{b}&=\\boxed{-\\frac{2}{3}}. \n \\end{align*} \n Final Answer: The final answer is $-\\frac{2}{3}$."},
            {'role': 'user', 'content': 'You will answer this question and start a new line after each single step. '+text}
                ] for text in batch[column]]*gfn.number_generation 
                ground_truth_answers = []
                for text in batch[response]:
                    if choice == 0:
                        ground_truth_answers.append(extract_answer_gsm8k(text))
                    else:
                        ground_truth_answers.append(0)
                        
                inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
                tokenized_inputs = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding='max_length', max_length=1010, padding_side='left')

                if 'token_type_ids' in tokenized_inputs: #for falcon models
                    tokenized_inputs.pop('token_type_ids')

                gfn.model.module.gradient_checkpointing_disable()
                gfn.generate(tokenized_inputs, ground_truth_answers)
                gfn.model.module.gradient_checkpointing_enable()

                loss, reward = gfn.step()
                k+=1
                if k%200 ==0:
                    accelerator.print(f'iteration numero {k}')
                    #gfn.save_model(f'./falcon3bGFN0vv0inter{k}', end=False)




        accelerator.print(f'===========end epoch {iteration + 1}============')

    gfn.save_model('./save_model')

mini_train(choice=1,batch_size=32)


#accelerate launch GFlowNets_FineTuning.py
