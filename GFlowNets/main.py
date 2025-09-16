import argparse
import os
import re
import torch
from torch.nn.utils import *
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from GFlowNets import GFlowNet
from skywork.model_utils.prm_model import PRM_MODEL

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def set_environment_variables():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    # os.environ["HF_HOME"] = "/home/ec2-user/SageMaker/huggingface_cache"
    # os.environ["TRANSFORMERS_CACHE"] = "/home/ec2-user/SageMaker/huggingface_cache/transformers"
    # os.environ["HF_DATASETS_CACHE"] = "/home/ec2-user/SageMaker/huggingface_cache/datasets"
    # os.environ["TORCH_HOME"] = "/home/ec2-user/SageMaker/huggingface_cache/torch"
    # os.environ["TMPDIR"] = "/home/ec2-user/SageMaker/tmp"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
    os.environ["TORCH_NCCL_TIMEOUT_MS"] = "12000000"
    os.environ["TORCH_TIMEOUT_MS"] = "12000000"
    os.environ["NCCL_TIMEOUT"] = "12000000"
    os.environ["TORCH_NCCL_TIMEOUT"] = "12000000"
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    os.environ["NCCL_P2P_DISABLE"] = "0"


def extract_answer_gsm8k(answer):
    return float(answer.split("####")[1].replace(",", ""))


def extract_answer_nvidia(answer):
    match = re.search(r"\\boxed\{(\d+)\}", answer)
    return match.group(1)


def load_model_and_tokenizer(model_name, reward_model_name, accelerator):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    reward_model = PRM_MODEL.from_pretrained(reward_model_name, torch_dtype=torch.bfloat16).eval()
    reward_model = reward_model.to(accelerator.device)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name, trust_remote_code=True)
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.padding_side = "left"

    return model, tokenizer, reward_model, reward_tokenizer


def mini_train(choice, batch_size, epoch, model_name, reward_model_name, saved_model_path):
    accelerator = Accelerator(
        mixed_precision="bf16", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    model, tokenizer, reward_model, reward_tokenizer = load_model_and_tokenizer(model_name, reward_model_name, accelerator)

    if choice == 0:
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        column = "question"
        response = "answer"
    else:
        column = "problem"
        response = "expected_answer"
        dataset = load_dataset(
            "nvidia/OpenMathInstruct-2", split="train_1M", cache_dir="~/.cache/huggingface/datasets/"
        ).select([i for i in range(70000)])

        def filter_dataset(example):
            tokenized_inputs = tokenizer.apply_chat_template(
                [   {"role": "user", "content": "You will answer this question and start a new line after each single step. Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}"},
                    {"role": "assistant", "content": "The expressions inside each square root must be non-negative. \n Therefore, $x-2 \\ge 0$, so $x\\ge2$. \n And $5 - x \\ge 0$, so $x \\le 5$. \n Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. \n Therefore, the domain of the expression is $\\boxed{[2,5)}$. \n Final Answer: The final answer is $[2,5)$."},
                    {"role": "user", "content": "You will answer this question and start a new line after each single step. If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$"},
                    {"role": "assistant", "content": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B})$. \n So, $(2)(12) = \\boxed{24}$. \n Final Answer: The final answer is $24$."},
                    {"role": "user", "content": "You will answer this question and start a new line after each single step. Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?"},
                    {"role": "assistant", "content": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. \n If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. \n Equating this to 480 pounds, we can solve for $n$: \n \\begin{align*} \n 30n&=480 \n \\\Rightarrow\\qquad n&=480/30=\\boxed{16} \n \\end{align*} \n Final Answer: The final answer is $16$."},
                    {"role": "user", "content": "You will answer this question and start a new line after each single step. If the system of equations \n \\begin{align*} \n 6x-4y&=a, \n 6y-9x &=b. \n \\end{align*} has a solution $(x, y)$ where $x$ and $y$ are both nonzero, \n find $\\frac{a}{b},$ assuming $b$ is nonzero."},
                    {"role": "assistant", "content": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain \n \\begin{align*} \n 6y-9x&=-\\frac{3}{2}a. \n \\end{align*} \n Since we also know that $6y-9x=b$, we have \n \\begin{align*} \n -\\frac{3}{2}a&=b \n \\\Rightarrow\\frac{a}{b}&=\\boxed{-\\frac{2}{3}}. \n \\end{align*} \n Final Answer: The final answer is $-\\frac{2}{3}$."},
                    {"role": "user", "content": example[column]}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            return tokenized_inputs.shape[1] < 1000 and example["problem_source"] == "augmented_math"

        dataset = dataset.filter(filter_dataset) #train 100k examples

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
    gfn = GFlowNet(
        model=model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        dataloader=dataloader,
        reward_tokenizer=reward_tokenizer,
        accelerator=accelerator,
    )

    for iteration in range(epoch):
        for k, batch in enumerate(tqdm(gfn.dataloader), start=1):
            with accelerator.accumulate(gfn.model):
                messages = [
                    text for text in batch[column]
                ] * gfn.number_generation
                ground_truth_answers = [
                    extract_answer_gsm8k(text) if choice == 0 else text for text in batch[response]
                ]

                tokenized_inputs = tokenizer(
                    messages, return_tensors="pt", padding=True, max_length=1010
                )

                if "token_type_ids" in tokenized_inputs: #needed for some models
                    tokenized_inputs.pop("token_type_ids")

                # gfn.model.module.gradient_checkpointing_disable()
                gfn.generate(tokenized_inputs, ground_truth_answers)
                # gfn.model.module.gradient_checkpointing_enable()

                loss, reward = gfn.step()

                if k % 200 == 0:
                    accelerator.print(f"Iteration {k}")

        accelerator.print(f"=========== End of epoch {iteration + 1} ===========")

    gfn.save_model(saved_model_path)
    accelerator.print("Model saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Train a GFlowNet model.")
    parser.add_argument("--choice", type=int, default=1, help="Dataset choice: 0 for gsm8k, 1 for nvidia openmath")
    parser.add_argument("--batch_size", type=int, default=18, help="Batch size for training")
    parser.add_argument("--epoch", type=int, default=1, help="Number of epochs")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the model")
    parser.add_argument("--reward_model_name", type=str, required=True, help="Path to the reward model")
    parser.add_argument("--saved_model_path", type=str, default="./save_model", help="Path to the trained model")


    args = parser.parse_args()

    set_environment_variables()
    mini_train(args.choice, args.batch_size, args.epoch, args.model_name, args.reward_model_name, args.saved_model_path)


if __name__ == "__main__":
    main()