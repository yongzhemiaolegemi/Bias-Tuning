import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset,concatenate_datasets
import os
import math
from tqdm import tqdm
import pickle
from peft import get_peft_model, LoraConfig
from transformers import get_cosine_schedule_with_warmup
import argparse
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from peft import LoraConfig, get_peft_model
from peft import PeftModel

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_custom_model(model, tokenizer, model_path):
    torch.save(model, f"{model_path}.pth")
    tokenizer.save_pretrained(model_path)

def get_parent_module(model, module_name):
    module_name_parts = module_name.split('.')
    parent = model
    for part in module_name_parts[:-1]:
        parent = getattr(parent, part)
    return parent

def load_custom_model(model_path):
    # 从指定路径加载模型
    torch.load(model_path)

def initialize_adam_state(model, rank):
    m_A = {}
    v_A = {}
    m_B = {}
    v_B = {}
    return m_A, v_A, m_B, v_B

class CustomLinearLayer(nn.Module):
    def __init__(self, original_linear, name, device_id, world_size, ranks_per_gpu=None):
        super(CustomLinearLayer, self).__init__()
        self.name = name
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # 获取原始权重并转换为 float32 以进行 SVD
        self.register_buffer('W', original_linear.weight.data.clone().detach())
        
        rank = ranks_per_gpu
        self.A = nn.Parameter(torch.zeros(rank, self.in_features)) 
        self.B = nn.Parameter(torch.randn(self.out_features, rank))
        for param in self.parameters():
            param.requires_grad = False

        # 设置 A 和 B 为可训练
        self.A.requires_grad = True
        self.B.requires_grad = True

        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.clone().detach())
        else:
            self.bias = None
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # 计算低秩偏差矩
        #print("x shape",x.shape)
        x_avg = x.mean(dim=0)
        #print("x avg",x_avg.shape)
        result = torch.matmul(x_avg.t(), x_avg)
        #print("result",result.shape)
        hidden = nn.functional.linear(result, self.A)
        hidden = self.relu(hidden)
        #print("hidden",hidden.shape)
        bias_matrix = nn.functional.linear(hidden, self.B)
        #print("bias matrix",bias_matrix.shape)
        #print("W",self.W.shape) 
        # 将偏差矩阵加到原始权重矩阵上
        weight_with_bias = self.W + bias_matrix.t()
        
        # 计算输出
        return nn.functional.linear(x, weight_with_bias, self.bias)

    
    def __repr__(self):
        return (f"CustomLinearLayer(name={self.name}, "
                f"in_features={self.in_features}, out_features={self.out_features})")

def replace_with_custom_layer(model, target_modules, rank, world_size, ranks_per_gpu=None):
    for name, module in model.named_modules():
        for target_name in target_modules:
            if target_name in name and isinstance(module, nn.Linear):
                parent_module = get_parent_module(model, name)
                setattr(parent_module, name.split('.')[-1], CustomLinearLayer(module, name, rank, world_size, ranks_per_gpu))
                break


def lora_fine_tune(model, ds_list, batch_size,num_epochs, rank, world_size,accumulation_steps, output_path):
    # 配置 LoRA 参数
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module = module.to(torch.float32)
    # 包装模型为 DDP 模型
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    loss_list = []
    for ds in ds_list:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(
            ds,
            drop_last=True,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        total_steps = num_epochs * len(dataloader)//accumulation_steps
        warmup_steps = int(0.03 * total_steps)
        current_step = 1
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        if rank == 0:
            print(f"Start LoRA fine-tuning for {num_epochs} epochs.")
        for epoch in range(num_epochs):
            dataloader.sampler.set_epoch(epoch)
            accumulated_loss = 0
            for i,batch in enumerate(dataloader):

                input_ids = batch['input_ids'].cuda(rank, non_blocking=True)
                attention_mask = batch['attention_mask'].cuda(rank, non_blocking=True)
                labels = batch['labels'].cuda(rank, non_blocking=True)

                batch_mean = input_ids.float().mean().item()
            
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss = loss / accumulation_steps  # 缩放 Loss 以进行梯度累积

            # 同步所有 GPU 上的 Loss
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / world_size  # 计算平均 Loss

            # 累积 Loss 用于日志记录
                accumulated_loss += loss.item()
                loss.backward()
                if (i+1)% accumulation_steps == 0:
                    loss_list.append(accumulated_loss)
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Batch Mean: {batch_mean:.6f}, Learning Rate: {current_lr:.8f}")
                    if rank==0:
                        os.makedirs(output_path, exist_ok=True)
                        with open(f'{output_path}/lr.txt','a') as f:
                            f.write(f"Batch Mean: {batch_mean:.6f}, Learning Rate: {current_lr:.8f}\n")
                        with open(f'{output_path}/loss.txt','a') as file:
                            file.write(f'Step:{current_step} Loss:{accumulated_loss}\n')
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    current_step += 1
                
                    accumulated_loss = 0
                
                if current_step % 10 == 0 and rank == 0 and  (i+1) % accumulation_steps ==0:
                    print(f"Step {current_step}/{total_steps} completed, remaining: {total_steps - current_step} steps.")
                    print(f"GPU {rank} processing step {current_step}, Loss: {loss_list[-1]}")
            
                if current_step % 500 == 0 and rank == 0 and  (i+1) % accumulation_steps ==0:
                    model_path = os.path.join(output_path, f"saved_model_step_{current_step}")
                    ensure_dir(os.path.dirname(model_path))
                    model.module.save_pretrained(model_path)
                    print(f"Model saved at step {current_step}")
            if (i+1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            if rank == 0:
                print(f"Epoch {epoch + 1} completed.")

                model_path = os.path.join(output_path, f"saved_model_step_{current_step}")
                ensure_dir(os.path.dirname(model_path))
                model.module.save_pretrained(model_path)
                print(f"Model saved at step {current_step}")

        # 保存损失列表
    if rank == 0:
        loss_list_path = os.path.join(output_path, "loss_list.pkl")
        with open(loss_list_path, 'wb') as f:
            pickle.dump(loss_list, f)

import math

def get_cosine_lr(step, total_steps, warmup_steps, lr_initial, lr_min=0.0):
    if step < warmup_steps:
        # 线性预热
        return lr_initial * step / warmup_steps
    elif step < total_steps:
        # 余弦衰减
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_initial - lr_min) * (1 + math.cos(math.pi * progress))
    else:
        return lr_min


import datetime
import time


def main(rank, world_size, model_path, output_path,training_mode='distributed',  ranks_per_gpu=16,batch_size=16,accumulation_steps=1,samedata=False,num_epochs=1,bf16=False):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if bf16==True:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    model = model.cuda(rank)

    # 加载数据集
    ds = load_dataset("meta-math/MetaMathQA",split="train").select(range(100000))
    ds2 = load_dataset("fxmeng/CodeFeedback-Python105K",split="train")
    # 定义筛选函数
    ds3 = load_dataset("fxmeng/WizardLM_evol_instruct_V2_143k",split="train")
    def filter_function(example):
        # # 排除类型为 "MATH" 的数据
        if "MATH" in example["type"]:
            return False
        return True

    # 对训练集进行筛选
    filtered_ds = ds.filter(filter_function)
    filtered_ds = concatenate_datasets([filtered_ds,ds2,ds3])
    # 训练集的 token 化
    def preprocess(examples):
        # 拼接 query 和 response 后进行整体 tokenization
        if "response" in examples and "query" in examples:
            concatenated_inputs = [
                PROMPT.format(instruction=q) + f"{a}\n{tokenizer.eos_token}"
                for q, a in zip(examples["query"], examples["response"])
            ]
        elif "query" in examples and "output" in examples:
            concatenated_inputs = [
                PROMPT.format(instruction=q) + f"{a}\n{tokenizer.eos_token}"
                for q, a in zip(examples["query"], examples["output"])
            ]
        elif "human" in examples and "assistant" in examples:
            concatenated_inputs = [
                PROMPT.format(instruction=q) + f"{a}\n{tokenizer.eos_token}"
                for q, a in zip(examples["human"], examples["assistant"])
            ]
        else:
            raise KeyError("Unsupport feature name!")


        tokenized = tokenizer(
            concatenated_inputs,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        # ?~H~[建 labels?~L?~F?~S?~E??~C??~H~F?| ~G记为 IGNORE_INDEX
        labels = tokenized['input_ids'].clone()
        if "query" in examples:
            input_ids_lens = [len(tokenizer(PROMPT.format(instruction=q))['input_ids']) for q in examples["query"]]
        elif "human" in examples:
            input_ids_lens = [len(tokenizer(PROMPT.format(instruction=q))['input_ids']) for q in examples["human"]]
        for label, source_len in zip(labels, input_ids_lens):
            label[:source_len] = -100  # 忽?~U??~S?~E??~C??~H~F
        pad_mask = tokenized['input_ids'] == tokenizer.pad_token_id


        first_pad_indices = pad_mask.int().argmax(dim=1)  # ?~N??~O~V?~O?~L第?~@个 pad_token ?~Z~D?~M置

        # 对?~I~@?~\~I?~M置?~[?~L mask?~L?~F第?~@个 pad_token ?~M置?~N~R?~Y?
        for i, first_pad_idx in enumerate(first_pad_indices):
            pad_mask[i, first_pad_idx] = False  # ?~N~R?~Y?第?~@个 pad_token ?~Z~D?~M置

        # ?~F?~I??~Y?~Z~D pad_token ?~Z~D labels 设置为 -100
        labels[pad_mask] = -100

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }
    # 测试集的 token 化
    def tokenize_function_test(examples):
        # 仅对 query 进行编码
        inputs = tokenizer(examples["question"], truncation=True, max_length=512)
        # 将 response 保存下来，供后续评估使用
        outputs = tokenizer(examples["answer"], truncation=True, max_length=512)
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': outputs['input_ids']
        }

    # 应用预处理函数
    tokenized_datasets = filtered_ds.map(preprocess, batched=True, remove_columns=filtered_ds.column_names)
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    #ds1 = filtered_ds.map(preprocess, batched=True, remove_columns=filtered_ds.column_names)
    #ds1.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    #ds2 = ds2.map(preprocess, batched=True, remove_columns=ds2.column_names)
    #ds2.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    #ds3 = ds3.map(preprocess, batched=True, remove_columns=ds3.column_names)
    #ds3.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    ds_list = [tokenized_datasets]
    def filter_invalid_labels(example):
        # 只保留那些标签不是全部为 -100 的样本
        return not all(label == -100 for label in example['labels'])

    tokenized_datasets = tokenized_datasets.filter(filter_invalid_labels)
    #for ds in ds_list:
    #    ds = ds.filter(filter_invalid_labels)
    ds_list = [tokenized_datasets]
    ##test first batch loss###
    dataloader = DataLoader(
                tokenized_datasets,
                drop_last=True,
                batch_size=1,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
    # 获取第一个 batch
    batch = next(iter(dataloader))

    # 确定设备
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # 将模型移动到设备上
    model = model.to(device)

    accumulation_steps = accumulation_steps//world_size
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if training_mode=="bias_tuning":
        sampler = DistributedSampler(tokenized_datasets, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(
                tokenized_datasets,
                drop_last=True,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True
            )
        target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
        replace_with_custom_layer(model, target_modules, rank, world_size, ranks_per_gpu=ranks_per_gpu)
        #model.print_trainable_parameters()
        for name, module in model.named_modules():
            if 'norm' in name or 'gate' in name:
                module = module.to(torch.float32)
        model = model.to(device)
        # 包装模型为 DDP 模型
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        loss_list = [] 
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=2e-5)
        total_steps = num_epochs * len(dataloader)//accumulation_steps
        warmup_steps = int(0.03 * total_steps)
        current_step = 1
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        if rank == 0:
            print(f"Start pissa fine-tuning for {num_epochs} epochs.")
        for epoch in range(num_epochs):
            dataloader.sampler.set_epoch(epoch)
            accumulated_loss = 0
            for i,batch in enumerate(dataloader):

                input_ids = batch['input_ids'].cuda(rank, non_blocking=True)
                attention_mask = batch['attention_mask'].cuda(rank, non_blocking=True)
                labels = batch['labels'].cuda(rank, non_blocking=True)

                batch_mean = input_ids.float().mean().item()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss = loss / accumulation_steps  # 缩放 Loss 以进行梯度累积

                # 同步所有 GPU 上的 Loss
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / world_size  # 计算平均 Loss

                # 累积 Loss 用于日志记录
                accumulated_loss += loss.item()
                loss.backward()
                if (i+1)% accumulation_steps == 0:
                    loss_list.append(accumulated_loss)
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Batch Mean: {batch_mean:.6f}, Learning Rate: {current_lr:.8f}")
                    if rank==0:
                        os.makedirs(output_path, exist_ok=True)
                        with open(f'{output_path}/lr.txt','a') as f:
                            f.write(f"Batch Mean: {batch_mean:.6f}, Learning Rate: {current_lr:.8f}\n")

                        with open(f'{output_path}/loss.txt','a') as file:
                            file.write(f'Step:{current_step} Loss:{accumulated_loss}\n')
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    current_step += 1
                    accumulated_loss = 0

                if current_step % 10 == 0 and rank == 0 and  (i+1) % accumulation_steps ==0:
                    print(f"Step {current_step}/{total_steps} completed, remaining: {total_steps - current_step} steps.")
                    print(f"GPU {rank} processing step {current_step}, Loss: {loss_list[-1]}")
                
                if current_step % 500 == 0 and rank == 0 and  (i+1) % accumulation_steps ==0:
                    model_path = os.path.join(output_path, f"saved_model_step_{current_step}")
                    ensure_dir(os.path.dirname(model_path))
                    save_custom_model(model.module, tokenizer, model_path)
                    print(f"Model saved at step {current_step}")
            if (i+1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            if rank == 0:
                print(f"Epoch {epoch + 1} completed.")

                model_path = os.path.join(output_path, f"saved_model_step_{current_step}")
                ensure_dir(os.path.dirname(model_path))
                print(f"Model saved at step {current_step}")
                save_custom_model(model.module, tokenizer, model_path)
            # 保存损失列表
        if rank == 0:
            loss_list_path = os.path.join(output_path, "loss_list.pkl")
            with open(loss_list_path, 'wb') as f:
                pickle.dump(loss_list, f)

    elif training_mode == 'lora_finetune':
        #sampler = DistributedSampler(tokenized_datasets, num_replicas=world_size, rank=rank, shuffle=False)
        #dataloader = DataLoader(
        #        tokenized_datasets,
        #        drop_last=True,
        #        batch_size=batch_size,
        #        sampler=sampler,
        #        num_workers=4,
        #        pin_memory=True
        #)
        lora_fine_tune(model, ds_list,batch_size, num_epochs, rank, world_size,accumulation_steps, output_path)

    dist.destroy_process_group()

import socket
def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        _, port = s.getsockname()
    return str(port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', help='Model Path')
    parser.add_argument('--output_path', type=str, default='./output', help='Output Path')
    parser.add_argument('--world_size', type=int, default=4, help='Number of GPUs')
    parser.add_argument('--training_mode', type=str, default='bias_tuning', help='Training mode')
    parser.add_argument('--ranks_per_gpu', type=int, default=16, help='Ranks per GPU')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Accumulation steps')
    parser.add_argument('--samedata', type=bool, default=False, help='Same or different data on GPUs')
    parser.add_argument('--num_epochs', type=int, default=1, help='Training epochs')
    parser.add_argument('--bf16',type=bool,default=False,help='model dtype')
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = find_free_port()

    processes = []
    for rank in range(args.world_size):
        p = Process(target=main, args=(
            rank, args.world_size, args.model_path, args.output_path, args.training_mode,
             args.ranks_per_gpu, args.batch_size,
            args.accumulation_steps, args.samedata, args.num_epochs,args.bf16
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
