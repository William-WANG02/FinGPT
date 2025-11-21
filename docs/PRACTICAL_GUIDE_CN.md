# FinGPT 实践指南

**⚠️ 重要提醒**: 
- 本文档仅供学习和研究使用
- 文档中的预测和分析示例不构成投资建议
- 实际投资决策应基于全面研究和专业咨询
- 妥善保管API密钥等敏感信息，切勿提交到公开仓库

---

## 📋 目录
1. [环境配置](#环境配置)
2. [快速开始](#快速开始)
3. [模型训练实践](#模型训练实践)
4. [模型推理实践](#模型推理实践)
5. [常见问题解决](#常见问题解决)
6. [进阶实践](#进阶实践)

---

## 环境配置

### 系统要求

#### 最低配置
```
- OS: Ubuntu 20.04+ / Windows 10+ (with WSL2) / macOS 12+
- Python: 3.8+
- GPU: NVIDIA GPU with 6GB+ VRAM (GTX 1660, RTX 3060等)
- RAM: 16GB+
- Storage: 50GB+ (用于模型和数据)
```

#### 推荐配置
```
- OS: Ubuntu 22.04
- Python: 3.10
- GPU: RTX 3090 (24GB) / A100 (40GB)
- RAM: 32GB+
- Storage: 100GB+ NVMe SSD
```

### 安装步骤

#### 1. 克隆仓库

```bash
# 克隆FinGPT仓库
git clone https://github.com/AI4Finance-Foundation/FinGPT.git
cd FinGPT
```

#### 2. 创建Python环境

**使用conda（推荐）**:
```bash
# 创建新环境
conda create -n fingpt python=3.10
conda activate fingpt

# 安装PyTorch (根据你的CUDA版本选择)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**使用venv**:
```bash
# 创建虚拟环境
python -m venv fingpt_env
source fingpt_env/bin/activate  # Linux/Mac
# 或
fingpt_env\Scripts\activate  # Windows

# 安装PyTorch
pip install torch torchvision torchaudio
```

#### 3. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装额外依赖（根据需要）
pip install transformers==4.35.0
pip install peft==0.7.0
pip install bitsandbytes==0.41.3  # 用于量化
pip install accelerate==0.25.0
pip install datasets==2.15.0
pip install gradio==4.8.0  # 用于Web界面
pip install wandb  # 可选：实验跟踪
```

#### 4. 验证安装

```python
# 创建test.py文件
import torch
import transformers
import peft

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Transformers version: {transformers.__version__}")
print(f"PEFT version: {peft.__version__}")

# 运行
python test.py
```

**预期输出**:
```
PyTorch version: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 3090
Transformers version: 4.35.0
PEFT version: 0.7.0
```

### HuggingFace配置

#### 设置访问令牌

某些模型需要HuggingFace账号：

```bash
# 1. 访问 https://huggingface.co/settings/tokens
# 2. 创建新token（read权限）
# 3. 登录
huggingface-cli login
# 输入你的token

# 或者设置环境变量
export HUGGINGFACE_TOKEN="hf_xxxxx"
```

#### 配置镜像（可选，国内用户）

```bash
# 使用国内镜像加速下载
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 快速开始

### Demo 1: 情感分析推理

这是最简单的入门demo。

#### 步骤1: 创建推理脚本

创建文件 `sentiment_demo.py`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from peft import PeftModel
import torch

print("Loading model...")

# 1. 加载基础模型（8-bit量化）
base_model = LlamaForCausalLM.from_pretrained(
    "NousResearch/Llama-2-13b-hf",
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)

# 2. 加载LoRA适配器
model = PeftModel.from_pretrained(
    base_model,
    "FinGPT/fingpt-sentiment_llama2-13b_lora"
)
model.eval()

# 3. 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-13b-hf")
tokenizer.pad_token = tokenizer.eos_token

print("Model loaded successfully!")

# 4. 准备测试数据
test_news = [
    "Apple Inc. reported record quarterly revenue driven by strong iPhone sales.",
    "Tesla faces challenges as competition in EV market intensifies.",
    "Microsoft announces new AI features in Office suite."
]

# 5. 批量推理
prompts = []
for news in test_news:
    prompt = f"""Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}.
Input: {news}
Answer: """
    prompts.append(prompt)

# Tokenize
tokens = tokenizer(
    prompts,
    return_tensors='pt',
    padding=True,
    max_length=512,
    truncation=True
)
tokens = {key: value.to(model.device) for key, value in tokens.items()}

# 生成
print("\nGenerating predictions...")
with torch.no_grad():
    outputs = model.generate(
        **tokens,
        max_length=512,
        do_sample=False,  # 确定性输出
        pad_token_id=tokenizer.eos_token_id
    )

# 解码
results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# 提取答案
print("\n=== Results ===")
for i, (news, result) in enumerate(zip(test_news, results)):
    answer = result.split("Answer: ")[-1].strip()
    print(f"\n{i+1}. News: {news}")
    print(f"   Sentiment: {answer}")
```

#### 步骤2: 运行

```bash
python sentiment_demo.py
```

**预期输出**:
```
Loading model...
Model loaded successfully!

Generating predictions...

=== Results ===

1. News: Apple Inc. reported record quarterly revenue driven by strong iPhone sales.
   Sentiment: positive

2. News: Tesla faces challenges as competition in EV market intensifies.
   Sentiment: negative

3. News: Microsoft announces new AI features in Office suite.
   Sentiment: positive
```

#### 常见问题

**Q: 显存不足怎么办？**

A: 尝试使用更小的模型或4-bit量化：

```python
# 使用7B模型代替13B
base_model = "meta-llama/Llama-2-7b-chat-hf"

# 或使用4-bit量化
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Llama-2-13b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```

---

### Demo 2: 股价预测

#### 步骤1: 安装额外依赖

```bash
pip install yfinance finnhub-python
```

#### 步骤2: 获取API Key

访问 [Finnhub](https://finnhub.io/) 注册并获取免费API key。

**⚠️ 安全提醒**: 
- 永远不要将API密钥硬编码到代码中并提交到版本控制系统
- 使用环境变量或配置文件存储敏感信息
- 在生产环境使用密钥管理服务（如AWS Secrets Manager）

#### 步骤3: 创建预测脚本

创建文件 `forecaster_demo.py`:

```python
import yfinance as yf
import finnhub
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 配置
# 推荐做法：使用环境变量存储API key
import os
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise ValueError("请设置环境变量 FINNHUB_API_KEY")
# 设置环境变量方法: export FINNHUB_API_KEY="your_key_here"

TICKER = "AAPL"
WEEKS_BACK = 2

print(f"Analyzing {TICKER}...")

# 1. 获取股价数据
end_date = datetime.now()
start_date = end_date - timedelta(weeks=WEEKS_BACK)

stock = yf.Ticker(TICKER)
hist = stock.history(start=start_date, end=end_date)

start_price = hist['Close'].iloc[0]
end_price = hist['Close'].iloc[-1]
price_change = ((end_price - start_price) / start_price) * 100

print(f"Price: ${start_price:.2f} -> ${end_price:.2f} ({price_change:+.2f}%)")

# 2. 获取公司新闻
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

news = finnhub_client.company_news(
    TICKER,
    _from=start_date.strftime('%Y-%m-%d'),
    to=end_date.strftime('%Y-%m-%d')
)

# 格式化新闻
news_text = ""
for i, item in enumerate(news[:5], 1):  # 取前5条
    news_text += f"[Headline {i}]: {item['headline']}\n"
    news_text += f"[Summary {i}]: {item['summary']}\n\n"

# 3. 构建prompt
info = stock.info
company_intro = f"""{info.get('longName', TICKER)} is a leading entity in the {info.get('industry', 'Technology')} sector."""

prompt = f"""[INST] <<SYS>>
You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week.
<</SYS>>

[Company Introduction]:
{company_intro}

From {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}, {TICKER}'s stock price {"increased" if end_price > start_price else "decreased"} from ${start_price:.2f} to ${end_price:.2f}.

Company news during this period are listed below:

{news_text}

Based on all the information before {end_date.strftime('%Y-%m-%d')}, let's first analyze the positive developments and potential concerns for {TICKER}. Come up with 2-4 most important factors respectively and keep them concise. Then make your prediction of the {TICKER} stock price movement for next week. Provide a summary analysis to support your prediction. [/INST]"""

# 4. 加载模型
print("\nLoading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    load_in_8bit=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# 5. 生成预测
print("Generating prediction...")
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {key: value.to(model.device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=4096,
        temperature=0.1,
        do_sample=True
    )

prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 提取答案部分
answer = prediction.split("[/INST]")[-1].strip()

print("\n=== Prediction ===")
print(answer)
```

#### 步骤4: 运行

```bash
python forecaster_demo.py
```

**预期输出示例**:
```
Analyzing AAPL...
Price: $178.50 -> $185.20 (+3.75%)

Loading model...
Generating prediction...

=== Prediction ===

[Positive Developments]:
1. Strong iPhone sales drove record revenue
2. Services segment showing continued growth
3. New product announcements received positive market response

[Potential Concerns]:
1. Supply chain constraints in certain regions
2. Increased competition in smartphone market
3. Regulatory scrutiny in international markets

[Prediction & Analysis]:
Prediction: Up by 0-2%

The company's strong fundamentals and positive recent developments
suggest continued momentum. However, the gains may be moderate due
to profit-taking after recent rally and broader market uncertainty.
The positive sentiment from new products should provide support.
```

**⚠️ 免责声明**: 
此预测仅为演示目的，不构成投资建议。实际投资决策应基于全面的研究和专业咨询。过往表现不代表未来结果。

---

## 模型训练实践

### 训练场景1: 情感分析微调（小规模）

适合学习和实验的入门训练。

#### 准备数据

创建 `prepare_data.py`:

```python
from datasets import load_dataset
import pandas as pd

# 加载FinGPT的训练数据
dataset = load_dataset("FinGPT/fingpt-sentiment-train")

# 查看数据格式
print("Dataset structure:")
print(dataset)
print("\nSample:")
print(dataset['train'][0])

# 可以创建小规模训练集用于快速实验
small_train = dataset['train'].select(range(1000))  # 只用1000个样本
small_train.save_to_disk("./data/small_sentiment_train")

print(f"\nSmall dataset saved: {len(small_train)} samples")
```

#### 训练脚本

创建 `train_sentiment.py`:

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk

# 1. 配置
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
OUTPUT_DIR = "./output/fingpt-sentiment-demo"
DATASET_PATH = "./data/small_sentiment_train"

# 2. 加载数据
print("Loading dataset...")
dataset = load_from_disk(DATASET_PATH)

# 3. 加载模型和tokenizer
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 4. 配置LoRA
lora_config = LoraConfig(
    r=8,  # rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Llama-2的attention模块
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 5. 数据预处理
def preprocess_function(examples):
    # 格式化为instruction格式
    prompts = []
    for instruction, input_text, output_text in zip(
        examples['instruction'],
        examples['input'],
        examples['output']
    ):
        prompt = f"""Instruction: {instruction}
Input: {input_text}
Answer: {output_text}"""
        prompts.append(prompt)
    
    return tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

print("Preprocessing data...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 6. 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 有效batch_size=16
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    report_to="none",  # 不使用wandb
)

# 7. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 8. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 9. 开始训练
print("\nStarting training...")
trainer.train()

# 10. 保存模型
print("\nSaving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nTraining complete! Model saved to {OUTPUT_DIR}")
```

#### 运行训练

```bash
# 准备数据
python prepare_data.py

# 开始训练
python train_sentiment.py
```

#### 监控训练

训练过程中会显示：
```
Trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06

Epoch 1/3:
  10/250 [04%] - loss: 2.145, lr: 1.92e-04
  20/250 [08%] - loss: 1.987, lr: 1.84e-04
  ...

Epoch 2/3:
  ...

Training complete! Model saved to ./output/fingpt-sentiment-demo
```

#### 测试训练的模型

创建 `test_trained_model.py`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 加载训练的模型
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    load_in_8bit=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "./output/fingpt-sentiment-demo"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# 测试
test_text = "Tesla stock surges on strong delivery numbers"
prompt = f"""Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}.
Input: {test_text}
Answer: """

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=256)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

---

### 训练场景2: 多GPU并行训练（进阶）

当你有多张GPU时，可以使用DeepSpeed加速训练。

#### DeepSpeed配置

创建 `ds_config.json`:

```json
{
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 100
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true,
    "contiguous_gradients": true
  }
}
```

#### 修改训练脚本

在 `training_args` 中添加：

```python
training_args = TrainingArguments(
    # ... 其他参数
    deepspeed="./ds_config.json",
    local_rank=-1,  # DeepSpeed会自动设置
)
```

#### 运行多GPU训练

```bash
# 使用4张GPU
deepspeed --num_gpus=4 train_sentiment.py

# 或指定GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 train_sentiment.py
```

---

## 模型推理实践

### 批量推理脚本

处理大量数据时的高效推理。

创建 `batch_inference.py`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import pandas as pd

# 加载模型
print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "NousResearch/Llama-2-13b-hf",
    load_in_8bit=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(
    base_model,
    "FinGPT/fingpt-sentiment_llama2-13b_lora"
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-13b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 读取数据
df = pd.read_csv("news_to_analyze.csv")  # 包含'text'列
texts = df['text'].tolist()

# 批量推理函数
def batch_predict(texts, batch_size=8):
    results = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        
        # 准备prompts
        prompts = [
            f"Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}.\nInput: {text}\nAnswer: "
            for text in batch
        ]
        
        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch_results = [r.split("Answer: ")[-1].strip() for r in batch_results]
        
        results.extend(batch_results)
    
    return results

# 执行推理
print(f"Processing {len(texts)} texts...")
predictions = batch_predict(texts, batch_size=8)

# 保存结果
df['sentiment'] = predictions
df.to_csv("news_with_sentiment.csv", index=False)

print("Done! Results saved to news_with_sentiment.csv")

# 统计
print("\nSentiment distribution:")
print(df['sentiment'].value_counts())
```

---

## 常见问题解决

### 问题1: CUDA Out of Memory

**错误信息**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**解决方案**:

```python
# 1. 使用更激进的量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 改用4-bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 2. 减小batch size
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # 改小
    gradient_accumulation_steps=16,  # 增大以保持有效batch size
)

# 3. 使用梯度检查点
model.gradient_checkpointing_enable()

# 4. 清理缓存
torch.cuda.empty_cache()

# 5. 使用更小的模型
base_model = "meta-llama/Llama-2-7b-chat-hf"  # 而不是13b
```

### 问题2: 模型下载缓慢

**解决方案**:

```bash
# 方法1: 使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方法2: 手动下载
# 访问 https://hf-mirror.com/
# 搜索模型，手动下载到本地
# 然后指定本地路径
model = AutoModelForCausalLM.from_pretrained("./local_models/Llama-2-7b-chat-hf")

# 方法3: 使用代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### 问题3: Import Error

**错误信息**:
```
ImportError: cannot import name 'XXX' from 'transformers'
```

**解决方案**:

```bash
# 检查版本
pip show transformers peft

# 升级到指定版本
pip install transformers==4.35.0 --upgrade
pip install peft==0.7.0 --upgrade

# 如果还有问题，重新安装
pip uninstall transformers peft
pip install transformers==4.35.0 peft==0.7.0
```

### 问题4: 生成质量差

**现象**: 模型输出乱码或不相关内容

**解决方案**:

```python
# 1. 调整生成参数
outputs = model.generate(
    **inputs,
    max_length=512,
    temperature=0.1,  # 降低随机性
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2,  # 避免重复
    do_sample=True,
    num_beams=1  # 如果显存允许可以增加
)

# 2. 检查prompt格式
# 确保完全匹配训练时的格式

# 3. 使用正确的tokenizer
# 必须与训练时的tokenizer一致

# 4. 检查模型是否正确加载
print(model)
print(model.peft_config)  # 检查LoRA配置
```

### 问题5: 训练loss不下降

**解决方案**:

```python
# 1. 检查学习率
training_args = TrainingArguments(
    learning_rate=1e-4,  # 尝试调整
    warmup_steps=100,   # 添加warmup
)

# 2. 检查数据
# 打印几个样本确认格式正确
for i in range(3):
    print(tokenized_dataset[i])

# 3. 检查梯度
# 添加梯度裁剪
training_args = TrainingArguments(
    max_grad_norm=1.0,
)

# 4. 尝试更大的LoRA rank
lora_config = LoraConfig(
    r=16,  # 增大
)

# 5. 增加训练数据
# 确保至少有几千个样本
```

---

## 进阶实践

### 实践1: 创建自定义数据集

#### 收集数据

```python
import yfinance as yf
import finnhub
import pandas as pd
from datetime import datetime, timedelta

# 初始化API（使用环境变量）
import os
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))
# 警告：不要硬编码API密钥！

# 收集多只股票的数据
tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
all_data = []

for ticker in tickers:
    print(f"Collecting data for {ticker}...")
    
    # 获取新闻
    news = finnhub_client.company_news(
        ticker,
        _from="2023-01-01",
        to="2023-12-31"
    )
    
    for item in news:
        all_data.append({
            'ticker': ticker,
            'date': item['datetime'],
            'headline': item['headline'],
            'summary': item['summary'],
        })

# 保存
df = pd.DataFrame(all_data)
df.to_csv("collected_news.csv", index=False)
print(f"Collected {len(df)} news articles")
```

#### 标注数据

```python
import pandas as pd
from transformers import pipeline

# 使用FinGPT给数据标注
labeler = pipeline(
    "text-generation",
    model="FinGPT/fingpt-sentiment_llama2-13b_lora",
    device=0
)

df = pd.read_csv("collected_news.csv")

labels = []
for text in df['headline']:
    prompt = f"Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}.\nInput: {text}\nAnswer: "
    
    result = labeler(prompt, max_length=256, do_sample=False)
    label = result[0]['generated_text'].split("Answer: ")[-1].strip()
    labels.append(label)

df['sentiment'] = labels
df.to_csv("labeled_news.csv", index=False)
```

#### 创建训练格式

```python
from datasets import Dataset

df = pd.read_csv("labeled_news.csv")

# 转换为instruction格式
data = []
for _, row in df.iterrows():
    data.append({
        'instruction': "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.",
        'input': row['headline'],
        'output': row['sentiment']
    })

# 创建Dataset
dataset = Dataset.from_list(data)

# 划分训练/测试集
dataset = dataset.train_test_split(test_size=0.1)

# 保存
dataset.save_to_disk("./my_sentiment_dataset")
```

### 实践2: 模型合并和量化

#### 合并LoRA到基础模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载模型
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    device_map="cpu"  # 先加载到CPU
)

lora_model = PeftModel.from_pretrained(
    base_model,
    "./output/my_trained_lora"
)

# 合并
merged_model = lora_model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("./merged_model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.save_pretrained("./merged_model")

print("Model merged and saved!")
```

#### 量化模型

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载FP16模型
model = AutoModelForCausalLM.from_pretrained(
    "./merged_model",
    torch_dtype=torch.float16,
    device_map="cpu"
)

# 转换为8-bit并保存
model_8bit = model.quantize(8)
model_8bit.save_pretrained("./merged_model_8bit")

print("Model quantized to 8-bit!")
```

### 实践3: 部署为API服务

#### 使用FastAPI

创建 `api_server.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import uvicorn

# 初始化FastAPI
app = FastAPI(title="FinGPT API")

# 全局变量存储模型
model = None
tokenizer = None

class PredictionRequest(BaseModel):
    text: str
    task: str = "sentiment"  # sentiment 或 forecast

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = None

@app.on_event("startup")
async def load_model():
    """启动时加载模型"""
    global model, tokenizer
    
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Llama-2-13b-hf",
        load_in_8bit=True,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(
        base_model,
        "FinGPT/fingpt-sentiment_llama2-13b_lora"
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-13b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """预测接口"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 构建prompt
    if request.task == "sentiment":
        prompt = f"""Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}.
Input: {request.text}
Answer: """
    else:
        raise HTTPException(status_code=400, detail="Invalid task")
    
    # 推理
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            do_sample=False
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = result.split("Answer: ")[-1].strip()
    
    return PredictionResponse(prediction=prediction)

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### 启动服务

```bash
# 安装依赖
pip install fastapi uvicorn

# 启动服务
python api_server.py
```

#### 测试API

```python
import requests

# 测试
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "Apple announces record revenue",
        "task": "sentiment"
    }
)

print(response.json())
# 输出: {"prediction": "positive", "confidence": null}
```

### 实践4: 使用Gradio创建Web界面

创建 `gradio_app.py`:

```python
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# 加载模型
print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    load_in_8bit=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(
    base_model,
    "FinGPT/fingpt-sentiment_llama2-13b_lora"
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

def analyze_sentiment(text):
    """分析情感"""
    prompt = f"""Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}.
Input: {text}
Answer: """
    
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sentiment = result.split("Answer: ")[-1].strip()
    
    return sentiment

# 创建界面
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Enter financial news here...",
        label="Financial News"
    ),
    outputs=gr.Textbox(label="Sentiment"),
    title="FinGPT Sentiment Analyzer",
    description="Analyze the sentiment of financial news using FinGPT",
    examples=[
        ["Apple reports strong quarterly earnings beating expectations"],
        ["Tesla faces production challenges amid supply chain issues"],
        ["Federal Reserve announces interest rate hike to combat inflation"]
    ]
)

# 启动
demo.launch(share=True)  # share=True 可以生成公开链接
```

运行:
```bash
python gradio_app.py
```

---

## 总结

### 学习路径建议

**第1周**: 环境配置 + Demo 1 (情感分析推理)
**第2周**: Demo 2 (股价预测) + 理解代码
**第3周**: 训练场景1 (小规模微调)
**第4周**: 进阶实践 (自定义数据集)
**第5-8周**: 深入某个感兴趣的方向

### 实践建议

1. **由浅入深**: 先跑通demo，再理解原理，最后动手改进
2. **记录过程**: 记录遇到的问题和解决方案
3. **对比实验**: 尝试不同参数，对比效果
4. **参与社区**: 在GitHub提issue，Discord讨论
5. **分享经验**: 写博客或做presentation

### 进一步学习资源

- **官方文档**: https://github.com/AI4Finance-Foundation/FinGPT
- **HuggingFace**: https://huggingface.co/FinGPT
- **论文**: 查看README中列出的论文
- **Discord**: 加入FinGPT社区讨论

---

**祝学习顺利！记住：实践是最好的老师。**
