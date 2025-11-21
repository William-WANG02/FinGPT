# FinGPT 代码结构详解

**⚠️ 免责声明**: 本文档仅供学习和研究使用。文档中的任何代码示例和技术实现不构成投资建议。

---

## 📋 目录
1. [项目目录结构](#项目目录结构)
2. [核心模块详解](#核心模块详解)
3. [重要文件说明](#重要文件说明)
4. [数据流程](#数据流程)
5. [如何阅读代码](#如何阅读代码)

---

## 项目目录结构

```
FinGPT/
├── README.md                          # 项目主README
├── LICENSE                            # MIT开源协议
├── requirements.txt                   # 依赖包列表
├── setup.py                          # 安装配置
├── docs/                             # 文档目录（新增）
│   ├── PROJECT_OVERVIEW_CN.md        # 项目概述
│   ├── ARCHITECTURE_CN.md            # 架构文档
│   ├── CODE_STRUCTURE_CN.md          # 本文档
│   ├── INTERVIEW_GUIDE_CN.md         # 面试指南
│   └── PRACTICAL_GUIDE_CN.md         # 实践指南
├── figs/                             # 图片资源
│   ├── logo_transparent_background.png
│   ├── FinGPT_framework_20240301.png
│   └── ...
├── fingpt/                           # 核心代码目录
│   ├── __init__.py
│   ├── readme.md                     # fingpt模块说明
│   │
│   ├── FinGPT_Benchmark/             # 多任务基准测试
│   │   ├── train_lora.py            # LoRA训练脚本
│   │   ├── utils.py                 # 工具函数
│   │   ├── config.json              # 训练配置
│   │   ├── data/                    # 数据处理
│   │   │   └── prepare_data.ipynb
│   │   └── benchmarks/              # 评估脚本
│   │
│   ├── FinGPT_Forecaster/           # 股价预测应用
│   │   ├── train_lora.py            # 训练脚本
│   │   ├── data.py                  # 数据获取
│   │   ├── data_pipeline.py         # 数据处理管线
│   │   ├── prompt.py                # 提示模板
│   │   ├── app.py                   # Gradio应用
│   │   ├── demo.ipynb               # 演示notebook
│   │   └── requirements.txt
│   │
│   ├── FinGPT_Sentiment_Analysis_v1/ # 情感分析v1
│   │   # 基于市场标签的版本
│   │
│   ├── FinGPT_Sentiment_Analysis_v3/ # 情感分析v3 (SOTA)
│   │   ├── README.md
│   │   ├── training_8bit/           # 8位量化训练
│   │   │   └── train_Llama2_13B.ipynb
│   │   ├── training_int4/           # 4位量化训练(QLoRA)
│   │   │   └── train.ipynb
│   │   ├── training_parallel/       # 多GPU并行训练
│   │   │   └── train.sh
│   │   ├── benchmark/               # 基准测试
│   │   │   └── benchmarks.ipynb
│   │   └── data/                    # 数据准备
│   │       └── making_data.ipynb
│   │
│   ├── FinGPT_RAG/                  # 检索增强生成
│   │   ├── instruct-FinGPT/         # 指令微调版本
│   │   ├── multisource_retrieval/   # 多源检索
│   │   └── requirements.txt
│   │
│   ├── FinGPT_FinancialReportAnalysis/ # 财报分析
│   │
│   ├── FinGPT_MultiAgentsRAG/       # 多Agent RAG系统
│   │
│   └── FinGPT_Others/               # 其他应用
│       ├── FinGPT_Robo_Advisor/     # 智能投顾
│       ├── FinGPT_Trading/          # 交易应用
│       └── FinGPT_Low_Code_Development/ # 低代码开发
│
└── Jupyter Notebooks (根目录)
    ├── FinGPT_Training_LoRA_with_ChatGLM2_6B_for_Beginners.ipynb
    ├── FinGPT_Inference_Llama2_13B_falcon_7B_for_Beginners.ipynb
    └── FinGPT_ Training with LoRA and Meta-Llama-3-8B.ipynb
```

---

## 核心模块详解

### 1. FinGPT_Benchmark - 多任务基准测试

**目的**: 提供标准化的评估框架，支持多个金融NLP任务的训练和评估。

**核心文件**:

#### `train_lora.py` - 主训练脚本
```python
# 关键功能：
# 1. 加载多任务数据集
# 2. 配置LoRA参数
# 3. 使用Trainer进行训练
# 4. 保存模型检查点

# 核心代码结构：
def main():
    # 1. 解析命令行参数
    args = parse_args()
    
    # 2. 加载数据
    dataset = load_dataset(args.dataset_name)
    
    # 3. 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 4. 配置LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
    )
    
    # 5. 训练
    trainer = Trainer(model, args, train_dataset)
    trainer.train()
```

**使用方式**:
```bash
python train_lora.py \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --dataset_name FinGPT/fingpt-sentiment-train \
    --output_dir ./output
```

#### `utils.py` - 工具函数
```python
# 包含的功能：
# - 数据预处理函数
# - 指令格式化
# - 评估指标计算
# - 模型加载辅助函数

def format_instruction(instruction, input_text, output_text=None):
    """格式化为instruction格式"""
    prompt = f"Instruction: {instruction}\nInput: {input_text}\n"
    if output_text:
        prompt += f"Answer: {output_text}"
    return prompt

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    # 计算accuracy, F1等
    return metrics
```

#### `config.json` - 配置文件
```json
{
  "model_name": "meta-llama/Llama-2-7b-chat-hf",
  "datasets": [
    "FinGPT/fingpt-sentiment-train",
    "FinGPT/fingpt-headline"
  ],
  "lora_config": {
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1
  },
  "training_args": {
    "learning_rate": 1e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4
  }
}
```

---

### 2. FinGPT_Forecaster - 股价预测系统

**目的**: 基于新闻和基本面数据预测股价走势，是FinGPT的旗舰应用。

#### `data.py` - 数据获取模块
```python
import yfinance as yf
import finnhub

class DataFetcher:
    """数据获取类"""
    
    def __init__(self, api_key=None):
        self.finnhub_client = finnhub.Client(api_key)
    
    def get_stock_data(self, ticker, start_date, end_date):
        """获取股票价格数据"""
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        return hist
    
    def get_company_news(self, ticker, start_date, end_date):
        """获取公司新闻"""
        news = self.finnhub_client.company_news(
            ticker, start_date, end_date
        )
        return news
    
    def get_basic_financials(self, ticker):
        """获取基本财务数据"""
        financials = self.finnhub_client.company_basic_financials(
            ticker, 'all'
        )
        return financials
```

#### `data_pipeline.py` - 数据处理管线
```python
class DataPipeline:
    """数据处理管线"""
    
    def process_news(self, news_list):
        """处理新闻数据"""
        processed = []
        for news in news_list:
            # 清洗标题和摘要
            cleaned = self.clean_text(news['summary'])
            # 提取关键信息
            processed.append({
                'headline': news['headline'],
                'summary': cleaned,
                'datetime': news['datetime']
            })
        return processed
    
    def format_prompt(self, company_info, news, financials, 
                     start_price, end_price):
        """格式化为模型输入"""
        # 构建完整的prompt
        prompt = f"""
        [Company Introduction]:
        {company_info}
        
        From {start_date} to {end_date}, stock price 
        {"increased" if end_price > start_price else "decreased"} 
        from {start_price} to {end_price}.
        
        Company news during this period:
        {self.format_news(news)}
        
        Basic financials:
        {self.format_financials(financials)}
        
        Based on the information above, analyze...
        """
        return prompt
```

#### `prompt.py` - 提示模板
```python
SYSTEM_PROMPT = """
You are a seasoned stock market analyst. Your task is to list 
the positive developments and potential concerns for companies 
based on relevant news and basic financials from the past weeks, 
then provide an analysis and prediction for the companies' stock 
price movement for the upcoming week.

Your answer format should be as follows:

[Positive Developments]:
1. ...

[Potential Concerns]:
1. ...

[Prediction & Analysis]:
...
"""

def create_llama2_prompt(system_prompt, user_prompt):
    """创建Llama2格式的prompt"""
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    
    prompt = f"{B_INST} {B_SYS}{system_prompt}{E_SYS}{user_prompt} {E_INST}"
    return prompt
```

#### `train_lora.py` - 训练脚本
```python
def train_forecaster():
    # 1. 准备训练数据
    dataset = prepare_forecaster_dataset()
    
    # 2. 加载Llama-2模型
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        load_in_8bit=True,
        device_map="auto"
    )
    
    # 3. 配置LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 4. 训练
    model = get_peft_model(model, peft_config)
    trainer = Trainer(model, training_args, train_dataset)
    trainer.train()
    
    # 5. 保存
    model.save_pretrained("./fingpt-forecaster")
```

#### `app.py` - Gradio Web应用
```python
import gradio as gr

def predict_stock_movement(ticker, date, news_weeks, use_financials):
    """预测函数"""
    # 1. 获取数据
    data = fetch_data(ticker, date, news_weeks)
    
    # 2. 格式化prompt
    prompt = format_prompt(data, use_financials)
    
    # 3. 模型推理
    prediction = model.generate(prompt)
    
    return prediction

# 创建Gradio界面
demo = gr.Interface(
    fn=predict_stock_movement,
    inputs=[
        gr.Textbox(label="Ticker Symbol"),
        gr.Textbox(label="Date (YYYY-MM-DD)"),
        gr.Slider(1, 4, value=2, label="Past Weeks"),
        gr.Checkbox(label="Include Financials")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="FinGPT Forecaster"
)

demo.launch()
```

---

### 3. FinGPT_Sentiment_Analysis_v3 - 情感分析（SOTA版本）

**目的**: 提供最先进的金融情感分析能力，性能超越GPT-4。

#### 目录结构
```
FinGPT_Sentiment_Analysis_v3/
├── training_8bit/              # 8位量化训练
│   └── train_Llama2_13B.ipynb # Jupyter notebook
├── training_int4/              # 4位量化训练
│   └── train.ipynb            # QLoRA训练
├── training_parallel/          # 多GPU训练
│   ├── train.sh               # 训练脚本
│   └── deepspeed_config.json  # DeepSpeed配置
├── benchmark/                  # 基准测试
│   └── benchmarks.ipynb       # 评估notebook
└── data/                       # 数据准备
    └── making_data.ipynb      # 数据生成
```

#### 关键代码片段

**8-bit训练** (`training_8bit/train_Llama2_13B.ipynb`):
```python
# 1. 加载模型（8-bit）
model = LlamaForCausalLM.from_pretrained(
    "NousResearch/Llama-2-13b-hf",
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)

# 2. 准备数据
train_data = load_dataset("FinGPT/fingpt-sentiment-train")

# 3. 配置LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 4. 训练参数
training_args = TrainingArguments(
    output_dir="./fingpt-sentiment",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=100,
    save_steps=500,
    evaluation_strategy="steps",
)

# 5. 开始训练
model = get_peft_model(model, peft_config)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)
trainer.train()
```

**4-bit训练(QLoRA)** (`training_int4/train.ipynb`):
```python
from transformers import BitsAndBytesConfig

# 配置4-bit量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# QLoRA配置
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 训练...
```

**基准测试** (`benchmark/benchmarks.ipynb`):
```python
from sklearn.metrics import f1_score, accuracy_score

def evaluate_model(model, test_dataset):
    """评估模型性能"""
    predictions = []
    labels = []
    
    for sample in test_dataset:
        # 生成预测
        pred = model.generate(sample['input'])
        predictions.append(pred)
        labels.append(sample['label'])
    
    # 计算指标
    f1 = f1_score(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    
    return {
        'weighted_f1': f1,
        'accuracy': acc
    }

# 在多个数据集上评估
datasets = ['FPB', 'FiQA-SA', 'TFNS', 'NWGI']
results = {}
for ds in datasets:
    test_data = load_test_dataset(ds)
    results[ds] = evaluate_model(model, test_data)
```

---

### 4. FinGPT_RAG - 检索增强生成

**目的**: 结合外部知识库提高回答准确性。

#### 核心组件

**向量数据库设置**:
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 1. 初始化embedding模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. 创建向量数据库
vectorstore = FAISS.from_documents(
    documents=financial_docs,
    embedding=embeddings
)

# 3. 创建检索器
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}  # 返回top-5
)
```

**RAG推理流程**:
```python
from langchain.chains import RetrievalQA

def rag_inference(query):
    # 1. 检索相关文档
    relevant_docs = retriever.get_relevant_documents(query)
    
    # 2. 构建增强的prompt
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    enhanced_prompt = f"""
    Context:
    {context}
    
    Question: {query}
    
    Answer based on the context above:
    """
    
    # 3. LLM生成答案
    answer = model.generate(enhanced_prompt)
    
    # 4. 返回答案和来源
    return {
        'answer': answer,
        'sources': [doc.metadata for doc in relevant_docs]
    }
```

---

## 重要文件说明

### 配置文件

#### `requirements.txt`
```
# 核心依赖
transformers>=4.30.0     # HuggingFace transformers
peft>=0.4.0             # Parameter-Efficient Fine-Tuning
torch>=2.0.0            # PyTorch
datasets>=2.12.0        # HuggingFace datasets

# 金融数据
yfinance               # Yahoo Finance
finnhub-python         # Finnhub API
tushare                # 中国市场数据

# 工具
gradio                 # Web界面
pandas                 # 数据处理
numpy                  # 数值计算
```

#### `setup.py`
```python
from setuptools import setup, find_packages

setup(
    name="FinGPT",
    version="0.0.1",
    author="AI4Finance Foundation",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "peft",
        "torch",
        # ...
    ],
    python_requires=">=3.8",
)
```

### 教学Notebooks（根目录）

#### `FinGPT_Training_LoRA_with_ChatGLM2_6B_for_Beginners.ipynb`
**适合人群**: 初学者，想用中文模型

**内容**:
1. 环境配置
2. 数据准备
3. ChatGLM2-6B加载
4. LoRA微调
5. 模型评估
6. 保存和部署

#### `FinGPT_Inference_Llama2_13B_falcon_7B_for_Beginners.ipynb`
**适合人群**: 想了解推理过程

**内容**:
1. 加载预训练的FinGPT模型
2. 准备输入数据
3. 运行推理
4. 解析输出
5. 性能对比

#### `FinGPT_ Training with LoRA and Meta-Llama-3-8B.ipynb`
**适合人群**: 想用最新Llama-3模型

**内容**:
1. Llama-3特性介绍
2. 环境配置
3. LoRA微调
4. 性能评估

---

## 数据流程

### 完整数据流程图

```
1. 数据采集
   ├── Yahoo Finance API → 股价数据
   ├── Finnhub API → 新闻 + 财务数据
   └── Twitter API → 社交媒体数据
            ↓
2. 数据清洗 (data_pipeline.py)
   ├── 去除HTML标签
   ├── 标准化日期格式
   ├── 处理缺失值
   └── 文本预处理
            ↓
3. 特征工程
   ├── 情感特征提取
   ├── 实体识别
   └── 技术指标计算
            ↓
4. 格式化为Instruction格式
   ├── 构建instruction
   ├── 准备input
   └── 生成expected output
            ↓
5. 模型训练 (train_lora.py)
   ├── 加载基础模型
   ├── 配置LoRA
   └── 开始训练
            ↓
6. 模型评估 (benchmarks.ipynb)
   ├── 在测试集上评估
   ├── 计算F1、Accuracy等指标
   └── 与baseline对比
            ↓
7. 模型部署
   ├── 保存LoRA适配器
   ├── 上传到HuggingFace Hub
   └── 创建Gradio应用
```

### 代码层面的数据流

```python
# 完整示例：从数据到预测

# 1. 数据获取
from fingpt.FinGPT_Forecaster.data import DataFetcher
fetcher = DataFetcher()
stock_data = fetcher.get_stock_data("AAPL", "2023-01-01", "2023-12-31")
news_data = fetcher.get_company_news("AAPL", "2023-01-01", "2023-12-31")

# 2. 数据处理
from fingpt.FinGPT_Forecaster.data_pipeline import DataPipeline
pipeline = DataPipeline()
processed_news = pipeline.process_news(news_data)

# 3. 构建prompt
from fingpt.FinGPT_Forecaster.prompt import create_llama2_prompt, SYSTEM_PROMPT
user_prompt = pipeline.format_prompt(
    company_info="Apple Inc...",
    news=processed_news,
    financials=financials_data,
    start_price=150.0,
    end_price=180.0
)
full_prompt = create_llama2_prompt(SYSTEM_PROMPT, user_prompt)

# 4. 模型推理
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = PeftModel.from_pretrained(base_model, "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

inputs = tokenizer(full_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=4096)
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(prediction)
```

---

## 如何阅读代码

### 学习路径建议

#### 初级 (1-2周)
1. **从教学Notebooks开始**
   - 先运行 `FinGPT_Training_LoRA_with_ChatGLM2_6B_for_Beginners.ipynb`
   - 理解完整的训练流程
   - 动手修改参数看效果

2. **理解数据处理**
   - 阅读 `FinGPT_Forecaster/data.py`
   - 了解如何获取金融数据
   - 查看数据格式

3. **学习prompt工程**
   - 阅读 `FinGPT_Forecaster/prompt.py`
   - 理解instruction格式
   - 尝试修改prompt

#### 中级 (2-4周)
1. **深入训练代码**
   - 阅读 `train_lora.py`
   - 理解LoRA配置
   - 理解训练参数

2. **学习评估方法**
   - 阅读 `benchmark/benchmarks.ipynb`
   - 了解评估指标
   - 在自己的数据上评估

3. **探索不同应用**
   - 对比Forecaster和Sentiment模块
   - 理解不同任务的差异
   - 尝试复现结果

#### 高级 (4-8周)
1. **优化和改进**
   - 尝试不同的LoRA配置
   - 实验不同的基础模型
   - 改进数据处理流程

2. **扩展功能**
   - 添加新的数据源
   - 实现新的任务
   - 优化推理速度

3. **深入理论**
   - 阅读相关论文
   - 理解LoRA数学原理
   - 研究RLHF实现

### 代码阅读技巧

#### 1. 从主流程入手
```python
# 找到main函数或主要的训练/推理函数
# 例如在 train_lora.py 中：

def main():
    # 这是整个训练流程的入口
    # 按照执行顺序阅读
    args = parse_args()           # ← 从这里开始
    dataset = load_data(args)     # ← 然后是数据加载
    model = load_model(args)      # ← 接着是模型
    train(model, dataset)         # ← 最后是训练
```

#### 2. 关注配置文件
```json
// config.json 告诉你：
// - 使用什么模型
// - 什么超参数
// - 数据在哪里
{
  "model_name": "...",
  "lora_r": 8,
  "learning_rate": 1e-4
}
```

#### 3. 查看数据格式
```python
# 在notebook或脚本中打印数据样本
print(dataset[0])
# 输出:
# {
#   'instruction': '...',
#   'input': '...',
#   'output': '...'
# }
```

#### 4. 使用调试工具
```python
# 在关键位置添加断点或打印
import pdb; pdb.set_trace()  # 调试

# 或者简单打印
print(f"Model: {model}")
print(f"Dataset size: {len(dataset)}")
```

#### 5. 参考README
- 每个子模块都有README
- 说明了功能和使用方法
- 提供了运行示例

---

## 常见代码模式

### 模式1: 加载FinGPT模型
```python
# 这个模式在多个文件中出现
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    load_in_8bit=True,
    device_map="auto"
)

# 2. 加载LoRA适配器
model = PeftModel.from_pretrained(
    base_model,
    "FinGPT/fingpt-xxx_lora"  # 不同应用不同的adapter
)

# 3. 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf"
)
```

### 模式2: LoRA训练
```python
# LoRA训练的标准流程
from peft import LoraConfig, get_peft_model

# 1. 配置
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 2. 应用到模型
model = get_peft_model(base_model, peft_config)

# 3. 打印可训练参数
model.print_trainable_parameters()
# 输出示例（实际数值取决于模型和配置）: 
# trainable params: 4M || all params: 6.7B || trainable: 0.06%
```

### 模式3: Instruction格式化
```python
# 所有任务都使用类似的格式
def format_instruction(instruction, input_text, output_text=None):
    prompt = f"Instruction: {instruction}\nInput: {input_text}\n"
    if output_text:
        prompt += f"Answer: {output_text}"
    else:
        prompt += "Answer: "
    return prompt

# 使用
prompt = format_instruction(
    "What is the sentiment of this news?",
    "Apple reports strong earnings...",
    "positive"
)
```

---

## 面试中如何展示代码理解

### 1. 能说出关键文件的作用
- "训练主要在 `train_lora.py` 中进行"
- "数据获取使用 `data.py` 中的 DataFetcher 类"
- "prompt模板定义在 `prompt.py` 中"

### 2. 能解释代码流程
- "首先加载数据，然后配置LoRA，接着训练，最后评估"
- "推理时先format prompt，然后tokenize，生成输出，最后decode"

### 3. 能指出优化点
- "可以使用gradient checkpointing减少显存"
- "可以增加gradient accumulation steps提高batch size"
- "可以尝试不同的LoRA rank找到最优配置"

### 4. 能对比不同方案
- "8-bit量化比4-bit准确但需要更多显存"
- "Llama-2在英文市场效果好，ChatGLM2适合中文"
- "多任务学习可以提高泛化，但单任务可能在特定任务上更好"

---

**下一步**: 阅读 `INTERVIEW_GUIDE_CN.md` 准备面试问题
