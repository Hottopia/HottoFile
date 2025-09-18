

---

# 📑 项目设计文档
1. 项目概述

本项目旨在构建一个 **智能文件管理与检索系统**。 系统支持对多类型文件进行解析、分类、向量化存储和语义检索，帮助用户快速找到所需文件。

项目采用 **主系统 + 分系统** 的分层架构：

+ **主系统**：负责统一调度、接口整合、对外服务
+ **分系统**：负责各自的任务（文件解析、分类、Embedding、存储、检索等）

---

1. 系统架构图

```plain
┌────────────────────┐
                │      主系统(调度)   │
                │  - 监听目标文件夹   │
                │  - 调用各分系统API │
                │  - 执行文件移动     │
                └─────────┬──────────┘
                          │
     ┌────────────────────┼────────────────────┐
     │                    │                    │
┌─────────────┐   ┌────────────────┐   ┌─────────────────┐
│ 文件监控子系统 │   │ 预处理子系统    │   │ Embedding子系统 │
│ - 发现新文件   │   │ - 抽取文件名    │   │ - 生成向量 调用 BGE-M3 Embedding      │
│ - 触发流程     │──▶│ - 文本摘要      │──▶│ - 存入向量库 FAISS 向量存储    │
└─────────────┘   │ - 图像特征提取   │   │ - 相似度检索    │
                  └────────────────┘   └─────────────────┘
                          │                    │
                          ▼                    │
                  ┌────────────────┐           │
                  │ 分类子系统      │◀──────────┘
                  │ - 文档分类      │
                  │ - 图像分类      │
                  │ - LLM/规则混合 │
                  └───────┬────────┘
                          │
                  ┌────────────────┐
                  │ 学习子系统      │
                  │ - 接收用户反馈  │
                  │ - 更新向量库    │
                  │ - 修正规则      │
                  └────────────────┘
```

---

1. 模块说明

### 3.1 主系统 (Orchestrator)
+ **任务**
    - 调用分系统执行任务
    - 接收并整合结果
    - 通过统一接口对外提供服务
+ **输入**
    - 用户请求（添加文件 / 检索文件 / 更新分类）
+ **输出**
    - 处理结果（检索到的文件 / 分类结果 / 系统日志）

---

### 3.2 文件解析系统
+ **任务**
    - 读取文件（文本、图片等）
    - 提取内容：文本、OCR 识别（图片）
    - 生成简要摘要（text_summary）
+ **输入**
    - 本地文件路径
+ **输出**
    - `FileInfo` 对象（包含基本信息 + 内容摘要）

---

### 3.3 分类系统
分类系统由 **三层逻辑** 组成：

1. **规则分类（快速判断）**
    1. 按扩展名初步分类（例如`.pdf → document`, `.jpg → image`）
2. **Embedding + 最近邻检索**
    1. 文件摘要 → Embedding
    2. 与已有类别向量比对，获取候选类别
3. **大模型****分类（****LLM****Classifier****）**
    1. 输入：文件摘要 + 候选类别
    2. 输出：最终分类结果（如 “科研论文” / “代码脚本” / “合同”）

✅ **特点**：Embedding 负责找候选类别，如果Embedding分不了，置信度不够，LLM 负责做判别。

---

### 3.4 学习模块（Feedback Learning）
+ **任务**
    - 用户如果手动修正分类 → 系统更新向量数据库
    - 同时存储 **修正样本**（原始文本 + 正确类别）
    - 周期性微调分类器 / 更新 Embedding 相似度搜索权重
+ **输入**: 用户修正结果
+ **输出**: 更新后的分类模型 / 向量库

---

### 3.5 向量化与检索系统
+ **任务**
    - 将文件内容 Embedding 存入向量数据库
    - 支持语义搜索、文件相似度检索
+ **输入**: `FileInfo` 或 查询文本
+ **输出**: 相似文件列表

---

1. **数据结构设计**

### 4.1 FileInfo 类
```plain
import json
import uuid
from datetime import datetime

class FileInfo:
    def __init__(self, path, name, ext, ftype, size, created_at=None, modified_at=None, 
                 text_summary=None, image_features=None, audio_transcript=None, video_keyframes=None):
        # 基础信息
        self.file_id = str(uuid.uuid4())  # 自动生成唯一ID
        self.path = path
        self.name = name
        self.ext = ext
        self.type = ftype  # "document" / "image" / "audio" / "video" / "other"

        # 元信息
        self.metadata = {
            "size": size,
            "created_at": created_at or datetime.now().isoformat(),
            "modified_at": modified_at or datetime.now().isoformat()
        }

        # 内容
        self.content = {
            "text_summary": text_summary,
            "image_features": image_features,
            "audio_transcript": audio_transcript,
            "video_keyframes": video_keyframes
        }

        # Embedding & 分类信息
        self.embedding = None
        self.candidates = []
        self.final_label = None
        self.feedback = None

    # 转换成字典
    def to_dict(self):
        return {
            "file_id": self.file_id,
            "path": self.path,
            "name": self.name,
            "ext": self.ext,
            "type": self.type,
            "metadata": self.metadata,
            "content": self.content,
            "embedding": self.embedding,
            "candidates": self.candidates,
            "final_label": self.final_label,
            "feedback": self.feedback
        }

    # 转换成JSON
    def to_json(self):
        return json.dumps(self.to_dict(), indent=4, ensure_ascii=False)

    # 更新标签
    def update_label(self, label, feedback=False):
        if feedback:
            self.feedback = label
        else:
            self.final_label = label
      # 内容字典 { "text_summary": "...", "ocr_text": "..." }
```

### 4.2 EmbeddingManager 类
```plain
class EmbeddingManager:
    def __init__(self, model_name="BAAI/bge-m3", device="cpu", normalize=True):
        ...
    def add_file(self, file_obj):
        ...
    def search(self, query, k=3):
        ...
    def search_file(self, file_obj, k=3):
        ...
```

---

1. 工作流程
2. **文件解析** 用户上传文件 → 系统解析内容 → 生成 `FileInfo`
3. **分类** 系统自动分类 → 用户可手动修正
4. **向量化存储**`FileInfo` → EmbeddingManager → FAISS 向量库
5. **语义检索** 用户输入查询 → 向量搜索 → 返回最相似文件

---

1. 运行环境
+ **语言**: Python 3.10+
+ **主要依赖**:

```plain
pip install langchain faiss-cpu sentence-transformers transformers
pip install opencv-python pillow pytesseract  # 如果涉及图片OCR
```

+ **硬件环境**:
    - CPU: Intel Ultra 7
    - GPU: 无（使用 CPU 推理）

## **4.3 ****Classifier****（****大模型****分类器****）**
```plain
class LLMClassifier:
    def __init__(self, llm):
        self.llm = llm
    def classify(self, file_info, candidate_labels):
        """
        输入: FileInfo + 候选类别
        输出: 最终分类标签
        """
```

## 4.4 FeedbackManager（学习模块）
```plain
class FeedbackManager:
    def __init__(self):
        self.corrections = []  # 存用户修正数据
    def add_feedback(self, file_info, correct_label):
        self.corrections.append((file_info.content, correct_label))
        # 可选: 更新 Embedding / 存入数据库
```

# 工作流程
1. **文件上传 → FileInfo 解析**
2. **初步分类（规则 + Embedding 候选）**
3. **大模型****分类 → 给出语义标签**
4. **用户修正（可选） → 学习模块更新**
5. **存入向量数据库**
6. **用户检索 → 相似度搜索 + ****LLM**** 语义增强**

# 运行环境
+ Python 3.10+
+ 必要依赖：

```plain
pip install langchain faiss-cpu sentence-transformers transformers
pip install openai  # 如果用 OpenAI 或其他 LLM
pip install opencv-python pillow pytesseract
```

---

