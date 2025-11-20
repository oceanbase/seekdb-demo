# 使用 SeekDB 快速构建 RAG

本教程将引导您把 Markdown 文档导入 SeekDB，构建向量检索知识库并通过 Streamlit 启动 RAG 界面。

## 前提条件

- 已安装 Python 3.8 或以上版本
- 已准备好 LLM API Key（用于生成答案，embedding 使用本地模型，无需 API Key）

## 准备工作

### 1. 下载代码

```bash
git clone https://github.com/oceanbase/seekdb-demo.git
cd seekdb-demo/RAG_SEARCH_WITH_SEEKDB
```

### 2. 设置环境

#### 安装依赖

```bash
pip install -r requirements.txt
```

#### 设置环境变量

**步骤一：复制环境变量模板**

```bash
cp .env.example .env
```

**步骤二：编辑 `.env` 文件，设置环境变量**

> **注意：** Embedding 模型使用本地 `DefaultEmbeddingFunction`（384维），无需配置 API Key。只需为 LLM 配置 API Key。

以下使用通义千问作为示例：

```env
OPENAI_API_KEY=sk-your-dashscope-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL_NAME=qwen-plus
SEEKDB_DIR=./seekdb_rag
SEEKDB_NAME=test
COLLECTION_NAME=embeddings
```

**环境变量说明：**

| 变量名           | 说明                                           | 默认值/示例值                                    |
|------------------|------------------------------------------------|--------------------------------------------------|
| OPENAI_API_KEY   | LLM API Key（支持 OpenAI、通义千问等兼容服务） | 必须设置                                         |
| OPENAI_BASE_URL  | LLM API 基础 URL                               | https://dashscope.aliyuncs.com/compatible-mode/v1 |
| OPENAI_MODEL_NAME        | 语言模型名称                                   | qwen-plus                                        |
| SEEKDB_DIR       | SeekDB 数据库目录                              | ./seekdb_rag                                     |
| SEEKDB_NAME      | 数据库名称                                     | test                                             |
| COLLECTION_NAME       | 嵌入表名称                                     | embeddings                                       |

> **说明：** Embedding 模型使用 pyseekdb 自带的 `DefaultEmbeddingFunction`（基于 sentence-transformers，384维），首次使用会自动下载模型，无需配置 API Key。

### 3. 准备数据

我们使用 SeekDB 文档作为示例，您也可以使用自己的 Markdown 文档。

**下载示例文档：**

```bash
mkdir -p seekdb_docs && curl -o seekdb_docs/pyseekdb-sdk.md https://raw.githubusercontent.com/oceanbase/seekdb/develop/docs/user-guide/en/pyseekdb-sdk.md
```

**导入数据：**

运行数据导入脚本：

```bash
python seekdb_insert.py seekdb_docs
```

**导入说明：**

在此步骤中，系统会执行如下操作：

- 读取指定目录下的所有 Markdown 文件
- 将文档按标题分割成文本块（使用 `# ` 分隔符）
- 使用本地 embedding 模型（DefaultEmbeddingFunction，384维）自动生成文本嵌入
- 将嵌入向量存储到 SeekDB 数据库
- 自动跳过失败的文档块，确保批量处理的稳定性

## 构建 RAG

通过 Streamlit 启动应用：

```bash
streamlit run seekdb_app.py
```

启动后，您可以在浏览器中访问 RAG 界面，查询您要检索的数据了。
