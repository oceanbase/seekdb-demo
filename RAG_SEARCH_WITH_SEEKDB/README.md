# 使用 SeekDB 快速构建 RAG

本教程将引导您把 Markdown 文档导入 SeekDB，构建向量检索知识库并通过 Streamlit 启动 RAG 界面。

## 前提条件

- 已安装 Python 3.8 或以上版本。
- 已准备好 API Key。

## 准备工作

### 1. 下载代码

```bash
git clone https://github.com/oceanbase/seekdb-demo.git
cd seekdb-demo/RAG_SEARCH_WITH_SEEKDB
```

### 2. 设置环境

#### 安装依赖

```bash
pip install -r seekdb_requirements.txt
```

#### 设置环境变量

**步骤一：复制环境变量模板**

```bash
cp .env.example .env
```

**步骤二：编辑 .env 文件，设置环境变量**

以下使用通义千问作为示例：

```env
OPENAI_API_KEY=sk-your-dashscope-key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v1
LLM_MODEL=qwen-plus
SEEKDB_DIR=./seekdb_rag
SEEKDB_NAME=test
TABLE_NAME=embeddings
```

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| OPENAI_API_KEY | API Key（支持 OpenAI、通义千问等兼容服务） | 必须设置 |
| OPENAI_BASE_URL | API 基础 URL | https://api.openai.com/v1 |
| EMBEDDING_MODEL | 嵌入模型名称 | text-embedding-ada-002 |
| LLM_MODEL | 语言模型名称 | gpt-3.5-turbo |
| SEEKDB_DIR | SeekDB 数据库目录 | ./seekdb_rag |
| SEEKDB_NAME | 数据库名称 | test |
| TABLE_NAME | 嵌入表名称 | embeddings |

### 3. 准备数据

我们使用 SeekDB 文档作为示例，您也可以使用自己的 Markdown 文档。

```bash
mkdir -p seekdb_docs && curl -o seekdb_docs/pyseekdb-sdk.md https://raw.githubusercontent.com/oceanbase/seekdb/develop/docs/user-guide/en/pyseekdb-sdk.md
```

**导入数据**

运行数据导入脚本：

```bash
python seekdb_insert.py seekdb_docs
```

**说明**

在此步骤中，系统会执行如下操作：

- 读取指定目录下的所有 Markdown 文件。
- 将文档按标题分割成文本块（使用 "# " 分隔符）。
- 使用配置的嵌入模型生成文本嵌入。
- 将嵌入向量存储到 SeekDB 数据库。
- 使用配置的嵌入模型生成文本嵌入。
- 自动跳过失败的文档块，确保批量处理的稳定性。

## 构建 RAG

通过 Streamlit 启动应用：

```bash
streamlit run seekdb_app.py
```

现在，您可以在浏览器中访问 RAG 界面，查询您要检索的数据了。
