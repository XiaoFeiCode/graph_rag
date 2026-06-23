# 基于 Hybrid Retrieval 与 GraphRAG 的电商知识增强问答系统

面向电商商品咨询场景，本项目构建了一个结合结构化知识图谱、向量检索、全文检索和大模型生成的知识增强问答系统。系统围绕商品属性长尾、关键词召回不足和大模型幻觉问题，使用 Neo4j 管理商品图谱，使用 BGE embedding 与全文索引完成实体对齐，再由 DeepSeek 生成 Cypher 查询和最终自然语言回答。

## 项目亮点

- 构建电商商品知识图谱，覆盖 `SKU`、`SPU`、品牌、三级品类、平台属性、销售属性和商品标签等节点与关系。
- 使用 MySQL -> Neo4j 数据同步链路，将业务库中的商品、类目、属性和品牌数据批量写入图数据库。
- 使用 BERT token classification 训练中文商品文本 NER 模型，从商品描述中抽取长尾标签并补充到图谱。
- 使用 BGE embedding + Neo4j 向量索引 + 全文索引实现 Hybrid Retrieval，用于品牌、商品、品类等实体对齐。
- 使用 LangChain 串联问题解析、Cypher 生成、图谱查询和答案生成，提供 FastAPI 聊天接口与前端页面。

## 技术栈

Python, PyTorch, Transformers, FastAPI, LangChain, DeepSeek, Neo4j, MySQL, BGE Embedding, HuggingFace Datasets

## 架构

```text
MySQL(gmall)
  -> TableSync: category / spu / sku / trademark / attributes
  -> Neo4j knowledge graph

Label Studio JSON
  -> BERT NER training
  -> product tags
  -> Neo4j Tag nodes

User question
  -> DeepSeek generates parameterized Cypher
  -> BGE + fulltext hybrid retrieval aligns entities
  -> Neo4j executes graph query
  -> DeepSeek generates final answer
  -> FastAPI /api/chat
```

## 目录结构

```text
src/
  configuration/     # 路径、模型、数据库和超参数配置
  datasync/          # MySQL -> Neo4j 数据同步
  ner/               # 商品文本 NER 数据处理、训练、评估和预测
  web/               # FastAPI 服务、GraphRAG 问答链路和静态聊天页
data/                # 本地数据，不建议完整上传大规模处理产物
checkpoints/         # 本地模型权重目录
docs/                # 架构设计与模块说明
```

## 快速开始

推荐使用 `uv` 管理环境。

```bash
uv sync
copy .env.example .env
```

编辑 `.env`，填入 DeepSeek、MySQL 和 Neo4j 配置。

启动 Web 服务：

```bash
uv run uvicorn src.web.app:app --host 0.0.0.0 --port 8000
```

访问：

```text
http://localhost:8000
```

## 数据与训练

预处理 Label Studio 导出的 NER 标注数据：

```bash
uv run python -m src.ner.preprocess
```

训练 NER 模型：

```bash
uv run python -m src.ner.train
```

评估模型：

```bash
uv run python -m src.ner.eval
```

抽取商品标签：

```bash
uv run python -m src.ner.predict
```

同步 MySQL 表数据到 Neo4j：

```bash
uv run python -m src.datasync.table_sync
```

同步 NER 标签到 Neo4j：

```bash
uv run python -m src.datasync.text_sync
```

创建 Neo4j 全文索引和向量索引：

```bash
uv run python -m src.web.utils
```

## 核心链路

1. 用户在前端输入商品咨询问题。
2. `ChatService._get_question_cypher` 根据 Neo4j schema 和用户问题生成参数化 Cypher。
3. `ChatService._align_entities` 使用 Neo4jVector 的 hybrid search 对齐商品、品牌和品类实体。
4. `ChatService._execute_cypher` 执行图查询。
5. `ChatService._generate_answer` 结合查询结果生成自然语言回答。

## 当前能力

- 1000 条商品 query 标注数据，完成 B/I/O 序列标注训练流程。
- 已训练 BERT NER 模型，并支持批量抽取商品卖点标签。
- 图谱同步覆盖类目、品牌、商品、属性和标签关系。
- FastAPI 聊天接口与静态聊天页面可用于演示 GraphRAG 问答流程。

## 后续优化方向

- 将当前批量同步升级为 Debezium + Kafka 增量同步链路。
- 将 Neo4j 向量索引扩展为 Milvus，保留 Neo4j 负责结构化关系推理。
- 增加 Recall@5、MRR、Precision@5 等检索评估脚本，沉淀可复现实验报告。
- 为 Cypher 生成增加模板约束、错误重试和只读安全校验。
- 增加 Docker Compose，一键启动 MySQL、Neo4j 和 Web 服务。

## 安全说明

项目通过 `.env` 管理数据库密码、API key 等敏感配置，仓库仅提供 `.env.example` 作为配置模板。模型 checkpoint、日志和预处理数据作为本地运行产物管理。
