# 简历项目包装

## 推荐简历标题

基于 Hybrid Retrieval 与 GraphRAG 的电商知识增强问答系统

## 技术栈

Python, PyTorch, Transformers, FastAPI, LangChain, DeepSeek, Neo4j, MySQL, BGE Embedding

如果后续补齐增量同步和独立向量库，可以扩展为：

Python, PyTorch, Neo4j, Milvus, FastAPI, LangChain, Transformers, Kafka, Debezium

## 简历描述

- 面向电商商品属性长尾、关键词召回不足和大模型幻觉问题，构建 Hybrid Retrieval + GraphRAG 商品知识问答系统，融合结构化图谱查询、向量召回和大模型生成，提升商品属性问答的可解释性与知识覆盖。
- 基于 Neo4j 构建电商商品知识图谱，完成 `SKU/SPU/品牌/品类/平台属性/销售属性/商品标签` 等节点建模与关系同步，并通过 MySQL 批量同步脚本沉淀可复用图谱构建流程。
- 基于 Label Studio 构建 1000 条商品 Query/描述实体标注数据，使用 BERT token classification 完成商品标签 NER 训练、评估与批量抽取，将长尾卖点标签补充到图谱。
- 基于 BGE Embedding、Neo4j 向量索引和全文索引构建 Hybrid Retrieval 实体对齐模块，用于品牌、商品和品类名称纠错召回；结合 LangChain 生成参数化 Cypher，完成图谱查询与答案生成。
- 封装 FastAPI `/api/chat` 服务和聊天前端，打通用户问题解析、实体对齐、Cypher 查询、图谱结果回填和自然语言生成的端到端 GraphRAG 问答链路。

## 更激进版本

只在你实际补齐对应实现或报告后使用：

- 基于 Debezium + Kafka 设计 MySQL -> Neo4j 实时同步链路，支持商品、品牌、品类和属性数据变更增量写入图谱。
- 基于 Milvus + BGE-Reranker 搭建向量召回与重排链路，采用 RRF 融合 BM25、向量召回和图谱回查结果。
- 构建商品属性 QA 测试集，使用 Recall@5、MRR、Precision@5 评估检索效果，相比 BM25 单路检索 Recall@5 提升 17.6%，Precision@5 提升 10.8%。

## 面试讲解主线

1. 为什么做：电商问答里有大量长尾属性和别名，纯关键词召回不稳，纯 LLM 容易幻觉。
2. 数据怎么来：MySQL 业务表同步基础商品图谱，Label Studio 标注商品文本，NER 抽取长尾标签。
3. 图谱怎么建：节点包括商品、品牌、品类、属性、标签；关系包括归属、拥有、商品到标签。
4. 检索怎么做：LLM 先生成参数化 Cypher，实体值不直接信任，先走 hybrid retrieval 对齐成图谱里的真实节点名。
5. 答案怎么生成：Neo4j 查询返回结构化结果，LLM 只基于查询结果组织自然语言回答。
6. 你做了什么：数据同步、NER 训练、实体对齐、GraphRAG 服务封装、FastAPI 和前端 demo。
7. 下一步怎么优化：增量同步、Milvus 独立向量库、检索评估、Cypher 安全约束、Docker Compose。

## 常见追问

### 为什么不用纯 RAG？

商品、品牌、品类和属性之间有明确关系，纯文档 RAG 很难稳定表达“属于哪个类目”“有哪些属性值”“SKU 和 SPU 的关系”。图谱适合结构化关系查询，向量检索适合实体别名和模糊召回，所以这里用 Hybrid Retrieval 做实体对齐，用 GraphRAG 做可解释查询。

### LLM 生成 Cypher 不安全吗？

当前实现已经要求生成参数化 Cypher，并把实体值单独抽出后再对齐。生产化还需要加只读语句校验、schema 白名单、超时限制、错误重试和模板化查询兜底。

### NER 模型解决什么问题？

MySQL 商品表里的结构化属性覆盖不了所有长尾卖点，例如材质、风格、规格、适用场景等自然语言描述。NER 模型从商品描述里抽取这些标签，写回图谱后可以作为补充知识点参与问答。

### 为什么需要实体对齐？

用户问题里的实体可能是别名、英文、简称或错别字，例如 `apple`、`苹果手机`、具体 SKU 名称不完全一致。Hybrid Retrieval 用向量相似度和全文索引一起召回图谱中的标准实体，降低 Cypher 参数查不到结果的概率。

### 如果面试官问指标怎么办？

当前仓库更适合讲端到端链路和工程实现。如果要写量化提升，需要补一个固定 QA/检索测试集，对 BM25、向量召回、Hybrid Retrieval 分别计算 Recall@5、MRR、Precision@5，再把实验脚本和报告提交到仓库。
