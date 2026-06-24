import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 1. 目录路径
ROOT_DIR = Path(__file__).parent.parent.parent

DATA_DIR = ROOT_DIR / 'data'
NER_DIR = 'ner'
RAW_DATA_DIR = DATA_DIR / NER_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / NER_DIR / 'processed'

LOGS_DIR = ROOT_DIR / 'logs'
CHECKPOINTS_DIR = ROOT_DIR / 'checkpoints'
MODELS_DIR = ROOT_DIR / 'models'

# Web 静态目录
WEB_STATIC_DIR = ROOT_DIR / 'src' / 'web' / 'static'

# 2. 数据文件名和模型名称
RAW_DATA_FILE = str(RAW_DATA_DIR / 'data.json')
MODEL_NAME = os.getenv("MODEL_NAME", "google-bert/bert-base-chinese")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-zh-v1.5")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
UIE_MODEL_ID = os.getenv("UIE_MODEL_ID", "iic/nlp_structbert_siamese-uie_chinese-base")
UIE_MODEL_DIR = ROOT_DIR / os.getenv("UIE_MODEL_DIR", "models/uie")

# 3. 超参数
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 7e-6

SAVE_STEPS = 20

# 4. NER任务的分类标签
LABELS = ['B', 'I', 'O']

# 5. 数据库连接
MYSQL_CONFIG = {
    'host': os.getenv("MYSQL_HOST", "localhost"),
    'port': int(os.getenv("MYSQL_PORT", "3306")),
    'user': os.getenv("MYSQL_USER", "root"),
    'password': os.getenv("MYSQL_PASSWORD", ""),
    'db': os.getenv("MYSQL_DB", "gmall"),
}

NEO4J_CONFIG = {
    'uri': os.getenv("NEO4J_URI", "neo4j://localhost"),
    'auth': (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "graph_rag_demo"))
}

CDC_CONFIG = {
    "bootstrap_servers": os.getenv("CDC_KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
    "group_id": os.getenv("CDC_KAFKA_GROUP_ID", "graph-rag-cdc"),
    "topic_prefix": os.getenv("CDC_TOPIC_PREFIX", "gmall"),
}

DEBEZIUM_CONFIG = {
    "connect_url": os.getenv("DEBEZIUM_CONNECT_URL", "http://localhost:8083"),
    "connector_name": os.getenv("DEBEZIUM_CONNECTOR_NAME", "gmall-mysql-connector"),
    "database_host": os.getenv("DEBEZIUM_DATABASE_HOST", "mysql"),
    "database_port": os.getenv("DEBEZIUM_DATABASE_PORT", "3306"),
    "database_server_id": os.getenv("DEBEZIUM_DATABASE_SERVER_ID", "184054"),
}

MILVUS_CONFIG = {
    "backend": os.getenv("ENTITY_RETRIEVAL_BACKEND", "neo4j"),
    "uri": os.getenv("MILVUS_URI", "http://localhost:19530"),
    "token": os.getenv("MILVUS_TOKEN", ""),
    "collection": os.getenv("MILVUS_COLLECTION", "commerce_entities"),
    "embedding_dim": int(os.getenv("MILVUS_EMBEDDING_DIM", "1024")),
    "metric_type": os.getenv("MILVUS_METRIC_TYPE", "COSINE"),
}

UIE_TRAINING_CONFIG = {
    "model_id": UIE_MODEL_ID,
    "model_dir": UIE_MODEL_DIR,
    "max_length": int(os.getenv("UIE_MAX_LENGTH", "256")),
    "batch_size": int(os.getenv("UIE_BATCH_SIZE", "16")),
    "epochs": int(os.getenv("UIE_EPOCHS", "5")),
    "learning_rate": float(os.getenv("UIE_LEARNING_RATE", "2e-5")),
    "device": os.getenv("UIE_DEVICE", "cuda"),
}
