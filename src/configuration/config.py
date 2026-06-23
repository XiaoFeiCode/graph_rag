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
AUTO_DIR = "/root/tf-logs/"
CHECKPOINTS_DIR = ROOT_DIR / 'checkpoints'

# Web 静态目录
WEB_STATIC_DIR = ROOT_DIR / 'src' / 'web' / 'static'

# 2. 数据文件名和模型名称
RAW_DATA_FILE = str(RAW_DATA_DIR / 'data.json')
MODEL_NAME = os.getenv("MODEL_NAME", "google-bert/bert-base-chinese")

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
    'auth': (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
}
