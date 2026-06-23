from datasets import load_from_disk
from transformers import Trainer, AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, \
    EvalPrediction

from src.configuration.config import CHECKPOINTS_DIR, LABELS, NER_DIR, PROCESSED_DATA_DIR

# 1. 分词器
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINTS_DIR / NER_DIR / 'best_model')

# 训好的模型里面有这两个
# # 标签映射
# id2label = {k: v for k, v in enumerate(LABELS)}
# label2id = {v: k for k, v in id2label.items()}

# 2. 模型
model = AutoModelForTokenClassification.from_pretrained(CHECKPOINTS_DIR / NER_DIR / 'best_model',
                                                        num_labels=len(LABELS))

# 3. 数据集
test_dataset = load_from_disk(PROCESSED_DATA_DIR / 'test')

# 数据整理器
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    return_tensors='pt'  # 会自动转换
)

import evaluate

metric = evaluate.load('seqeval')


# 评估函数
def compute_metrics(prediction: EvalPrediction):
    predictions, labels = prediction.predictions, prediction.label_ids
    # 1. 在最后一个维度取最大值，把概率变成分类ID
    # predictions 形状: (batch_size, seq_len, num_labels) -> (batch_size, seq_len)
    predictions = predictions.argmax(axis=-1)

    # 2. 将 ID 转回文字标签，并剔除 -100
    true_predictions = [
        # 遍历每个标签
        [model.config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        # 遍历每个句子
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [model.config.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # 3. 计算指标
    results = metric.compute(predictions=true_predictions, references=true_labels)

    # 4. 整理返回格式
    return results


# 定义训练器
trainer = Trainer(
    model=model,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 验证 输出一个字典对象
results = trainer.evaluate()

print(results)
