import time

from datasets import load_from_disk

from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, \
    EvalPrediction, EarlyStoppingCallback
from src.configuration.config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EPOCHS,
    LABELS,
    LEARNING_RATE,
    LOGS_DIR,
    MODEL_NAME,
    NER_DIR,
    PROCESSED_DATA_DIR,
    SAVE_STEPS,
)

# 1. 分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 标签映射
id2label = {k: v for k, v in enumerate(LABELS)}
label2id = {v: k for k, v in id2label.items()}

# 2. 模型
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS), id2label=id2label,
                                                        label2id=label2id)

# 3. 数据集
train_dataset = load_from_disk(PROCESSED_DATA_DIR / 'train')
valid_dataset = load_from_disk(PROCESSED_DATA_DIR / 'valid')

# 创建训练器
from transformers import Trainer, TrainingArguments

# 训练参数
training_args = TrainingArguments(
    # 输出目录
    output_dir=str(CHECKPOINTS_DIR / NER_DIR),
    logging_dir=str(LOGS_DIR),

    # 训练强度
    num_train_epochs=EPOCHS,  # 练 5 轮通常效果较好
    learning_rate=LEARNING_RATE,  # 经典 NER 学习率
    weight_decay=0.01,
    lr_scheduler_type='linear',

    # 显存管理
    per_device_train_batch_size=BATCH_SIZE,  # 根据你显存大小调整
    gradient_accumulation_steps=1,  # 如果显存小，调大这个
    fp16=True,  # 开启加速

    # 评估策略
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    logging_strategy="steps",
    logging_steps=SAVE_STEPS,  # 经常打印进度

    # 自动择优
    load_best_model_at_end=True,  # 留存最好的模型
    metric_for_best_model="eval_overall_f1",  # 以 F1 为准
    save_total_limit=3,  # 最多存两个，省空间

)

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
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        # 遍历每个句子
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # 3. 计算指标
    results = metric.compute(predictions=true_predictions, references=true_labels)

    # 4. 整理返回格式
    return results


# 早停回调
# early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=10)

# 评估
trainer = Trainer(
    model=model,  # 你加载的模型
    args=training_args,  # 训练参数
    train_dataset=train_dataset,  # 训练集
    eval_dataset=valid_dataset,  # 验证集
    data_collator=data_collator,  # 数据整理器
    compute_metrics=compute_metrics,
    # callbacks=[early_stopping_callback],
)

trainer.train()

# 保存模型
trainer.save_model(CHECKPOINTS_DIR / NER_DIR / 'best_model')
