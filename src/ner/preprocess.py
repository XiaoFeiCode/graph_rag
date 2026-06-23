# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


from datasets import load_dataset
from transformers import AutoTokenizer

from dotenv import load_dotenv
from src.configuration.config import LABELS, MODEL_NAME, PROCESSED_DATA_DIR, RAW_DATA_FILE

# 加载环境变量
load_dotenv()


def process():
    # 1. 读取数据
    dataset = load_dataset('json', data_files=RAW_DATA_FILE)['train']
    # print(dataset)
    # 2. 去除多余列
    dataset = dataset.remove_columns(['id', 'annotator', 'annotation_id', 'created_at', 'updated_at', 'lead_time'])
    # 3. 划分数据集
    dataset_dict = dataset.train_test_split(test_size=0.2)
    dataset_dict['test'], dataset_dict['valid'] = dataset_dict['test'].train_test_split(test_size=0.5).values()

    # 4. 定义分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 5. 数据编码
    def encoder(example):
        # 1. 将文本数据转成字符列表 相当于变成一个一个的
        tokens = list(example['text'])
        # 2. 文本编码
        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=512  # 显式指定一个长度
        )

        # 3. 添加一个labels列 实体标注 实体就是里面 label
        entities = example['label']
        # 4. 定义标注列表, 存放所有的id，默认存放O的id
        labels = [LABELS.index('O')] * len(tokens)

        # 5. 遍历每个TAG，标记为‘B’和‘I'的id
        for entity in entities:
            start = entity['start']
            end = entity['end']
            labels[start:end] = [LABELS.index('B')] + [LABELS.index('I')] * (end - start - 1)

        # 6. 对齐labels
        aligned_labels = []
        word_ids = inputs.word_ids()
        for word_idx in word_ids:
            if word_idx is None:
                # 特殊字符（如 [CLS], [SEP], [PAD]）统一给 -100
                aligned_labels.append(-100)
            else:
                # 正常字符，取对应的原始标签
                aligned_labels.append(labels[word_idx])

        inputs['labels'] = aligned_labels

        return inputs

    # 一条一条的进来 删除原始列
    dataset_dict = dataset_dict.map(encoder, remove_columns=['text', 'label'])

    print(dataset_dict['train'][0])

    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)


if __name__ == '__main__':
    process()
