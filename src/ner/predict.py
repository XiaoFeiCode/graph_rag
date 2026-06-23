from dotenv import load_dotenv
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from src.configuration.config import CHECKPOINTS_DIR, NER_DIR

load_dotenv()


class Predictor:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        # 将映射关系存入 predictor 方便调用
        self.id2label = self.model.config.id2label

    def predict(self, texts: str | list[str]):
        # 1. 记录原始输入是否为单个字符串
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        # 2. 预分词，得到字符列表
        token_list = [list(text) for text in texts]

        # 3. 编码
        token_tensor = self.tokenizer(
            token_list,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # 4. 加载到设备
        token_tensor = {key: value.to(self.device) for key, value in token_tensor.items()}

        # 5. 预测
        with torch.no_grad():
            outputs = self.model(**token_tensor)
            logits = outputs.logits
            predictions = logits.argmax(dim=-1).tolist()

        # 6. 转换为标签
        final_results = []
        for token, prediction in zip(token_list, predictions):
            # 获取 word_ids 以便精确对齐（比手动切片 [1:len+1] 更稳健）
            # 但如果你确定是 BERT 且没有特殊字符，你的切片法也可以
            valid_prediction = prediction[1: len(token) + 1]
            final_results.append([self.id2label[p] for p in valid_prediction])

        # 7. 根据原始输入类型返回
        return final_results[0] if is_single else final_results

    # 抽取实体
    def extract(self, texts: str | list[str]):
        # 1. 记录原始输入是否为单个字符串
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        # 得到预测标签列表
        predictions = self.predict(texts)
        # 从当前列表中抽取实体列表
        entities_list = []
        for labels, text in zip(predictions, texts):
            # 调用内部函数，抽取一个样本的所有实体标签
            entities = self._extract_entities(labels, list(text))
            entities_list.append(entities)

        if is_single:
            return entities_list[0]
        else:
            return entities_list

    def _extract_entities(self, labels, tokens):
        entities = []
        current_entity = ""
        for label, token in zip(labels, tokens):
            # 如果是B标签，则开始一个新的实体
            if label == 'B':
                # 如果不为空，说明里面有一个实体正在抽取
                if current_entity:
                    entities.append(current_entity)
                current_entity = token
            # 如果是I标签，则继续添加实体
            elif label == 'I':
                if current_entity:  # 只有前面有 B 才接续
                    current_entity += token
                else:
                    # I 前没有 B，直接跳过
                    continue
            else:
                # 前面有B, 才结束
                if current_entity:
                    entities.append(current_entity)
                    current_entity = ""

        if current_entity:
            entities.append(current_entity)

        return entities


def predict():
    # 建议使用绝对路径确保服务器运行正常
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        model = AutoModelForTokenClassification.from_pretrained(CHECKPOINTS_DIR / NER_DIR / 'best_model')
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINTS_DIR / NER_DIR / 'best_model')
    except Exception as e:
        print(f"模型加载失败，请检查路径：{e}")
        return

    predictor = Predictor(model, tokenizer, device=device)

    # 抽取实体
    text = ["380克x3袋装韩国风味炒粘糕辣酱炒年糕条韩式部队火锅辣酱",
            "2018秋冬季新款韩版平底高帮鞋女休闲二棉鞋加绒运动厚底高邦鞋潮"]
    print(predictor.extract(text))


if __name__ == '__main__':
    predict()
