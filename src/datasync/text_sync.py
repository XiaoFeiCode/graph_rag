import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.configuration.config import CHECKPOINTS_DIR, NER_DIR
from src.datasync.utils import MySQLReader, Neo4jWriter
from src.ner.predict import Predictor


class TextSync:
    def __init__(self):
        self.reader = MySQLReader()
        self.writer = Neo4jWriter()
        self.extractor = self._init_extractor()

    # 内部函数，初始化模型
    def _init_extractor(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AutoModelForTokenClassification.from_pretrained(CHECKPOINTS_DIR / NER_DIR / 'best_model')
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINTS_DIR / NER_DIR / 'best_model')
        return Predictor(model, tokenizer, device)

    # 同步TAG标签
    def sync_tag(self):
        # 提取商品描述信息
        sql = """
              select id, description
              from gmall.spu_info \
              """

        spu_desc = self.reader.query(sql)
        # 拆分id和description
        ids = [item['id'] for item in spu_desc]
        desc = [item['description'] for item in spu_desc]

        # 提取所有数据的TAG列表
        tags_list = self.extractor.extract(desc)

        # for id, tags in zip(ids, tags_list):
        #     print(id, tags)

        # 构建TAG节点的属性（id, name） 以及 spu-tag关系
        tags_properties = []
        tags_relationship = []

        # 遍历当前的spu的每个标签
        for id, tags in zip(ids, tags_list):
            # 1-1 , 1-2....
            for index, tag in enumerate(tags):
                tag_id = '-'.join([str(id), str(index)])
                property = {
                    'id': tag_id,
                    'name': tag
                }
                tags_properties.append(property)
                # 构建关系
                relationship = {
                    'start_id': id,
                    'end_id': tag_id
                }
                tags_relationship.append(relationship)

        self.writer.write_node('Tag', tags_properties)
        self.writer.write_relationship('Have', 'SPU', 'Tag', tags_relationship)

if __name__ == '__main__':
    sync = TextSync()
    sync.sync_tag()
