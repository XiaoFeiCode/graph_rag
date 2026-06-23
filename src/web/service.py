from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from neo4j_graphrag.types import SearchType

from dotenv import load_dotenv
from src.configuration.config import NEO4J_CONFIG

load_dotenv()


class ChatService:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_CONFIG['uri'],
            username=NEO4J_CONFIG['auth'][0],
            password=NEO4J_CONFIG['auth'][1]
        )
        self.embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            encode_kwargs={'normalize_embeddings': True}  # 归一化向量，算余弦相似度必须开
        )
        # llm
        self.llm = ChatDeepSeek(model="deepseek-chat")
        # 定义所有实体对应的混合检索Neo4jVector对象
        self.neo4j_vector = {
            "Trademark": Neo4jVector.from_existing_index(
                embedding=self.embedding,
                url=NEO4J_CONFIG['uri'],
                username=NEO4J_CONFIG['auth'][0],
                password=NEO4J_CONFIG['auth'][1],
                index_name="trademark_embedding_index",
                keyword_index_name="trademark_fulltext_index",
                search_type=SearchType.HYBRID
            ),
            "SPU": Neo4jVector.from_existing_index(
                embedding=self.embedding,
                url=NEO4J_CONFIG['uri'],
                username=NEO4J_CONFIG['auth'][0],
                password=NEO4J_CONFIG['auth'][1],
                index_name="spu_embedding_index",
                keyword_index_name="spu_fulltext_index",
                search_type=SearchType.HYBRID
            ),
            "SKU": Neo4jVector.from_existing_index(
                embedding=self.embedding,
                url=NEO4J_CONFIG['uri'],
                username=NEO4J_CONFIG['auth'][0],
                password=NEO4J_CONFIG['auth'][1],
                index_name="sku_embedding_index",
                keyword_index_name="sku_fulltext_index",
                search_type=SearchType.HYBRID
            ),
            "Category1": Neo4jVector.from_existing_index(
                embedding=self.embedding,
                url=NEO4J_CONFIG['uri'],
                username=NEO4J_CONFIG['auth'][0],
                password=NEO4J_CONFIG['auth'][1],
                index_name="category1_embedding_index",
                keyword_index_name="category1_fulltext_index",
                search_type=SearchType.HYBRID
            ),
            "Category2": Neo4jVector.from_existing_index(
                embedding=self.embedding,
                url=NEO4J_CONFIG['uri'],
                username=NEO4J_CONFIG['auth'][0],
                password=NEO4J_CONFIG['auth'][1],
                index_name="category2_embedding_index",
                keyword_index_name="category2_fulltext_index",
                search_type=SearchType.HYBRID
            ),
            "Category3": Neo4jVector.from_existing_index(
                embedding=self.embedding,
                url=NEO4J_CONFIG['uri'],
                username=NEO4J_CONFIG['auth'][0],
                password=NEO4J_CONFIG['auth'][1],
                index_name="category3_embedding_index",
                keyword_index_name="category3_fulltext_index",
                search_type=SearchType.HYBRID
            )
        }
        # 定义Parser
        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()

    # 核心聊天服务流程
    def chat(self, question):
        pass
        # 1. 获取用户问题，生成cypher以及需要对齐的实体
        question_cypher = self._get_question_cypher(question)
        cypher = question_cypher["cypher_query"]
        entities_to_align = question_cypher["entities_to_align"]
        print(cypher)
        print("对齐之前的实体名称：", entities_to_align)

        # 2. 实体对齐，混合检索
        aligned_entities = self._align_entities(entities_to_align)
        print("对齐之后的实体名称：", aligned_entities)

        # 3. 执行cypher，获取结果
        result = self._execute_cypher(cypher, aligned_entities)
        print("查询结果：", result)

        # 4. 结合结果，生成答案
        answer = self._generate_answer(question, result)
        print("生成的答案：", answer)
        return answer

    # 获取用户问题，生成cypher以及需要对齐的实体
    def _get_question_cypher(self, question):
        # 提示词
        template = """
                你是一个专业的Neo4j Cypher查询生成器。你的任务是根据用户问题生成一条Cypher查询语句，用于从知识图谱中获取回答用户问题所需的信息。

                用户问题：{question}

                知识图谱结构信息：{schema_info}

                要求：
                1. 生成参数化Cypher查询语句，用param_0, param_1等代替具体值
                2. 识别需要对齐的实体
                3. 必须严格使用以下JSON格式输出结果
                {{
                 "cypher_query": "生成的Cypher语句",
                 "entities_to_align": [
                  {{
                   "param_name": "param_0",
                   "entity": "原始实体名称",
                   "label": "节点类型"
                  }}
                 ]
                }}"""
        prompt = PromptTemplate.from_template(template)
        # 拿到表结构
        prompt = prompt.format(question=question, schema_info=self.graph.schema)
        # 输出
        output = self.llm.invoke(prompt)
        # 解析
        return self.json_parser.invoke(output)

    # 实体对齐
    def _align_entities(self, entities_to_align):
        for index, entity in enumerate(entities_to_align):
            # 获取实体名称
            entity_name = entity["entity"]
            # 获取实体类型
            entity_type = entity["label"]
            # 获取实体的混合检索对象
            neo4j_vector = self.neo4j_vector[entity_type]
            # 混合检索
            results = neo4j_vector.similarity_search(entity_name, k=1)
            aligned_entity = results[0].page_content
            # 覆盖原来的实体名称
            entities_to_align[index]["entity"] = aligned_entity

        return entities_to_align

    # 执行Cypher
    def _execute_cypher(self, cypher_query, aligned_entities):
        # 提取对齐后的实体
        params = {entity['param_name']: entity['entity'] for entity in aligned_entities}
        return self.graph.query(cypher_query, params=params)

    # 结合结果，生成答案
    def _generate_answer(self, question, results):
        prompt = PromptTemplate.from_template(
            """
               你是一个电商智能客服，根据用户问题，以及数据库查询结果生成一段简洁、准确的自然语言回答。
               用户问题: {question}
               数据库返回结果: {query_result}
             """
        )

        prompt = prompt.format(question=question, query_result=results)
        output = self.llm.invoke(prompt)
        return self.str_parser.invoke(output)


if __name__ == '__main__':
    service = ChatService()
    service.chat("请给我推荐一个apple手机")
