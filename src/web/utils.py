from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph

from src.configuration.config import NEO4J_CONFIG


# 创建索引
class IndexUtils:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_CONFIG['uri'],
            username=NEO4J_CONFIG['auth'][0],
            password=NEO4J_CONFIG['auth'][1]
        )
        # 嵌入模型
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            encode_kwargs={'normalize_embeddings': True}  # 归一化向量，算余弦相似度必须开
        )

    # 创建全文索引，传入索引名称，节点标签，属性
    def create_fulltext_index(self, index_name, label, property):
        cypher = f"""
        CREATE FULLTEXT INDEX {index_name} FOR (n:{label}) ON EACH [n.{property}]
        """
        self.graph.query(cypher)

    # 创建向量索引 需要传入生成向量的原属性和向量属性
    def create_embedding_index(self, index_name, label, source_property, embedding_property):
        # 生成嵌入向量，并添加到节点属性中
        embedding_dim = self._add_embedding(label, source_property, embedding_property)

        cypher = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS  //防止重复创建索引
        FOR (n:{label}) ON (n.{embedding_property})
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {embedding_dim},          // 必须和你的模型维度一致，这里假设 embedding_dim 是一个已定义的变量
            `vector.similarity_function`: 'cosine' // 相似度算法：常用 cosine 或 euclidean
          }}
        }}
        """

        self.graph.query(cypher)

    # 生成嵌入向量，添加到节点属性中，返回向量维度
    def _add_embedding(self, label, source_property, embedding_property):
        # 1. 查询需要生成 embedding 的节点，还需要查出来节点id
        query = f"""
             MATCH (n:{label})
             RETURN id(n) as id, n.{source_property} as text
         """
        results = self.graph.query(query)

        # 2. 获取查询结果中的文本内容
        docs = [result["text"] for result in results]  # 获取文本内容

        # 3. 调用嵌入模型，得到嵌入向量
        embeddings = self.embedding_model.embed_documents(docs)

        # 4. 将id和嵌入向量组合成字典形式
        batch = []
        for result, embedding in zip(results, embeddings):
            item = {
                "id": result["id"],
                "embedding": embedding
            }
            batch.append(item)

        # 5. 执行cypher, 按照id查节点，写到新的嵌入式向量
        cypher = f"""
            UNWIND $batch AS item MATCH (n:{label}) WHERE id(n) = item.id SET n.{embedding_property} = item.embedding
        """

        self.graph.query(cypher, params={"batch": batch})

        # 返回向量维度
        return len(embeddings[0])


if __name__ == '__main__':
    index_utils = IndexUtils()
    # index_utils.create_fulltext_index("trademark_fulltext_index", "Trademark", "name")
    # index_utils.create_embedding_index("trademark_embedding_index", "Trademark", "name", "embedding")

    # 混合检索
    # index_name = "trademark_embedding_index"
    # keyword_index_name = "trademark_fulltext_index"
    #
    # store = Neo4jVector.from_existing_index(
    #     embedding=index_utils.embedding_model,  # 负责把用户的提问转成向量
    #     url=NEO4J_CONFIG['uri'],
    #     username=NEO4J_CONFIG['auth'][0],
    #     password=NEO4J_CONFIG['auth'][1],
    #     index_name=index_name,  # 你在 Neo4j 里创建的向量索引名
    #     keyword_index_name=keyword_index_name,  # 你在 Neo4j 里创建的全文索引名
    #     search_type=SearchType.HYBRID  # 开启最强的混合搜索模式
    # )
    #
    # # 3. 开始搜索
    # query = "apple"
    # results = store.similarity_search(query, k=1)  # 返回最像的3条结果
    #
    # for doc in results:
    #     print(f"找到商品: {doc.page_content}")

    index_utils.create_fulltext_index("spu_fulltext_index", "SPU", "name")
    index_utils.create_embedding_index("spu_embedding_index", "SPU", "name", "embedding")
    index_utils.create_fulltext_index("sku_fulltext_index", "SKU", "name")
    index_utils.create_embedding_index("sku_embedding_index", "SKU", "name", "embedding")

    index_utils.create_fulltext_index("category1_fulltext_index", "Category1", "name")
    index_utils.create_embedding_index("category1_embedding_index", "Category1", "name", "embedding")
    index_utils.create_fulltext_index("category2_fulltext_index", "Category2", "name")
    index_utils.create_embedding_index("category2_embedding_index", "Category2", "name", "embedding")
    index_utils.create_fulltext_index("category3_fulltext_index", "Category3", "name")
    index_utils.create_embedding_index("category3_embedding_index", "Category3", "name", "embedding")
