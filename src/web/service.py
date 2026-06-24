import logging

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase

from src.configuration.config import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, MILVUS_CONFIG, NEO4J_CONFIG
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.search_factory import (
    make_milvus_vector_searcher,
    make_neo4j_fulltext_searcher,
    make_neo4j_vector_searcher,
)
from src.web.cypher_guard import UnsafeCypherError, ensure_declared_params, validate_readonly_cypher

logger = logging.getLogger(__name__)


class EntityAlignmentError(ValueError):
    """Raised when a query entity cannot be aligned to any graph node."""


def _try_init_milvus():
    """Try to initialise Milvus store; return None if unavailable."""
    try:
        from src.retrieval.milvus_entity_store import MilvusEntityStore

        store = MilvusEntityStore()
        store.ensure_collection()
        return store
    except Exception:
        logger.warning("Milvus unavailable, falling back to Neo4j-only retrieval.")
        return None


class ChatService:
    SUPPORTED_ENTITY_LABELS = {"Trademark", "SPU", "SKU", "Category1", "Category2", "Category3"}

    _RETRIEVER_LABELS = SUPPORTED_ENTITY_LABELS

    def __init__(self):
        neo4j_url = NEO4J_CONFIG["uri"]
        neo4j_user = NEO4J_CONFIG["auth"][0]
        neo4j_pass = NEO4J_CONFIG["auth"][1]

        # Graph connection for schema + Cypher execution
        self.graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_pass)

        # Separate driver for raw search operations
        self._driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pass))

        # Embedding model shared by all retrievers
        self.embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )

        self.llm = ChatDeepSeek(model=LLM_MODEL_NAME)

        # Milvus (optional)
        self._milvus = _try_init_milvus()

        # Build one HybridRetriever per entity label
        self._retrievers: dict[str, HybridRetriever] = {}
        for label in self._RETRIEVER_LABELS:
            sources = [
                make_neo4j_fulltext_searcher(self._driver, label),
                make_neo4j_vector_searcher(self._driver, label, self.embedding),
            ]
            if self._milvus is not None:
                sources.append(make_milvus_vector_searcher(self._milvus, self.embedding))
            self._retrievers[label] = HybridRetriever(
                neo4j_fulltext_search=sources[0],
                neo4j_vector_search=sources[1],
                milvus_vector_search=sources[2] if len(sources) > 2 else None,
            )

        self.json_parser = JsonOutputParser()
        self.str_parser = StrOutputParser()

        logger.info(
            "ChatService ready: %d retrievers (sources: neo4j_ft + neo4j_vec%s).",
            len(self._retrievers),
            " + milvus" if self._milvus else "",
        )

    def chat(self, question: str) -> str:
        try:
            question_cypher = self._get_question_cypher(question)
            cypher = validate_readonly_cypher(question_cypher["cypher_query"])
            entities_to_align = question_cypher.get("entities_to_align", [])
            logger.info("Generated Cypher: %s", cypher)
            logger.info("Entities before alignment: %s", entities_to_align)

            aligned_entities = self._align_entities(entities_to_align)
            logger.info("Entities after alignment: %s", aligned_entities)

            result = self._execute_cypher(cypher, aligned_entities)
            logger.info("Query returned %d rows.", len(result))

            if not result:
                return "没有在商品知识图谱中查询到足够信息，请换一种商品、品牌或属性描述。"

            return self._generate_answer(question, result)
        except UnsafeCypherError as error:
            logger.warning("Blocked unsafe generated Cypher: %s", error)
            return "当前问题生成的图查询未通过安全校验，请换一种问法或补充商品范围。"
        except EntityAlignmentError as error:
            logger.warning("Entity alignment failed: %s", error)
            return f"未能识别到「{error}」，请尝试使用更具体的品牌或商品名称。"
        except Exception:
            logger.exception("GraphRAG chat failed.")
            return "系统暂时无法完成图谱问答，请稍后重试。"

    def _get_question_cypher(self, question):
        template = """
                你是一个专业的Neo4j Cypher查询生成器。你的任务是根据用户问题生成一条Cypher查询语句，用于从知识图谱中获取回答用户问题所需的信息。

                用户问题：{question}

                知识图谱结构信息：{schema_info}

                要求：
                1. 只能生成只读查询，允许 MATCH、OPTIONAL MATCH、WITH、UNWIND、RETURN，不允许 CREATE、MERGE、SET、DELETE、REMOVE、DROP、LOAD、CALL
                2. 生成参数化Cypher查询语句，用 $param_0, $param_1 等代替具体值
                3. 识别需要对齐的实体，label 只能从 Trademark、SPU、SKU、Category1、Category2、Category3 中选择
                4. 必须严格使用以下JSON格式输出结果
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
        prompt = prompt.format(question=question, schema_info=self.graph.schema)
        output = self.llm.invoke(prompt)
        return self.json_parser.invoke(output)

    def _align_entities(self, entities_to_align):
        for index, entity in enumerate(entities_to_align):
            entity_name = entity["entity"]
            entity_type = entity["label"]
            if entity_type not in self.SUPPORTED_ENTITY_LABELS:
                raise UnsafeCypherError(f"Unsupported entity label: {entity_type}.")
            retriever = self._retrievers[entity_type]
            hits = retriever.retrieve(entity_name, top_k=1)
            if not hits:
                raise EntityAlignmentError(
                    f"No aligned entity found for {entity_type}: {entity_name}"
                )
            entities_to_align[index]["entity"] = hits[0].name
        return entities_to_align

    def _execute_cypher(self, cypher_query, aligned_entities):
        params = {entity["param_name"]: entity["entity"] for entity in aligned_entities}
        ensure_declared_params(cypher_query, params)
        return self.graph.query(cypher_query, params=params)

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
