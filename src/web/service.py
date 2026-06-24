"""GraphRAG chat service with template-based Cypher generation.

LLM classifies intent → matches a pre-defined Cypher template → fills extracted
entities via Hybrid Retrieval → executes safe, parameterised query.
"""

import json
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase

from src.configuration.config import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, NEO4J_CONFIG
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.search_factory import (
    make_milvus_vector_searcher,
    make_neo4j_fulltext_searcher,
    make_neo4j_vector_searcher,
)
from src.web.cypher_guard import UnsafeCypherError, ensure_declared_params, validate_readonly_cypher
from src.web.cypher_templates import (
    CYPHER_TEMPLATES,
    get_template,
    get_template_descriptions,
)

logger = logging.getLogger(__name__)


class EntityAlignmentError(ValueError):
    """Raised when a query entity cannot be aligned to any graph node."""


def _try_init_milvus():
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

        self.graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_pass)
        self._driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pass))

        self.embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.llm = ChatDeepSeek(model=LLM_MODEL_NAME)
        self._milvus = _try_init_milvus()

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

        self.str_parser = StrOutputParser()

        logger.info(
            "ChatService ready: %d retrievers (sources: neo4j_ft + neo4j_vec%s).",
            len(self._retrievers),
            " + milvus" if self._milvus else "",
        )

    # ── Public API ──────────────────────────────────────────────

    def chat(self, question: str) -> str:
        try:
            intent = self._parse_intent(question)
            template_name = intent["template"]
            raw_entities = intent.get("entities", [])

            template = get_template(template_name)
            if template is None:
                return self._chat_fallback(question)

            aligned = self._align_template_entities(template, raw_entities)
            cypher = template["cypher"]

            # Subgraph recall: use multi-hop expansion
            if cypher == "__SUBGRAPH_EXPAND__":
                entity_type = raw_entities[0].get("label", "SPU") if raw_entities else "SPU"
                result = self._expand_subgraph(entity_type, aligned)
            else:
                validate_readonly_cypher(cypher)
                result = self._execute_cypher(cypher, aligned)
            logger.info("Template %s returned %d rows.", template_name, len(result))

            if not result:
                return "没有在商品知识图谱中查询到匹配信息，请尝试其他品牌、品类或商品描述。"

            return self._generate_answer(question, result)

        except UnsafeCypherError as error:
            logger.warning("Blocked unsafe Cypher: %s", error)
            return "当前问题生成的图查询未通过安全校验，请换一种问法或补充商品范围。"
        except EntityAlignmentError as error:
            logger.warning("Entity alignment failed: %s", error)
            return f"未能识别到「{error}」，请尝试使用更具体的品牌或商品名称。"
        except Exception:
            logger.exception("GraphRAG chat failed.")
            return "系统暂时无法完成图谱问答，请稍后重试。"

    # ── Intent parsing ──────────────────────────────────────────

    def _parse_intent(self, question: str) -> dict:
        template_names = get_template_descriptions()
        prompt = PromptTemplate.from_template("""
你是一个电商知识图谱查询分析器。根据用户问题，从以下预定义模板中选择最匹配的一个，并抽取出需要对齐的实体。

可用模板：
{template_descriptions}

用户问题：{question}

要求：
1. 从可用模板中选一个最匹配的 template name
2. 根据模板的 entity_slots，从用户问题中抽取原始实体值
3. entity.label 只能是 Trademark、SPU、SKU、Category1、Category2、Category3 之一
4. 如果问题无法匹配任何模板，template 填 "none"
5. 严格输出 JSON，格式：
{{
  "template": "模板名称 或 none",
  "entities": [
    {{"param_name": "参数名", "entity": "原始实体文本", "label": "实体类型 或 null"}}
  ]
}}""")
        prompt_str = prompt.format(
            question=question,
            template_descriptions=template_names,
        )
        output = self.llm.invoke(prompt_str)
        raw = output.content if hasattr(output, "content") else str(output)
        return self._parse_json(raw)

    def _parse_json(self, text: str) -> dict:
        text = text.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)

    # ── Template entity alignment ───────────────────────────────

    def _align_template_entities(self, template: dict, raw_entities: list[dict]) -> dict:
        """Align raw entities using HybridRetriever; return {param_name: aligned_value}."""
        aligned = {}
        for slot in raw_entities:
            param = slot["param_name"]
            text = slot.get("entity", "")
            label = slot.get("label")

            if label is None or label not in self.SUPPORTED_ENTITY_LABELS:
                # keyword slot — use as-is
                aligned[param] = text
                continue

            hits = self._retrievers[label].retrieve(text, top_k=1)
            if not hits:
                raise EntityAlignmentError(f"No aligned entity found for {label}: {text}")
            aligned[param] = hits[0].name

        # Add defaults for missing optional slots
        for slot_def in template.get("entity_slots", []):
            if slot_def["param_name"] not in aligned:
                aligned[slot_def["param_name"]] = ""
        return aligned

    # ── Fallback: LLM-generated Cypher (for unmatched intents) ──

    def _chat_fallback(self, question: str) -> str:
        try:
            question_cypher = self._generate_cypher_fallback(question)
            cypher = validate_readonly_cypher(question_cypher["cypher_query"])
            entities_to_align = question_cypher.get("entities_to_align", [])

            aligned_list = []
            for entity in entities_to_align:
                entity_type = entity["label"]
                if entity_type not in self.SUPPORTED_ENTITY_LABELS:
                    raise UnsafeCypherError(f"Unsupported entity label: {entity_type}.")
                hits = self._retrievers[entity_type].retrieve(entity["entity"], top_k=1)
                if not hits:
                    raise EntityAlignmentError(f"No aligned entity for {entity_type}: {entity['entity']}")
                aligned_list.append({"param_name": entity["param_name"], "entity": hits[0].name})

            params = {e["param_name"]: e["entity"] for e in aligned_list}
            ensure_declared_params(cypher, params)
            result = self.graph.query(cypher, params=params)

            if not result:
                return "没有在商品知识图谱中查询到足够信息。"
            return self._generate_answer(question, result)

        except UnsafeCypherError as error:
            logger.warning("Fallback Cypher blocked: %s", error)
            return "当前问题较复杂，请尝试更具体的品牌或商品名称。"
        except EntityAlignmentError as error:
            logger.warning("Fallback alignment failed: %s", error)
            return f"未能识别到相关信息，请尝试使用更具体的品牌或商品名称。"
        except Exception:
            logger.exception("Fallback GraphRAG failed.")
            return "系统暂时无法完成图谱问答，请稍后重试。"

    def _generate_cypher_fallback(self, question):
        template = """
你是一个专业的Neo4j Cypher查询生成器。

用户问题：{question}
知识图谱结构信息：{schema_info}

要求：
1. 只读查询，禁止 CREATE/MERGE/SET/DELETE/REMOVE/DROP/LOAD/CALL
2. 参数化：用 $param_0, $param_1 代替具体值
3. 识别需对齐的实体，label 只能从 Trademark、SPU、SKU、Category1、Category2、Category3 中选择
4. JSON 格式输出：
{{
 "cypher_query": "Cypher语句",
 "entities_to_align": [
  {{"param_name": "param_0", "entity": "原始实体名称", "label": "节点类型"}}
 ]
}}"""
        prompt = PromptTemplate.from_template(template)
        prompt = prompt.format(question=question, schema_info=self.graph.schema)
        output = self.llm.invoke(prompt)
        raw = output.content if hasattr(output, "content") else str(output)
        return self._parse_json(raw)

    # ── Subgraph expansion ──────────────────────────────────────

    def _expand_subgraph(self, entity_type: str, params: dict) -> list[dict]:
        """Multi-hop graph expansion from a starting entity."""
        from src.retrieval.subgraph_expander import get_expansion_cypher

        entity_type = entity_type if entity_type in self.SUPPORTED_ENTITY_LABELS else "SPU"
        cypher = get_expansion_cypher(entity_type)
        if not cypher:
            return []
        return self.graph.query(cypher, params=params)

    # ── Shared ──────────────────────────────────────────────────

    def _execute_cypher(self, cypher_query, params):
        ensure_declared_params(cypher_query, params)
        return self.graph.query(cypher_query, params=params)

    def _generate_answer(self, question, results):
        prompt = PromptTemplate.from_template("""
你是一个电商智能客服，根据用户问题和数据库查询结果生成简洁、准确的自然语言回答。
用户问题: {question}
数据库返回结果: {query_result}""")
        output = self.llm.invoke(prompt.format(question=question, query_result=results))
        return self.str_parser.invoke(output)
