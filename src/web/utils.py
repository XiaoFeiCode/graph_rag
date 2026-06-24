from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph

from src.configuration.config import EMBEDDING_MODEL_NAME, NEO4J_CONFIG


class IndexUtils:
    def __init__(self):
        self.graph = Neo4jGraph(
            url=NEO4J_CONFIG['uri'],
            username=NEO4J_CONFIG['auth'][0],
            password=NEO4J_CONFIG['auth'][1]
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={'normalize_embeddings': True}
        )

    def create_fulltext_index(self, index_name, label, property):
        cypher = f"""
        CREATE FULLTEXT INDEX {index_name} FOR (n:{label}) ON EACH [n.{property}]
        """
        self.graph.query(cypher)

    def create_embedding_index(self, index_name, label, source_property, embedding_property):
        embedding_dim = self._add_embedding(label, source_property, embedding_property)

        cypher = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{label}) ON (n.{embedding_property})
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {embedding_dim},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
        self.graph.query(cypher)

    def _add_embedding(self, label, source_property, embedding_property):
        query = f"""
             MATCH (n:{label})
             RETURN id(n) as id, n.{source_property} as text
         """
        results = self.graph.query(query)
        docs = [result["text"] for result in results]
        embeddings = self.embedding_model.embed_documents(docs)

        batch = []
        for result, embedding in zip(results, embeddings):
            batch.append({"id": result["id"], "embedding": embedding})

        cypher = f"""
            UNWIND $batch AS item MATCH (n:{label}) WHERE id(n) = item.id SET n.{embedding_property} = item.embedding
        """
        self.graph.query(cypher, params={"batch": batch})
        return len(embeddings[0])
