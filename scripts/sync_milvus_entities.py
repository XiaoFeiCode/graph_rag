from __future__ import annotations

import argparse

from langchain_huggingface import HuggingFaceEmbeddings

from scripts.neo4j_client import neo4j_driver
from src.configuration.config import MILVUS_CONFIG
from src.retrieval.milvus_entity_store import EntityRecord, MilvusEntityStore

ENTITY_LABELS = ["Trademark", "SPU", "SKU", "Category1", "Category2", "Category3", "Tag"]


def load_entities() -> list[EntityRecord]:
    entities = []
    with neo4j_driver() as driver:
        for label in ENTITY_LABELS:
            rows = driver.execute_query(f"MATCH (n:{label}) RETURN n.id AS id, n.name AS name").records
            entities.extend(
                EntityRecord(entity_id=f"{label}:{row['id']}", label=label, name=row["name"])
                for row in rows
                if row["id"] and row["name"]
            )
    return entities


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Neo4j entity names into Milvus.")
    parser.add_argument("--model-name", default="BAAI/bge-large-zh-v1.5")
    args = parser.parse_args()

    entities = load_entities()
    embedding = HuggingFaceEmbeddings(
        model_name=args.model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    vectors = embedding.embed_documents([entity.name for entity in entities])
    store = MilvusEntityStore()
    store.upsert(entities, vectors)
    print(f"Synced {len(entities)} entities into Milvus collection `{MILVUS_CONFIG['collection']}`.")


if __name__ == "__main__":
    main()
