from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.configuration.config import MILVUS_CONFIG


@dataclass
class EntityRecord:
    entity_id: str
    name: str
    label: str
    source: str = "neo4j"


class MilvusEntityStore:
    def __init__(self):
        from pymilvus import DataType, MilvusClient

        self.client = MilvusClient(uri=MILVUS_CONFIG["uri"], token=MILVUS_CONFIG["token"] or None)
        self.collection = MILVUS_CONFIG["collection"]
        self.embedding_dim = MILVUS_CONFIG["embedding_dim"]
        self.metric_type = MILVUS_CONFIG["metric_type"]
        self._data_type = DataType

    def ensure_collection(self) -> None:
        if self.client.has_collection(self.collection):
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="entity_id", datatype=self._data_type.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(field_name="name", datatype=self._data_type.VARCHAR, max_length=512)
        schema.add_field(field_name="label", datatype=self._data_type.VARCHAR, max_length=64)
        schema.add_field(field_name="source", datatype=self._data_type.VARCHAR, max_length=64)
        schema.add_field(field_name="vector", datatype=self._data_type.FLOAT_VECTOR, dim=self.embedding_dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type=self.metric_type,
        )
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )

    def upsert(self, records: Iterable[EntityRecord], vectors: list[list[float]]) -> None:
        self.ensure_collection()
        rows = []
        for record, vector in zip(records, vectors, strict=True):
            rows.append(
                {
                    "entity_id": record.entity_id,
                    "name": record.name,
                    "label": record.label,
                    "source": record.source,
                    "vector": vector,
                }
            )
        if rows:
            self.client.upsert(collection_name=self.collection, data=rows)

    def search(self, vector: list[float], top_k: int = 5) -> list[dict]:
        self.ensure_collection()
        results = self.client.search(
            collection_name=self.collection,
            data=[vector],
            limit=top_k,
            output_fields=["entity_id", "name", "label", "source"],
        )
        return [
            {
                "score": hit["distance"],
                **hit["entity"],
            }
            for hit in results[0]
        ]
