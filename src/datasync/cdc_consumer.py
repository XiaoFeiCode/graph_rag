from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.configuration.config import CDC_CONFIG, MYSQL_CONFIG, NEO4J_CONFIG, ROOT_DIR

logger = logging.getLogger(__name__)

ALLOWED_LABELS = {
    "Category1",
    "Category2",
    "Category3",
    "Trademark",
    "SPU",
    "SKU",
    "BaseAttrName",
    "BaseAttrValue",
    "SaleAttrName",
    "SaleAttrValue",
    "Tag",
}
ALLOWED_RELATIONSHIPS = {"Belong", "Have"}


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _require_safe_graph_name(name: str, allowed: set[str]) -> str:
    if name not in allowed:
        raise ValueError(f"Unsupported graph name: {name}")
    return name


def parse_debezium_message(message: bytes | str) -> dict[str, Any] | None:
    payload = json.loads(message.decode("utf-8") if isinstance(message, bytes) else message)
    event = payload.get("payload", payload)
    op = event.get("op")
    source = event.get("source") or {}
    table = source.get("table")
    row = event.get("after") if op != "d" else event.get("before")
    if not table or not row or op not in {"c", "u", "d", "r"}:
        return None
    return {"table": table, "op": op, "row": row}


class CdcNeo4jSync:
    def __init__(self, mapping: dict[str, Any]):
        from neo4j import GraphDatabase

        self.mapping = mapping
        self.driver = GraphDatabase.driver(NEO4J_CONFIG["uri"], auth=NEO4J_CONFIG["auth"])

    def close(self) -> None:
        self.driver.close()

    def apply(self, event: dict[str, Any]) -> None:
        table_mapping = self.mapping.get(event["table"])
        if not table_mapping:
            logger.debug("Skip unmapped CDC table: %s", event["table"])
            return

        row = event["row"]
        if event["op"] == "d":
            self._delete_node(table_mapping["node"], row)
            return

        self._upsert_node(table_mapping["node"], row)
        for relationship in table_mapping.get("relationships", []):
            self._upsert_relationship(relationship, row)

    def _upsert_node(self, node_mapping: dict[str, str], row: dict[str, Any]) -> None:
        label = _require_safe_graph_name(node_mapping["label"], ALLOWED_LABELS)
        node_id = str(row[node_mapping["id"]])
        props = {
            "id": node_id,
            "name": row.get(node_mapping["name"]),
        }
        description_field = node_mapping.get("description")
        if description_field and row.get(description_field) is not None:
            props["description"] = row[description_field]

        cypher = f"""
        MERGE (n:{label} {{id: $id}})
        SET n += $props
        """
        self.driver.execute_query(cypher, id=node_id, props=props)

    def _delete_node(self, node_mapping: dict[str, str], row: dict[str, Any]) -> None:
        label = _require_safe_graph_name(node_mapping["label"], ALLOWED_LABELS)
        node_id = str(row[node_mapping["id"]])
        cypher = f"MATCH (n:{label} {{id: $id}}) DETACH DELETE n"
        self.driver.execute_query(cypher, id=node_id)

    def _upsert_relationship(self, mapping: dict[str, str], row: dict[str, Any]) -> None:
        rel_type = _require_safe_graph_name(mapping["type"], ALLOWED_RELATIONSHIPS)
        start_label = _require_safe_graph_name(mapping["start_label"], ALLOWED_LABELS)
        end_label = _require_safe_graph_name(mapping["end_label"], ALLOWED_LABELS)
        start_id = row.get(mapping["start_id"])
        end_id = row.get(mapping["end_id"])
        if start_id is None or end_id is None:
            return

        cypher = f"""
        MATCH (start:{start_label} {{id: $start_id}})
        MATCH (end:{end_label} {{id: $end_id}})
        MERGE (start)-[:{rel_type}]->(end)
        """
        self.driver.execute_query(cypher, start_id=str(start_id), end_id=str(end_id))


def _topics(mapping: dict[str, Any]) -> list[str]:
    prefix = CDC_CONFIG["topic_prefix"]
    database = MYSQL_CONFIG["db"]
    return [f"{prefix}.{database}.{table}" for table in mapping]


def consume(mapping_path: Path) -> None:
    from confluent_kafka import Consumer

    mapping = _load_mapping(mapping_path)
    consumer = Consumer(
        {
            "bootstrap.servers": CDC_CONFIG["bootstrap_servers"],
            "group.id": CDC_CONFIG["group_id"],
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
    )
    sync = CdcNeo4jSync(mapping)
    consumer.subscribe(_topics(mapping))
    logger.info("Subscribed CDC topics: %s", _topics(mapping))

    try:
        while True:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                logger.warning("Kafka consumer error: %s", message.error())
                continue
            event = parse_debezium_message(message.value())
            if event:
                sync.apply(event)
    finally:
        sync.close()
        consumer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consume Debezium MySQL CDC events into Neo4j.")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=ROOT_DIR / "src" / "configuration" / "cdc_table_mapping.json",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    consume(args.mapping)


if __name__ == "__main__":
    main()
