from __future__ import annotations

import json
from pathlib import Path

from scripts.create_indexes import main as create_indexes
from scripts.neo4j_client import neo4j_driver

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "demo" / "catalog.json"


def _load_catalog() -> dict:
    with DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_nodes(session, label: str, rows: list[dict]) -> None:
    if not rows:
        return

    cypher = f"""
    UNWIND $rows AS row
    MERGE (n:{label} {{id: row.id}})
    SET n.name = row.name,
        n.description = row.description,
        n.demo = true
    """
    normalized = [
        {"id": str(row["id"]), "name": row["name"], "description": row.get("description")}
        for row in rows
    ]
    session.run(cypher, rows=normalized).consume()


def _write_tags(session, products: list[dict]) -> None:
    rows = []
    relationships = []
    for product in products:
        for index, tag in enumerate(product.get("tags", [])):
            tag_id = f"{product['id']}-tag-{index}"
            rows.append({"id": tag_id, "name": tag})
            relationships.append({"start_id": str(product["id"]), "end_id": tag_id})

    _write_nodes(session, "Tag", rows)
    session.run(
        """
        UNWIND $rows AS row
        MATCH (spu:SPU {id: row.start_id}), (tag:Tag {id: row.end_id})
        MERGE (spu)-[:Have]->(tag)
        """,
        rows=relationships,
    ).consume()


def _write_relationships(session, products: list[dict]) -> None:
    session.run(
        """
        UNWIND $rows AS row
        MATCH (c2:Category2 {id: row.start_id}), (c1:Category1 {id: row.end_id})
        MERGE (c2)-[:Belong]->(c1)
        """,
        rows=[
            {"start_id": str(row["id"]), "end_id": str(row["category1_id"])}
            for row in _load_catalog()["category2"]
        ],
    ).consume()

    session.run(
        """
        UNWIND $rows AS row
        MATCH (c3:Category3 {id: row.start_id}), (c2:Category2 {id: row.end_id})
        MERGE (c3)-[:Belong]->(c2)
        """,
        rows=[
            {"start_id": str(row["id"]), "end_id": str(row["category2_id"])}
            for row in _load_catalog()["category3"]
        ],
    ).consume()

    session.run(
        """
        UNWIND $rows AS row
        MATCH (spu:SPU {id: row.spu_id})
        MATCH (sku:SKU {id: row.sku_id})
        MATCH (category:Category3 {id: row.category3_id})
        MATCH (trademark:Trademark {id: row.trademark_id})
        MERGE (sku)-[:Belong]->(spu)
        MERGE (spu)-[:Belong]->(category)
        MERGE (spu)-[:Belong]->(trademark)
        """,
        rows=[
            {
                "spu_id": str(row["id"]),
                "sku_id": str(sku["id"]),
                "category3_id": str(row["category3_id"]),
                "trademark_id": str(row["trademark_id"]),
            }
            for row in products
            for sku in row.get("skus", [])
        ],
    ).consume()


def main() -> None:
    catalog = _load_catalog()
    create_indexes()

    with neo4j_driver() as driver:
        with driver.session() as session:
            _write_nodes(session, "Category1", catalog["category1"])
            _write_nodes(session, "Category2", catalog["category2"])
            _write_nodes(session, "Category3", catalog["category3"])
            _write_nodes(session, "Trademark", catalog["trademarks"])
            _write_nodes(session, "SPU", catalog["products"])
            _write_nodes(
                session,
                "SKU",
                [sku for product in catalog["products"] for sku in product.get("skus", [])],
            )
            _write_relationships(session, catalog["products"])
            _write_tags(session, catalog["products"])

    print(
        "Seeded demo graph: "
        f"{len(catalog['products'])} SPUs, "
        f"{sum(len(product.get('skus', [])) for product in catalog['products'])} SKUs."
    )


if __name__ == "__main__":
    main()
