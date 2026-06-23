from __future__ import annotations

import json

from scripts.neo4j_client import neo4j_driver


QUERY = """
MATCH (sku:SKU)-[:Belong]->(spu:SPU)-[:Belong]->(brand:Trademark)
OPTIONAL MATCH (spu)-[:Have]->(tag:Tag)
WITH spu, brand, collect(DISTINCT tag.name) AS tags, collect(DISTINCT sku.name) AS skus
RETURN spu.name AS product, brand.name AS brand, tags, skus
ORDER BY product
"""


def main() -> None:
    with neo4j_driver() as driver:
        rows = driver.execute_query(QUERY).records

    payload = [dict(row) for row in rows]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
