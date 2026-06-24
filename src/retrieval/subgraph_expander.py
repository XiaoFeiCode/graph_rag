"""Subgraph recall via multi-hop graph path expansion.

Given an aligned entity (brand, SPU, category), expand along the knowledge
graph relationships to collect a rich context window for answer generation.

Example paths:
    Trademark → SPU → [SKU, Tag, Category3 → Category2 → Category1]
    SPU → [Trademark, SKU, Tag, Category3 → Category2 → Category1]
    Category3 → SPU → [Trademark, SKU, Tag]
"""

from __future__ import annotations

EXPANSION_TEMPLATES = {
    "Trademark": """
        MATCH (t:Trademark {{name: $entity}})<-[:Belong]-(spu:SPU)
        OPTIONAL MATCH (spu)-[:Belong]->(c3:Category3)
        OPTIONAL MATCH (c3)-[:Belong]->(c2:Category2)
        OPTIONAL MATCH (c2)-[:Belong]->(c1:Category1)
        OPTIONAL MATCH (spu)-[:Have]->(tag:Tag)
        OPTIONAL MATCH (spu)<-[:Belong]-(sku:SKU)
        RETURN t.name AS brand,
               COLLECT(DISTINCT {{name: spu.name, desc: spu.description}}) AS spus,
               COLLECT(DISTINCT c3.name) AS category3,
               COLLECT(DISTINCT c2.name) AS category2,
               COLLECT(DISTINCT c1.name) AS category1,
               COLLECT(DISTINCT tag.name) AS tags,
               COLLECT(DISTINCT sku.name) AS skus
    """,
    "SPU": """
        MATCH (spu:SPU {{name: $entity}})
        OPTIONAL MATCH (spu)-[:Belong]->(t:Trademark)
        OPTIONAL MATCH (spu)-[:Belong]->(c3:Category3)
        OPTIONAL MATCH (c3)-[:Belong]->(c2:Category2)
        OPTIONAL MATCH (c2)-[:Belong]->(c1:Category1)
        OPTIONAL MATCH (spu)-[:Have]->(tag:Tag)
        OPTIONAL MATCH (spu)<-[:Belong]-(sku:SKU)
        RETURN spu.name AS product, spu.description AS description,
               t.name AS brand,
               COLLECT(DISTINCT c3.name) AS category3,
               COLLECT(DISTINCT c2.name) AS category2,
               COLLECT(DISTINCT c1.name) AS category1,
               COLLECT(DISTINCT tag.name) AS tags,
               COLLECT(DISTINCT sku.name) AS skus
    """,
    "Category3": """
        MATCH (c3:Category3 {{name: $entity}})<-[:Belong]-(spu:SPU)
        OPTIONAL MATCH (spu)-[:Belong]->(t:Trademark)
        OPTIONAL MATCH (c3)-[:Belong]->(c2:Category2)
        OPTIONAL MATCH (c2)-[:Belong]->(c1:Category1)
        OPTIONAL MATCH (spu)-[:Have]->(tag:Tag)
        OPTIONAL MATCH (spu)<-[:Belong]-(sku:SKU)
        RETURN c3.name AS category,
               COLLECT(DISTINCT {{name: spu.name, brand: t.name}}) AS spus,
               COLLECT(DISTINCT c2.name) AS category2,
               COLLECT(DISTINCT c1.name) AS category1,
               COLLECT(DISTINCT tag.name) AS tags,
               COLLECT(DISTINCT sku.name) AS skus
    """,
}


def get_expansion_cypher(entity_type: str) -> str | None:
    return EXPANSION_TEMPLATES.get(entity_type)
