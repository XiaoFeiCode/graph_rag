from __future__ import annotations

from scripts.neo4j_client import neo4j_driver


CONSTRAINTS = [
    "CREATE CONSTRAINT category1_id IF NOT EXISTS FOR (n:Category1) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT category2_id IF NOT EXISTS FOR (n:Category2) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT category3_id IF NOT EXISTS FOR (n:Category3) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT trademark_id IF NOT EXISTS FOR (n:Trademark) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT spu_id IF NOT EXISTS FOR (n:SPU) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT sku_id IF NOT EXISTS FOR (n:SKU) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT tag_id IF NOT EXISTS FOR (n:Tag) REQUIRE n.id IS UNIQUE",
]

FULLTEXT_INDEXES = [
    "CREATE FULLTEXT INDEX trademark_fulltext_index IF NOT EXISTS FOR (n:Trademark) ON EACH [n.name]",
    "CREATE FULLTEXT INDEX spu_fulltext_index IF NOT EXISTS FOR (n:SPU) ON EACH [n.name]",
    "CREATE FULLTEXT INDEX sku_fulltext_index IF NOT EXISTS FOR (n:SKU) ON EACH [n.name]",
    "CREATE FULLTEXT INDEX category1_fulltext_index IF NOT EXISTS FOR (n:Category1) ON EACH [n.name]",
    "CREATE FULLTEXT INDEX category2_fulltext_index IF NOT EXISTS FOR (n:Category2) ON EACH [n.name]",
    "CREATE FULLTEXT INDEX category3_fulltext_index IF NOT EXISTS FOR (n:Category3) ON EACH [n.name]",
    "CREATE FULLTEXT INDEX tag_fulltext_index IF NOT EXISTS FOR (n:Tag) ON EACH [n.name]",
]


def main() -> None:
    with neo4j_driver() as driver:
        with driver.session() as session:
            for cypher in CONSTRAINTS:
                session.run(cypher).consume()
            for cypher in FULLTEXT_INDEXES:
                session.run(cypher).consume()

    print(f"Created {len(CONSTRAINTS)} constraints and {len(FULLTEXT_INDEXES)} fulltext indexes.")


if __name__ == "__main__":
    main()
