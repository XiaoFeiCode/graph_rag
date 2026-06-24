"""Pre-defined Cypher templates for common e-commerce QA intents.

Each template defines:
- name: unique identifier
- description: what the template does (for LLM to select)
- cypher: parameterised Cypher query (uses $entity_N placeholders)
- entity_slots: list of {param_name, description, label} for entity alignment
"""

CYPHER_TEMPLATES = [
    {
        "name": "brand_products",
        "description": "查询某个品牌下有哪些商品",
        "cypher": (
            "MATCH (t:Trademark {name: $brand})<-[:Belong]-(spu:SPU) "
            "RETURN spu.name AS name, spu.description AS description"
        ),
        "entity_slots": [
            {"param_name": "brand", "description": "品牌名称", "label": "Trademark"},
        ],
    },
    {
        "name": "product_tags",
        "description": "查询某个商品有哪些卖点/标签/特性",
        "cypher": (
            "MATCH (spu:SPU {name: $product})-[:Have]->(tag:Tag) "
            "RETURN tag.name AS tag"
        ),
        "entity_slots": [
            {"param_name": "product", "description": "商品名称", "label": "SPU"},
        ],
    },
    {
        "name": "product_detail",
        "description": "查询某个商品的完整信息，包括品牌、标签、品类和SKU",
        "cypher": (
            "MATCH (spu:SPU {name: $product}) "
            "OPTIONAL MATCH (spu)-[:Belong]->(t:Trademark) "
            "OPTIONAL MATCH (spu)-[:Belong]->(c:Category3) "
            "OPTIONAL MATCH (spu)-[:Have]->(tag:Tag) "
            "OPTIONAL MATCH (spu)<-[:Belong]-(sku:SKU) "
            "RETURN spu.name AS name, spu.description AS description, "
            "t.name AS brand, c.name AS category, "
            "COLLECT(DISTINCT tag.name) AS tags, "
            "COLLECT(DISTINCT sku.name) AS skus"
        ),
        "entity_slots": [
            {"param_name": "product", "description": "商品名称", "label": "SPU"},
        ],
    },
    {
        "name": "product_skus",
        "description": "查询某个商品有哪些SKU/型号/规格",
        "cypher": (
            "MATCH (spu:SPU {name: $product})<-[:Belong]-(sku:SKU) "
            "RETURN sku.name AS sku"
        ),
        "entity_slots": [
            {"param_name": "product", "description": "商品名称", "label": "SPU"},
        ],
    },
    {
        "name": "category_products",
        "description": "查询某个品类下有哪些商品/推荐某类商品",
        "cypher": (
            "MATCH (c:Category3 {name: $category})<-[:Belong]-(spu:SPU) "
            "RETURN spu.name AS name, spu.description AS description "
            "LIMIT 10"
        ),
        "entity_slots": [
            {"param_name": "category", "description": "三级品类名称", "label": "Category3"},
        ],
    },
    {
        "name": "hot_products",
        "description": "查询热门/推荐商品",
        "cypher": (
            "MATCH (spu:SPU) "
            "RETURN spu.name AS name, spu.description AS description "
            "LIMIT 10"
        ),
        "entity_slots": [],
    },
    {
        "name": "brand_category_products",
        "description": "查询某个品牌下某个品类的商品",
        "cypher": (
            "MATCH (t:Trademark {name: $brand})<-[:Belong]-(spu:SPU)-[:Belong]->(c:Category3 {name: $category}) "
            "RETURN spu.name AS name, spu.description AS description"
        ),
        "entity_slots": [
            {"param_name": "brand", "description": "品牌名称", "label": "Trademark"},
            {"param_name": "category", "description": "三级品类名称", "label": "Category3"},
        ],
    },
    {
        "name": "search_product",
        "description": "模糊搜索商品（按名称关键词匹配）",
        "cypher": (
            "MATCH (spu:SPU) "
            "WHERE toLower(spu.name) CONTAINS toLower($keyword) "
            "RETURN spu.name AS name, spu.description AS description "
            "LIMIT 10"
        ),
        "entity_slots": [
            {"param_name": "keyword", "description": "搜索关键词", "label": None},
        ],
    },
    {
        "name": "subgraph_recall",
        "description": "全面了解某个品牌/商品/品类的所有关联信息（子图召回）",
        "cypher": "__SUBGRAPH_EXPAND__",
        "entity_slots": [
            {"param_name": "entity", "description": "起点实体名称", "label": "SPU"},
        ],
        "expandable_labels": ["Trademark", "SPU", "Category3"],
    },
]

# Allow label=NONE for keyword slots that don't need entity alignment
_SUPPORTED_LABELS = {"Trademark", "SPU", "SKU", "Category1", "Category2", "Category3"}


def get_template(name: str) -> dict | None:
    for t in CYPHER_TEMPLATES:
        if t["name"] == name:
            return t
    return None


def get_template_names() -> list[str]:
    return [t["name"] for t in CYPHER_TEMPLATES]


def get_template_descriptions() -> str:
    lines = []
    for t in CYPHER_TEMPLATES:
        lines.append(f"- {t['name']}: {t['description']}")
    return "\n".join(lines)
