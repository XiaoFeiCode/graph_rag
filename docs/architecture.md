# Architecture Notes

## Modules

- `src.datasync`: reads product data from MySQL and writes nodes/relationships to Neo4j.
- `src.ner`: preprocesses Label Studio annotations, trains a BERT token classifier, evaluates it, and extracts product tags.
- `src.web.utils`: creates Neo4j full-text and vector indexes for entity alignment.
- `src.web.service`: implements the GraphRAG chain.
- `src.web.app`: exposes the FastAPI app and static chat UI.

## Graph Schema

Main nodes:

- `Category1`, `Category2`, `Category3`
- `Trademark`
- `SPU`, `SKU`
- `BaseAttrName`, `BaseAttrValue`
- `SaleAttrName`, `SaleAttrValue`
- `Tag`

Main relationships:

- `Belong`: category hierarchy, SKU -> SPU, SPU -> category, SPU -> trademark
- `Have`: category/product/attribute/tag ownership

## Request Flow

```text
POST /api/chat
  Question.message
    -> ChatService.chat
      -> _get_question_cypher
      -> _align_entities
      -> _execute_cypher
      -> _generate_answer
  Answer.message
```

## Demo Flow

```text
docker compose up -d
  -> Neo4j service
  -> scripts.create_indexes
  -> scripts.seed_graph
  -> scripts.smoke_query
```

## Training Flow

```text
data/ner/raw/data.json
  -> src.ner.preprocess
  -> data/ner/processed
  -> src.ner.train
  -> checkpoints/ner/best_model
  -> src.ner.predict
```
