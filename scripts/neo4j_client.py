from __future__ import annotations

from contextlib import contextmanager

from neo4j import GraphDatabase

from src.configuration.config import NEO4J_CONFIG


@contextmanager
def neo4j_driver():
    driver = GraphDatabase.driver(
        NEO4J_CONFIG["uri"],
        auth=NEO4J_CONFIG["auth"],
    )
    try:
        yield driver
    finally:
        driver.close()
