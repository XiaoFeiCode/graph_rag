import unittest

from src.web.cypher_guard import (
    UnsafeCypherError,
    ensure_declared_params,
    extract_param_names,
    validate_readonly_cypher,
)


class CypherGuardTest(unittest.TestCase):
    def test_allows_parameterized_match_return_query(self):
        cypher = "MATCH (n:SPU {name: $param_0}) RETURN n.name AS name"

        self.assertEqual(validate_readonly_cypher(cypher), cypher)
        self.assertEqual(extract_param_names(cypher), {"param_0"})

    def test_rejects_write_keywords(self):
        with self.assertRaises(UnsafeCypherError):
            validate_readonly_cypher("MATCH (n) DELETE n RETURN n")

    def test_rejects_multiple_statements(self):
        with self.assertRaises(UnsafeCypherError):
            validate_readonly_cypher("MATCH (n) RETURN n; MATCH (m) RETURN m")

    def test_rejects_query_without_return(self):
        with self.assertRaises(UnsafeCypherError):
            validate_readonly_cypher("MATCH (n)")

    def test_rejects_undeclared_params(self):
        cypher = "MATCH (n:SPU {name: $param_0}) RETURN n"

        with self.assertRaises(UnsafeCypherError):
            ensure_declared_params(cypher, {})

    def test_allows_declared_params(self):
        cypher = "MATCH (n:SPU {name: $param_0}) RETURN n"

        ensure_declared_params(cypher, {"param_0": "iPhone 15"})


if __name__ == "__main__":
    unittest.main()
