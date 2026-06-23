from __future__ import annotations

import re


class UnsafeCypherError(ValueError):
    """Raised when generated Cypher is not safe to execute as a read query."""


_WRITE_KEYWORDS = {
    "CALL",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "LOAD",
    "MERGE",
    "REMOVE",
    "SET",
}

_READ_STARTERS = {"MATCH", "OPTIONAL", "WITH", "UNWIND"}
_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)
_PARAM_RE = re.compile(r"\$param_\d+\b")


def validate_readonly_cypher(cypher: str) -> str:
    """Validate that a generated Cypher query is a parameterized read-only query."""
    if not cypher or not cypher.strip():
        raise UnsafeCypherError("Cypher query is empty.")

    query = _COMMENT_RE.sub(" ", cypher).strip()
    statements = [statement.strip() for statement in query.split(";") if statement.strip()]
    if len(statements) != 1:
        raise UnsafeCypherError("Only one Cypher statement is allowed.")

    tokens = [token.upper() for token in _TOKEN_RE.findall(statements[0])]
    if not tokens:
        raise UnsafeCypherError("Cypher query has no valid tokens.")

    if tokens[0] not in _READ_STARTERS:
        raise UnsafeCypherError("Cypher query must start with a read clause.")

    blocked = sorted(set(tokens) & _WRITE_KEYWORDS)
    if blocked:
        raise UnsafeCypherError(f"Cypher query contains blocked keywords: {', '.join(blocked)}.")

    if "RETURN" not in tokens:
        raise UnsafeCypherError("Cypher query must return data.")

    return statements[0]


def extract_param_names(cypher: str) -> set[str]:
    return {match.group()[1:] for match in _PARAM_RE.finditer(cypher)}


def ensure_declared_params(cypher: str, params: dict[str, str]) -> None:
    referenced = extract_param_names(cypher)
    missing = sorted(referenced - set(params))
    if missing:
        raise UnsafeCypherError(f"Cypher query references undeclared params: {', '.join(missing)}.")
