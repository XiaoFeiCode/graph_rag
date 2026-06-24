from __future__ import annotations

import argparse
import json

import requests

from src.configuration.config import CDC_CONFIG, DEBEZIUM_CONFIG, MYSQL_CONFIG


def build_connector_config() -> dict:
    connector_name = DEBEZIUM_CONFIG["connector_name"]
    return {
        "name": connector_name,
        "config": {
            "connector.class": "io.debezium.connector.mysql.MySqlConnector",
            "tasks.max": "1",
            "database.hostname": DEBEZIUM_CONFIG["database_host"],
            "database.port": DEBEZIUM_CONFIG["database_port"],
            "database.user": MYSQL_CONFIG["user"],
            "database.password": MYSQL_CONFIG["password"],
            "database.server.id": DEBEZIUM_CONFIG["database_server_id"],
            "topic.prefix": CDC_CONFIG["topic_prefix"],
            "database.include.list": MYSQL_CONFIG["db"],
            "table.include.list": ",".join(
                [
                    f"{MYSQL_CONFIG['db']}.base_category1",
                    f"{MYSQL_CONFIG['db']}.base_category2",
                    f"{MYSQL_CONFIG['db']}.base_category3",
                    f"{MYSQL_CONFIG['db']}.base_trademark",
                    f"{MYSQL_CONFIG['db']}.spu_info",
                    f"{MYSQL_CONFIG['db']}.sku_info",
                ]
            ),
            "schema.history.internal.kafka.bootstrap.servers": "kafka:9092",
            "schema.history.internal.kafka.topic": f"schema-changes.{MYSQL_CONFIG['db']}",
            "include.schema.changes": "false",
            "snapshot.mode": "initial",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the MySQL Debezium connector.")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    payload = build_connector_config()
    if args.print_only:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    url = f"{DEBEZIUM_CONFIG['connect_url'].rstrip('/')}/connectors"
    response = requests.post(url, json=payload, timeout=30)
    if response.status_code == 409:
        config_url = f"{url}/{payload['name']}/config"
        response = requests.put(config_url, json=payload["config"], timeout=30)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
