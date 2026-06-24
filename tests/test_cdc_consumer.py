import json
import unittest

from src.datasync.cdc_consumer import parse_debezium_message


class CdcConsumerTest(unittest.TestCase):
    def test_parse_create_event(self):
        message = {
            "payload": {
                "op": "c",
                "source": {"table": "spu_info"},
                "after": {"id": 1, "spu_name": "iPhone 15"},
            }
        }

        event = parse_debezium_message(json.dumps(message).encode("utf-8"))

        self.assertEqual(event["table"], "spu_info")
        self.assertEqual(event["op"], "c")
        self.assertEqual(event["row"]["spu_name"], "iPhone 15")

    def test_parse_delete_event_uses_before_row(self):
        message = {
            "payload": {
                "op": "d",
                "source": {"table": "sku_info"},
                "before": {"id": 10, "sku_name": "iPhone 15 128GB"},
            }
        }

        event = parse_debezium_message(json.dumps(message))

        self.assertEqual(event["table"], "sku_info")
        self.assertEqual(event["op"], "d")
        self.assertEqual(event["row"]["id"], 10)

    def test_skip_tombstone_or_invalid_event(self):
        self.assertIsNone(parse_debezium_message("{}"))


if __name__ == "__main__":
    unittest.main()
