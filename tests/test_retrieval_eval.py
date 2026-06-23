import unittest

from src.evaluation.retrieval_eval import RetrievalHit, _score_case


class RetrievalEvalTest(unittest.TestCase):
    def test_scores_recall_precision_and_mrr(self):
        hits = [
            RetrievalHit(name="Apple", label="Trademark", score=1.0),
            RetrievalHit(name="iPhone 15", label="SPU", score=0.8),
            RetrievalHit(name="智能手机", label="Category3", score=0.4),
        ]

        metrics = _score_case(["Apple", "iPhone 15"], hits)

        self.assertEqual(metrics["matched_count"], 2)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["mrr"], 1.0)

    def test_scores_missing_expected_entities(self):
        hits = [RetrievalHit(name="厨房小电", label="Category2", score=0.9)]

        metrics = _score_case(["美的 5L 空气炸锅"], hits)

        self.assertEqual(metrics["matched_count"], 0)
        self.assertAlmostEqual(metrics["recall"], 0.0)
        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["mrr"], 0.0)


if __name__ == "__main__":
    unittest.main()
