import unittest

from postprocess_char_positions import Span, choose_span


class ChooseSpanTests(unittest.TestCase):
    def test_uid_tag_wins_over_an_earlier_matching_mention(self):
        text = "OA appears early; later tagged OA appears here."
        tagged = {"04e2": [Span("04e2", 31, 33, "OA")]}

        self.assertEqual(
            choose_span("04e2", "OA", tagged, text, set()),
            (31, 33),
        )

    def test_text_search_is_used_when_uid_has_no_tag(self):
        text = "A derived timeline mention"

        self.assertEqual(
            choose_span("ffff", "timeline mention", {}, text, set()),
            (10, 26),
        )


if __name__ == "__main__":
    unittest.main()
