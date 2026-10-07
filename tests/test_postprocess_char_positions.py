import unittest

from postprocess_char_positions import (
    Span,
    align_spans_to_source,
    choose_span,
    extract_spans,
)


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


class NumberedTagMappingTests(unittest.TestCase):
    def test_stale_number_mapping_is_recovered_by_declared_mention(self):
        text = (
            '<tag_1 mention="No Known Allergies"> No Known Allergies </tag_1>'
            '<tag_2 mention="No Adverse Drug Reactions"> Adverse Drug Reactions </tag_2>'
        )
        mentions = {
            "002e": "male",
            "0065": "No Known Allergies / Adverse Drug Reactions",
        }
        stale_tag_map = {"1": "002e", "2": "0065"}

        _, spans = extract_spans(text, mentions, stale_tag_map)

        self.assertEqual(spans[0].uid4, "0065")
        self.assertEqual(spans[1].uid4, "tag_2")
        self.assertNotIn("002e", {span.uid4 for span in spans})


class SourceAlignmentTests(unittest.TestCase):
    def test_document_alignment_disambiguates_short_surface_forms(self):
        clean = "Admission Date: x Sex: M Allergies: No Known AllergiesMention table:"
        source = "Admission Date: x\nSex: M\nAllergies:\nNo Known Allergies"
        start = clean.index("M Allergies")
        spans = [Span("0410", start, start + 1, "M")]

        aligned, missing = align_spans_to_source(spans, clean, source)

        expected = source.index("M\nAllergies")
        self.assertEqual(missing, 0)
        self.assertEqual((aligned[0].start, aligned[0].end), (expected, expected + 1))
        self.assertEqual(source[aligned[0].start:aligned[0].end], "M")

    def test_alignment_preserves_internal_source_whitespace(self):
        clean = "No Known Allergies"
        source = "No Known\nAllergies"
        spans = [Span("05c4", 0, len(clean), clean)]

        aligned, missing = align_spans_to_source(spans, clean, source)

        self.assertEqual(missing, 0)
        self.assertEqual((aligned[0].start, aligned[0].end), (0, len(source)))


if __name__ == "__main__":
    unittest.main()
