import unittest

from pathlib import Path

from repair_char_positions_llm import patient_ids, resolve_suggestion, spans_are_compatible


class ResolveSuggestionTests(unittest.TestCase):
    def test_unique_evidence_resolves_ambiguous_short_span(self):
        note = "Admission Date: today\nSex: M\nAllergies: none\nMore M text"
        suggestion = {
            "uid4": "002e",
            "evidence": "Sex: M\nAllergies: none",
            "span": "M",
            "start": 27,
            "end": 28,
        }

        start = note.index("M\nAllergies")
        self.assertEqual(resolve_suggestion(suggestion, note), (start, start + 1, "M"))

    def test_hallucinated_quote_is_rejected(self):
        self.assertIsNone(resolve_suggestion({"span": "not present"}, "clinical note"))

    def test_wrong_existing_span_is_incompatible(self):
        self.assertFalse(spans_are_compatible("male", "No Known Allergies"))

    def test_related_existing_span_is_compatible(self):
        self.assertTrue(spans_are_compatible("chest pain", "severe chest pain"))

    def test_patient_id_is_extracted_without_date_components(self):
        path = Path("2023-03-31_annotations_162755_yes_structured.txt")
        self.assertEqual(patient_ids(path), {"162755"})


if __name__ == "__main__":
    unittest.main()
