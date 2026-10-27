import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MODEL_NAME", "BAAI/bge-m3")

import utils


class BGETokenizerTests(unittest.TestCase):
    def tearDown(self):
        utils.get_tokenizer.cache_clear()

    def test_bge_m3_is_the_default_model(self):
        self.assertEqual(utils.MODEL_NAME, "BAAI/bge-m3")

    def test_token_length_uses_model_tokenizer_without_special_tokens(self):
        tokenizer = Mock()
        tokenizer.encode.return_value = [10, 20, 30]

        with patch.object(utils.AutoTokenizer, "from_pretrained", return_value=tokenizer):
            utils.get_tokenizer.cache_clear()
            self.assertEqual(utils.token_length("texto bilingue"), 3)

        tokenizer.encode.assert_called_once_with(
            "texto bilingue",
            add_special_tokens=False,
            truncation=False,
        )

    def test_language_detection_uses_document_text(self):
        spanish = (
            "Informe de evaluación. La investigación se realizó con doce "
            "participantes y contiene resultados, conclusiones y limitaciones."
        )
        english = (
            "Evaluation report. The research involved twelve participants and "
            "contains results, conclusions, and limitations."
        )

        self.assertEqual(utils.detect_language(spanish), "es")
        self.assertEqual(utils.detect_language(english), "en")


if __name__ == "__main__":
    unittest.main()
