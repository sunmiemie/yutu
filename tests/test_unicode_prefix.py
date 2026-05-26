import unittest
from support import splitter


class UnicodePrefixTests(unittest.TestCase):
    def test_preserves_distinct_chinese_japanese_and_normalizes_accents(self):
        self.assertEqual(splitter.sanitize_output_prefix("中文 录音"), "中文-录音")
        self.assertEqual(splitter.sanitize_output_prefix("日本語"), "日本語")
        self.assertEqual(splitter.sanitize_output_prefix("cafe\u0301"), "café")
        self.assertNotEqual(splitter.sanitize_output_prefix("你好"), splitter.sanitize_output_prefix("谢谢"))

    def test_rejects_path_separators_and_limits_encoded_filename_size(self):
        self.assertEqual(splitter.sanitize_output_prefix("../nested\\name:*?"), "nested-name")
        self.assertEqual(splitter.sanitize_output_prefix(" /:*? "), "output")
        prefix = splitter.sanitize_output_prefix("长" * 200)
        self.assertLessEqual(len(prefix.encode("utf-8")), 120)
        self.assertTrue(prefix)
