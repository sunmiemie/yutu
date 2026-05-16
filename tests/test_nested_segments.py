import unittest
from support import splitter


class NestedSegmentTests(unittest.TestCase):
    def test_short_merge_retains_nested_speech_end_and_input(self):
        source = [splitter.Segment(0, 10), splitter.Segment(1, 2)]
        self.assertEqual(splitter.merge_short_segments(source, 20), [splitter.Segment(0, 10)])
        self.assertEqual(source, [splitter.Segment(0, 10), splitter.Segment(1, 2)])

    def test_postprocessing_keeps_overlapping_coverage(self):
        config = splitter.VadSplitConfig(min_segment_duration_sec=20, extra_pad_sec=0)
        self.assertEqual(splitter.post_process_segments(
            [splitter.Segment(0, 10), splitter.Segment(1, 2), splitter.Segment(9, 12)], config, 15
        ), [splitter.Segment(0, 12)])
