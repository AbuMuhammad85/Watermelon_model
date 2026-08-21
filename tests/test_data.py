import os
import unittest
from src.data.data_loader import load_dataset_paths_labels

class TestDataPipeline(unittest.TestCase):
    def test_watermelon_raw_exists(self):
        """Verify the raw watermelon dataset directory exists and contains the expected classes."""
        raw_dir = "Watermelon"
        self.assertTrue(os.path.exists(raw_dir))
        classes = sorted(os.listdir(raw_dir))
        expected_classes = [
            "watermelon___anthracnose",
            "watermelon___downy_mildew",
            "watermelon___healthy",
            "watermelon___mosaic_virus"
        ]
        self.assertEqual(classes, expected_classes)

    def test_processed_splits_structure(self):
        """Verify the processed splits directories are created after running splits."""
        processed_dir = "data/processed"
        # If splits have been prepared, verify structure
        if os.path.exists(processed_dir):
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(processed_dir, split)
                self.assertTrue(os.path.exists(split_path))
                classes = sorted(os.listdir(split_path))
                self.assertEqual(len(classes), 4)

    def test_detector_dataset_structure(self):
        """Verify detector dataset splits have watermelon and not_watermelon folders."""
        detector_dir = "data/detector"
        if os.path.exists(detector_dir):
            for split in ['train', 'val', 'test']:
                split_path = os.path.join(detector_dir, split)
                self.assertTrue(os.path.exists(split_path))
                classes = sorted(os.listdir(split_path))
                self.assertIn("watermelon", classes)
                self.assertIn("not_watermelon", classes)

if __name__ == '__main__':
    unittest.main()
