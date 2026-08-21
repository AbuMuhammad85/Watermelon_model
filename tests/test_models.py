import unittest
import tensorflow as tf
from src.models.detector import build_detector_model
from src.models.disease_classifier import build_disease_model

class TestModels(unittest.TestCase):
    def test_detector_architecture(self):
        """Verify the binary detector build parameters."""
        model, base_model = build_detector_model()
        
        # Verify input shape
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        # Verify output shape (sigmoid logit)
        self.assertEqual(model.output_shape, (None, 1))
        # Base model frozen
        self.assertFalse(base_model.trainable)
        
    def test_disease_classifier_architecture(self):
        """Verify the disease classifier build parameters."""
        num_classes = 4
        model, base_model = build_disease_model(num_classes=num_classes)
        
        # Verify input shape
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        # Verify output shape (softmax layer)
        self.assertEqual(model.output_shape, (None, num_classes))
        # Base model frozen
        self.assertFalse(base_model.trainable)

if __name__ == '__main__':
    unittest.main()
