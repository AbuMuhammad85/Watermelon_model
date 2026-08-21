import os
import unittest
import tensorflow as tf
import numpy as np

class TestTFLiteModels(unittest.TestCase):
    def check_tflite_inference(self, model_path, expected_output_dim):
        if not os.path.exists(model_path):
            self.skipTest(f"TFLite model not found at {model_path}. Skipped.")
            
        # Load the TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Verify input shape is [1, 224, 224, 3]
        input_shape = input_details[0]['shape']
        np.testing.assert_array_equal(input_shape, [1, 224, 224, 3])
        
        # Verify output shape is [1, expected_output_dim]
        output_shape = output_details[0]['shape']
        np.testing.assert_array_equal(output_shape, [1, expected_output_dim])
        
        # Test inference with dummy input
        dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        self.assertEqual(output_data.shape[1], expected_output_dim)

    def test_detector_tflite_float32(self):
        self.check_tflite_inference("exports/detector/watermelon_detector_float32.tflite", 1)
        
    def test_detector_tflite_float16(self):
        self.check_tflite_inference("exports/detector/watermelon_detector_float16.tflite", 1)
        
    def test_detector_tflite_int8(self):
        self.check_tflite_inference("exports/detector/watermelon_detector_int8.tflite", 1)

    def test_disease_tflite_float32(self):
        self.check_tflite_inference("exports/disease/watermelon_disease_float32.tflite", 4)
        
    def test_disease_tflite_float16(self):
        self.check_tflite_inference("exports/disease/watermelon_disease_float16.tflite", 4)
        
    def test_disease_tflite_int8(self):
        self.check_tflite_inference("exports/disease/watermelon_disease_int8.tflite", 4)

if __name__ == '__main__':
    unittest.main()
