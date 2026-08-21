import numpy as np

def run_tflite_inference(interpreter, input_tensor):
    """
    Runs inference using a loaded TFLite interpreter.
    Supports Float32, Float16, and INT8 formats.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get input tensor parameters
    input_dtype = input_details[0]['dtype']
    
    # If the interpreter input tensor is quantized INT8/UINT8, scale the input float to integers
    if input_dtype == np.int8 or input_dtype == np.uint8:
        scale, zero_point = input_details[0]['quantization']
        # f = (q - zero_point) * scale => q = (f / scale) + zero_point
        if scale > 0:
            input_tensor_quantized = (input_tensor / scale) + zero_point
        else:
            input_tensor_quantized = input_tensor
            
        if input_dtype == np.int8:
            input_tensor_quantized = np.clip(input_tensor_quantized, -128, 127).astype(np.int8)
        else:
            input_tensor_quantized = np.clip(input_tensor_quantized, 0, 255).astype(np.uint8)
            
        interpreter.set_tensor(input_details[0]['index'], input_tensor_quantized)
    else:
        # Standard float32/float16 input
        interpreter.set_tensor(input_details[0]['index'], input_tensor.astype(np.float32))
        
    interpreter.invoke()
    
    # Get outputs
    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    output_dtype = output_details[0]['dtype']
    
    # If the output tensor is quantized, scale it back to Float32 probabilities
    if output_dtype == np.int8 or output_dtype == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        if scale > 0:
            output_tensor = (output_tensor.astype(np.float32) - zero_point) * scale
            
    return output_tensor
