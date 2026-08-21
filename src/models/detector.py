import tensorflow as tf
from tensorflow.keras import layers, models

def build_detector_model(input_shape=(224, 224, 3), learning_rate=1e-3):
    """
    Builds the binary watermelon vs not-watermelon detector model using the Functional API.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Load MobileNetV2 base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    # Connect layers symbolically
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid', name="binary_output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="watermelon_detector")
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    return model, base_model
