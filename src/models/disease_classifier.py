import tensorflow as tf
from tensorflow.keras import layers, models

def build_disease_model(input_shape=(224, 224, 3), num_classes=4, learning_rate=1e-3):
    """
    Builds the 4-class disease classifier model using the Functional API.
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
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="disease_output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="disease_classifier")
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model
