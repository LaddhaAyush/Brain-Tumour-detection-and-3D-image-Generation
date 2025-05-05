from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from tensorflow.keras.layers import Layer

# Flask app initialization
app = Flask(__name__)


# Custom Layers
class Avg2MaxPooling(Layer):
    def __init__(self, **kwargs):
        super(Avg2MaxPooling, self).__init__(**kwargs)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        return avg_pool + max_pool

    def get_config(self):
        config = super().get_config()
        return config


class DepthwiseSeparableConv(Layer):
    def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1), **kwargs):
        super(DepthwiseSeparableConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size, strides=strides, padding='same')
        self.pointwise_conv = tf.keras.layers.Conv2D(
            filters, kernel_size=1, padding='same')

    def call(self, inputs):
        x = self.depthwise_conv(inputs)
        return self.pointwise_conv(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Try to load the model with custom objects
# Note: We'll add error handling to diagnose issues
try:
    print("Attempting to load model...")

    # Set memory growth to avoid GPU memory issues
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    # Alternative approach: Use a different model loading technique
    model = load_model(
        'models/fibonacci.h5',
        custom_objects={
            'Avg2MaxPooling': Avg2MaxPooling,
            'DepthwiseSeparableConv': DepthwiseSeparableConv
        },
        compile=False  # Try loading without compiling first
    )

    # If loading succeeds, compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Attempting an alternative approach...")

    # Alternative: Load model without custom objects first, then add them
    try:
        # Load the model structure only
        model = tf.keras.models.load_model('models/fibonacci.h5', compile=False)

        # Register the custom objects
        tf.keras.utils.get_custom_objects().update({
            'Avg2MaxPooling': Avg2MaxPooling,
            'DepthwiseSeparableConv': DepthwiseSeparableConv
        })

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Model loaded with alternative approach!")
    except Exception as e:
        print(f"Alternative approach failed: {e}")
        print("Please check if the model file exists and is in the correct format.")
        # Create a simple fallback model for testing the rest of the app
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(16, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Created fallback model for testing")

# Labels - based on your binary classifier
class_labels = ['Healthy', 'Tumor']

# Upload directory
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Prediction function
def predict_tumor(image_path):
    IMAGE_SIZE = 224  # your model input size
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    try:
        predictions = model.predict(img_array)
        predicted_class_index = np.round(predictions[0][0]).astype(int)
        confidence_score = predictions[0][0]

        # Handle index out of range
        if predicted_class_index < 0 or predicted_class_index >= len(class_labels):
            predicted_class_index = 0  # Default to first class if out of range

        return f"{class_labels[predicted_class_index]}", confidence_score
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error during prediction", 0.0


# Index Route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('fibo.html', error="No file part")

        file = request.files['file']
        if file.filename == '':
            return render_template('fibo.html', error="No selected file")

        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            result, confidence = predict_tumor(filepath)
            return render_template('fibo.html',
                                   result=result,
                                   confidence=f"{confidence * 100:.2f}%",
                                   file_path=f'/uploads/{file.filename}')
        except Exception as e:
            return render_template('fibo.html', error=f"Error processing file: {str(e)}")

    return render_template('fibo.html', result=None)


# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)