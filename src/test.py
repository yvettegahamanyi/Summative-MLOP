import tensorflow as tf

preprocess_input = tf.keras.applications.efficientnet.preprocess_input
F1Score = tf.keras.metrics.F1Score(name='f1_score')
# Load the saved model
best_model = tf.keras.models.load_model("models/efficientnetb0_waste_classifier.h5", custom_objects={'F1Score': F1Score, 'preprocess_input': preprocess_input})

print("Model loaded successfully!")