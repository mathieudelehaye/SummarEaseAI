# TensorFlow intent classifier model
import tensorflow as tf

def create_model(vocab_size, max_length, num_categories):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_categories, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
