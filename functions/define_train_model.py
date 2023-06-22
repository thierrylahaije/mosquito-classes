import tensorflow as tf
from tensorflow import keras

def define_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(350, 350, 3)),
        keras.applications.EfficientNetB0(input_shape=(350, 350, 3), include_top=False),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(4)
    ])

    return model

def train_model(model, train_images, train_bboxes, batch_size, epochs):
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_images, train_bboxes, batch_size=batch_size, epochs=epochs)