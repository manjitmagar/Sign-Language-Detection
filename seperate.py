import tensorflow as tf

model = tf.keras.models.load_model("/Users/saniyapunmagar/Desktop/converted_keras/keras_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("/Users/saniyapunmagar/Desktop/converted_keras/converted_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved!")
