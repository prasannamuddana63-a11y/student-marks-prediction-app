import tensorflow as tf
import numpy as np

# Better dataset
x = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)

y = np.array([20,30,40,50,60,70,80,90,95,100], dtype=float)

# Better model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(1)
])

# Compile
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# Train
model.fit(x, y, epochs=1000)

# Save
model.save("model.h5")

print("Better model trained!")
print("Prediction for 1 hour:", model.predict(np.array([1.0])))
print("Prediction for 5 hours:", model.predict(np.array([5.0])))