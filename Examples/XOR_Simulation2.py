import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

model = Sequential([
    Dense(8, activation='relu', input_shape=(2,)),  # Increased to 8 neurons in the first layer
    Dense(4, activation='relu'),  # Added a second hidden layer with 4 neurons
    Dense(1, activation='sigmoid')
])

optimizer = Adam(learning_rate=0.01)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=5000, verbose=0)  # Increased to 5000 epochs and set verbose to 0 for less output

print("Evaluating model performance:")
loss, accuracy = model.evaluate(X, y)
print(f"Loss: {loss}, Accuracy: {accuracy}")

print("\nInitial XOR inputs:\n", X)
predictions = model.predict(X)
print("Model's XOR predictions:\n", predictions.round())
