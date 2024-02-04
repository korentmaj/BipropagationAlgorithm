import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Nastavi XOR input output
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Sequential model
model = Sequential([
    # Input layer z 2 nevronoma (Za 2 input bita) + hidden layer z 4 nevroni
    Dense(4, input_dim=2, activation='relu'),
    # Output layer z 1 nevronom (Za 1 izhodni bit)
    Dense(1, activation='sigmoid')
])

# Compile
optimizer = Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=1000, verbose=1)

print("Model evaluation:")
model.evaluate(X, y)

print("Testing XOR operation:")
test_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
predictions = model.predict(test_data)
print("Input:\n", test_data)
print("Predicted XOR output:\n", predictions)
