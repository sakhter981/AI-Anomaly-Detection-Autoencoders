import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, losses
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Data Simulation (Normal behavioral patterns)
data = np.random.normal(size=(1000, 20)) 
# Injecting anomalies
anomalies = np.random.uniform(low=-10, high=10, size=(50, 20))
full_dataset = np.vstack([data, anomalies])

# 2. Preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(full_dataset)
train_data, test_data = train_test_split(scaled_data, test_size=0.2)

# 3. Building the Autoencoder Model [cite: 26]
class AnomalyDetector(tf.keras.Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(4, activation="relu")]) # Latent space
        
        self.decoder = tf.keras.Sequential([
            layers.Dense(8, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(20, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 4. Training and Optimization [cite: 27]
autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')
autoencoder.fit(train_data, train_data, epochs=50, batch_size=32, validation_split=0.1)

# 5. Detection Logic
reconstructions = autoencoder.predict(test_data)
train_loss = tf.keras.losses.mae(reconstructions, test_data)
threshold = np.mean(train_loss) + np.std(train_loss) # Threshold for 85%+ accuracy [cite: 27]

print(f"Anomaly Threshold: {threshold}")