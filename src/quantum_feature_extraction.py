"""
Quantum feature extraction module using PennyLane and autoencoders
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from sklearn.preprocessing import StandardScaler
import pennylane as qml


def build_autoencoder(input_dim, latent_dim=5):
    """
    Build autoencoder model for latent feature extraction
    
    Args:
        input_dim (int): Input feature dimension
        latent_dim (int): Latent space dimension
        
    Returns:
        tuple: (autoencoder_model, encoder_model)
    """
    # Define autoencoder architecture
    input_layer = Input(shape=(input_dim,))
    encoder_layer = Dense(32)(input_layer)
    encoder_layer = LeakyReLU(alpha=0.1)(encoder_layer)
    encoder_output = Dense(latent_dim, activation="relu")(encoder_layer)
    
    decoder_layer = Dense(32, activation="relu")(encoder_output)
    decoder_output = Dense(input_dim, activation="linear")(decoder_layer)
    
    # Create models
    autoencoder = Model(inputs=input_layer, outputs=decoder_output)
    encoder = Model(inputs=input_layer, outputs=encoder_output)
    
    return autoencoder, encoder


def train_autoencoder(autoencoder, features, epochs=200, batch_size=64, verbose=1):
    """
    Train the autoencoder model
    
    Args:
        autoencoder (Model): Autoencoder model
        features (array): Input features
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        verbose (int): Verbosity level
        
    Returns:
        History: Training history
    """
    autoencoder.compile(optimizer="adam", loss="mse")
    history = autoencoder.fit(
        features, features, 
        epochs=epochs, 
        batch_size=batch_size, 
        verbose=verbose
    )
    return history


def extract_latent_features(encoder, features):
    """
    Extract latent features using trained encoder
    
    Args:
        encoder (Model): Trained encoder model
        features (array): Input features
        
    Returns:
        array: Latent features
    """
    latent_features = encoder.predict(features)
    return latent_features


def create_quantum_circuit(n_qubits=2):
    """
    Create a quantum device and feature extraction circuit
    
    Args:
        n_qubits (int): Number of qubits
        
    Returns:
        tuple: (device, quantum_node_function)
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def quantum_feature_circuit(inputs):
        """
        Quantum circuit for feature extraction
        
        Args:
            inputs (array): Input features (at least 3 values)
            
        Returns:
            list: Expectation values from each qubit
        """
        qml.RX(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)
        qml.RZ(inputs[2], wires=0)
        qml.CNOT(wires=[0, 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return dev, quantum_feature_circuit


def extract_quantum_features(latent_features, n_qubits=2):
    """
    Extract quantum features from latent features
    
    Args:
        latent_features (array): Latent features from autoencoder
        n_qubits (int): Number of qubits
        
    Returns:
        array: Quantum features
    """
    _, quantum_circuit = create_quantum_circuit(n_qubits)
    
    # Apply quantum circuit to each latent feature
    quantum_features = np.array([quantum_circuit(f) for f in latent_features])
    
    return quantum_features


def generate_enhanced_dataset(df, feature_columns=['ph', 'temperature', 'turbidity'], 
                              latent_dim=5, n_qubits=2, epochs=200, batch_size=64):
    """
    Generate enhanced dataset with latent and quantum features
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_columns (list): Original feature columns
        latent_dim (int): Latent space dimension
        n_qubits (int): Number of qubits for quantum circuit
        epochs (int): Training epochs for autoencoder
        batch_size (int): Batch size for autoencoder
        
    Returns:
        pd.DataFrame: Enhanced dataframe with all features
    """
    # Extract and standardize features
    features = df[feature_columns].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Build and train autoencoder
    input_dim = features_scaled.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim, latent_dim)
    
    print("Training autoencoder...")
    train_autoencoder(autoencoder, features_scaled, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Extract latent features
    print("Extracting latent features...")
    latent_features = extract_latent_features(encoder, features_scaled)
    
    # Extract quantum features
    print("Extracting quantum features...")
    quantum_features = extract_quantum_features(latent_features, n_qubits)
    
    # Create enhanced dataframe
    enhanced_df = df.copy()
    
    # Add latent features
    for i in range(latent_dim):
        enhanced_df[f'latent{i+1}'] = latent_features[:, i]
    
    # Add quantum features
    for i in range(n_qubits):
        enhanced_df[f'quantum{i+1}'] = quantum_features[:, i]
    
    return enhanced_df, scaler, encoder


def save_enhanced_data(enhanced_df, filepath, target_column='fish'):
    """
    Save enhanced dataset to CSV
    
    Args:
        enhanced_df (pd.DataFrame): Enhanced dataframe
        filepath (str): Output file path
        target_column (str): Target column name
    """
    # If target needs encoding, ensure it's included
    enhanced_df.to_csv(filepath, index=False)
    print(f"Enhanced dataset saved to {filepath}")


def load_enhanced_data(filepath):
    """
    Load enhanced dataset from CSV
    
    Args:
        filepath (str): Input file path
        
    Returns:
        pd.DataFrame: Enhanced dataframe
    """
    df = pd.read_csv(filepath)
    print(f"Enhanced dataset loaded from {filepath}")
    return df
