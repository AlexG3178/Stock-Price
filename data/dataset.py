import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import config

def load_data(filepath, sequence_length=50):
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    
    data = df[['Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    sequences = []
    labels = []
    
    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:i+sequence_length])
        labels.append(scaled_data[i+sequence_length])
    
    print("Dataset size:", len(scaled_data))
    return np.array(sequences), np.array(labels), scaler

def split_data(sequences, labels, train_ratio=config['train_size']):
    train_size = int(len(sequences) * train_ratio)
    
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]
    
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]
    
    print("Train size:", len(train_sequences))
    print("Test size:", len(test_sequences))
    print("Sequence size:", len(sequences[0]))
    
    return train_sequences, test_sequences, train_labels, test_labels