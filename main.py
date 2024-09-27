import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

from models.lstm import StockLSTM
from data.dataset import load_data, split_data
from utils.train_model import train_model
from utils.evaluate_model import evaluate_model
from config import config

def main():
    print('CUDA Available =', torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    
    sequences, labels, scaler = load_data(config['dataset'], config['sequence_length'])
    train_sequences, test_sequences, train_labels, test_labels = split_data(sequences, labels)
    
    train_data = torch.FloatTensor(train_sequences).to(config['device'])
    test_data = torch.FloatTensor(test_sequences).to(config['device'])
    train_labels = torch.FloatTensor(train_labels).to(config['device'])
    test_labels = torch.FloatTensor(test_labels).to(config['device'])
    
    model = StockLSTM(input_size=config['input_size'], hidden_size=config['hidden_size'], num_layers=config['num_layers']).to(config['device'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    train_model(model, train_data, train_labels, config['epochs'], criterion, optimizer)
    
    actual_prices, predicted_prices = evaluate_model(model, test_data, test_labels, scaler)

    plt.plot(actual_prices, label="Actual Prices")
    plt.plot(predicted_prices, label="Predicted Prices")
    plt.title("S&P 500 Price Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()