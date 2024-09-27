import torch

config = {
    'dataset': 'datasets/train_dataset.csv',
    # 'dataset': 'datasets/train_dataset_cut.csv',
    # 'initial_dataset': 'misikoff/SPX',
    'epochs': 100,
    'sequence_length': 50,
    'train_size': 0.9,
    'learning_rate': 0.001,
    'input_size': 1,      
    'hidden_size': 50,    
    'num_layers': 2,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
}