Stock Price Prediction Using LSTM in PyTorch.  

Overview   
This repository provides an end-to-end pipeline for training and evaluating stock price prediction models using deep learning techniques. The project includes:

- Data Loading and Preprocessing: Loading historical stock price data, creating train/test splits, and applying normalization techniques for better model performance.
- LSTM Model Implementation: Design and implementation of a Long Short-Term Memory (LSTM) network for time-series forecasting.
- Training Pipeline: The training loop supports dynamic learning rate adjustment and loss monitoring for efficient training.
- Evaluation and Visualization: Tools for evaluating the model on test data and visualizing the predicted vs. actual stock prices.
- Device Management: Automatic GPU/CPU selection based on availability, with PyTorch support.
  
Features   
- Custom LSTM Model: The project includes a custom implementation of an LSTM model specifically designed for stock price forecasting.
- Data Preprocessing: Sliding window approach for generating time-series sequences and normalization of stock prices.
- Loss Monitoring: Track training loss for each epoch and log results.
- Dynamic Device Selection: Automatically selects GPU if available, otherwise defaults to CPU.
- Evaluation and Visualization: Tools for comparing model predictions with actual prices using line plots.

Requirements    
To run the project, the following libraries are required:
- PyTorch
- Scikit-learn
- Matplotlib
- Pandas
- Numpy
