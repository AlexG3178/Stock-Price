import torch

def evaluate_model(model, test_data, test_labels, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(test_data)
        predictions = predictions.cpu().numpy()

        # Reverse the scaling on predictions
        predictions = scaler.inverse_transform(predictions)

        # Reverse the scaling on actual labels
        actual_prices = scaler.inverse_transform(test_labels.cpu().numpy())

        return actual_prices, predictions