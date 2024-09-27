def train_model(model, train_data, train_labels, epochs, criterion, optimizer):
    model.train()
    
    for epoch in range(epochs):
        outputs = model(train_data)
        optimizer.zero_grad()

        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')