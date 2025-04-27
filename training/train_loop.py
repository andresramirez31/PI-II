import torch
import torch.nn as nn
import torch.optim as optim
from model.CNN import SpectrogramCNN
from audioDataloader import dataloaders


def pipeline_una_clase(train_loader, val_loader, train_dataset, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = len(train_dataset.encoders["etiqueta"].classes_)

    model = SpectrogramCNN(num_classes).to(device)

    loss_fn= nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            
            inputs = inputs.to(device)
            
            labels = labels.float().to(device).squeeze()  # (B, 1) → (B,) si vienen así

            optimizer.zero_grad()
           
            outputs = model(inputs).float()
            

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[{epoch+1}] Train Loss: {epoch_loss:.4f}")

        # Validación
        model.eval()
    
        correct, total = 0, 0

        with torch.no_grad():
            
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).squeeze()

                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"Val Accuracy: {acc:.2f}%")
        
    return model

def main():
    # Instanciación de dataloaders con solo la columna 'etiqueta'
    train_loader, val_loader, train_dataset, val_dataset = dataloaders(
        train_csv="train.csv",
        test_csv="test.csv",
        batch_size=32
    )

    # Dataset solo con la label 'etiqueta'
    train_dataset.label_columns = ["etiqueta"]
    val_dataset.label_columns = ["etiqueta"]

    pipeline_una_clase(train_loader, val_loader, train_dataset, epochs=10)

if __name__ == "__main__":
    main()