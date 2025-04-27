import torch
import torch.nn as nn
import torch.optim as optim
from model.CNN import SpectrogramCNN
from audioDataloader import dataloaders
from sklearn.metrics import precision_score, recall_score, f1_score


def logits_loss(train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SpectrogramCNN().to(device)

    criterio = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            
            inputs = inputs.to(device)
            
            labels = labels.float().to(device) 

            optimizer.zero_grad()
           
            outputs = model(inputs)
            

            loss = criterio(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"[{epoch+1}] Train Loss: {epoch_loss:.4f}")

        # Validación
        model.eval()
        all_preds = []
        all_labels = []

    
        correct, total = 0, 0

        with torch.no_grad():
            
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                preds = torch.sigmoid(outputs) > 0.5  
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        acc = 100 * correct / total
        
        print("Example preds:", all_preds[:10])
        print("Example labels:", all_labels[:10])
        print(f"Val Accuracy: {acc:.2f}% | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}")
        
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

    logits_loss(train_loader, val_loader, epochs=10)

if __name__ == "__main__":
    main()