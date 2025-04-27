import torch
import torch.nn as nn
import torch.optim as optim
from model.CNNMulticlass import SpectrogramCNNMulticlass
from audioDataloader import dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpectrogramCNNMulticlass(num_etiquetas=2, num_nombres=20000).to(device)

loss_fn_etiqueta = nn.CrossEntropyLoss()
loss_fn_nombre = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    _, _, train_loader, val_loader = dataloaders()
    
    for inputs, labels in train_loader:
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        out_etiqueta, out_nombre = model(inputs)

        label_etiqueta = labels[:, 0]
        label_nombre = labels[:, 1]

        loss1 = loss_fn_etiqueta(out_etiqueta, label_etiqueta)
        loss2 = loss_fn_nombre(out_nombre, label_nombre)
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"[{epoch+1}] Train Loss: {epoch_loss:.4f}")

    # Validación
    model.eval()
    correct1, correct2, total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            out_etiqueta, out_nombre = model(inputs)

            pred1 = out_etiqueta.argmax(dim=1)
            pred2 = out_nombre.argmax(dim=1)

            correct1 += (pred1 == labels[:, 0]).sum().item()
            correct2 += (pred2 == labels[:, 1]).sum().item()
            total += labels.size(0)

    acc1 = 100 * correct1 / total
    acc2 = 100 * correct2 / total
    print(f"       Val Accuracy: etiqueta={acc1:.2f}%, nombre={acc2:.2f}%")