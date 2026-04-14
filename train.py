import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ridgelet_dataset import RidgeletLineDataset
from model import RidgeletInterpretNet
from preprocessing import ImagePreprocessor

def train_model():
    prep = ImagePreprocessor()
    _ = prep.process()
    p = prep.p
    print(f"Запуск обучения для p = {p}")
    dataset = RidgeletLineDataset(p=p, samples_per_class=200)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RidgeletInterpretNet(p=p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Эпоха {epoch + 1}/{epochs} | Потери: {running_loss / len(train_loader):.4f} | Точность: {accuracy:.2f}%")
    torch.save(model.state_dict(), "ridgelet_net.pth")
    print("Обучение завершено. Веса сохранены в 'ridgelet_net.pth'")

if __name__ == "__main__":
    train_model()
