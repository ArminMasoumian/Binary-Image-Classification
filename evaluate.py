# Author: Armin Masoumian (masoumian.armin@gmail.com)

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import BinaryImageDataset
from model import BinaryImageClassifier

def evaluate(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    print('Accuracy on test set: {:.2f}%'.format(accuracy*100))

if __name__ == '__main__':
    # Set device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test dataset
    test_dataset = BinaryImageDataset(root_dir='data/test',
                                       transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load model checkpoint
    checkpoint = torch.load('models/binary_image_classifier.pth')
    model = BinaryImageClassifier()
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    # Evaluate model on test set
    evaluate(model, test_loader, device)
