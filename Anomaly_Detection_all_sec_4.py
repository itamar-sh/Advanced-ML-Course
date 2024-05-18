import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc
import random
from sklearn.metrics.pairwise import pairwise_distances

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256
dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
class Encoder(nn.Module):
    def __init__(self, D=128, device='cuda'):
        super(Encoder, self).__init__()
        self.resnet = resnet18(pretrained=False).to(device)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(512, 512)
        self.fc = nn.Sequential(nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, D))

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def encode(self, x):
        return self.forward(x)


# load encoders
encoder = Encoder()
encoder3 = Encoder()
encoder.load_state_dict(torch.load('model.pth'))
encoder3.load_state_dict(torch.load('encoder3.pth'))

# find representations
encoder.eval()
representations = []
with torch.no_grad():
    for images, _ in dataloader:
        images = images.to(device)
        images = [train_transform(images[i]) for i in range(len(images))]
        images = torch.stack(images).to(device)
        features = encoder.encode(images)
        representations.append(features.cpu().numpy())
representations1 = np.concatenate(representations, axis=0)

encoder3.eval()
representations3 = []

with torch.no_grad():
    for images, _ in dataloader:
        images = images.to(device)
        images = [train_transform(images[i]) for i in range(len(images))]
        images = torch.stack(images).to(device)
        features = encoder3.encode(images)
        representations3.append(features.cpu().numpy())

representations3 = np.concatenate(representations3, axis=0)

# Q1

resize_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
mnist_dataset = MNIST(root='./data', train=False, download=True, transform=resize_transform)
mnist_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)


def find_density_estimation(encoder, representations, show_msg):


    cifar_data_loader = test_dataloader

    encoder.eval()
    representations_tests = []

    with torch.no_grad():
        for images, _ in cifar_data_loader:
            images = images.to(device)
            augmented_images = [train_transform(images[i]) for i in range(len(images))]
            augmented_images = torch.stack(augmented_images).to(device)
            features = encoder.encode(augmented_images)
            representations_tests.append(features.cpu().numpy())

        for images, _ in mnist_dataloader:
            images = images.to(device)
            images = [images[i].repeat(3, 1, 1) for i in range(len(images))]
            augmented_images = [train_transform(images[i]) for i in range(len(images))]
            augmented_images = torch.stack(augmented_images).to(device)
            features = encoder.encode(augmented_images)
            representations_tests.append(features.cpu().numpy())

    representations_tests_con = np.concatenate(representations_tests, axis=0)
    k = 2
    cifar_neighbors = NearestNeighbors(n_neighbors=k).fit(representations)
    distances_cifar, indices_cifar = cifar_neighbors.kneighbors(representations_tests_con[:len(test_dataset)])

    mnist_neighbors = NearestNeighbors(n_neighbors=k).fit(representations)
    distances_mnist, indices_mnist = mnist_neighbors.kneighbors(representations_tests_con[len(test_dataset):])

    distances = np.concatenate((distances_cifar, distances_mnist), axis=0)
    indices = np.concatenate((indices_cifar, indices_mnist), axis=0)

    # compute the inverse density scores.
    inverse_density_scores = np.mean(distances, axis=1)
    plt.plot(list(range(len(inverse_density_scores))), inverse_density_scores)
    plt.xlabel('number of sample')
    plt.ylabel('density estimation')
    plt.title(show_msg)
    plt.show()

    return inverse_density_scores


inverse_density_scores_q1 = find_density_estimation(encoder, representations1, "knn - density estimation of the original model")

inverse_density_scores_q6 = find_density_estimation(encoder3, representations3, "knn-density estimation no genereated neighbors model")


# Q2
def plot_ROC_curve(inverse_density_scores, msg_show):
    true_labels = np.concatenate((np.zeros(len(test_dataset)), np.ones(len(mnist_dataset))), axis=0)

    # Compute the False Positive Rate (FPR) and True Positive Rate (TPR) for different thresholds
    fpr, tpr, thresholds = roc_curve(true_labels, inverse_density_scores)

    # Compute the Area Under the Curve (AUC) of the ROC curve
    auc_score = auc(fpr, tpr)

    # Plot the ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label='VICReg (AUC = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')  # Random classifier curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Anomaly Detection')
    plt.legend(loc="lower right")
    plt.show()

plot_ROC_curve(inverse_density_scores_q1, "ROC Curve of the first model")


# Q3
def present_anomalies(inverse_density_scores):
    top_indices = np.argsort(inverse_density_scores)[::-1][:7]

    print(top_indices)

    # Get the images corresponding to the top indices
    top_images_success = []
    top_images_fails = []

    for idx in top_indices:
        if idx < 10000:
            top_images_fails.append(test_dataset[idx][0])
        else:
            top_images_success.append(mnist_dataset[idx-10000][0])

    # Plot the images
    plt.figure(figsize=(12, 6))
    for i, image in enumerate(top_images_success):
        plt.subplot(1, 7, i+1)
        plt.imshow(image.squeeze().numpy(), cmap="gray")
        plt.axis('off')

    plt.figure(figsize=(12, 6))
    for i, image in enumerate(top_images_fails):
        plt.subplot(1, 7, i+1)
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        plt.axis('off')

    plt.show()

present_anomalies(inverse_density_scores_q1)
present_anomalies(inverse_density_scores_q6)