import torch
import random
import torch.nn as nn
from torchvision.models import resnet18
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc


batch_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

class Projector(nn.Module):
    def __init__(self, D=128, proj_dim=512):
        super(Projector, self).__init__()
        self.model = nn.Sequential(nn.Linear(D, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim)
                                   )
    def forward(self, x):
        return self.model(x)


def invariance_loss(Z, Z_prime):
    return F.mse_loss(Z, Z_prime)


def variance_loss(Z, Z_prime):
    eps = 1e-4
    zv = Z.var(dim=0)
    zvp = Z_prime.var(dim=0)
    std_z1 = torch.sqrt(zv + eps)
    std_z2 = torch.sqrt(zvp + eps)
    fr1 = F.relu(1 - std_z1)
    fr2 = F.relu(1 - std_z2)
    return torch.mean(fr1) + torch.mean(fr2)


def covariance_loss(Z, Z_prime):
    B, D = Z.size()
    Z_mean = Z - Z.mean(dim=0)
    Z_prime_mean = Z_prime - Z_prime.mean(dim=0)
    cov_Z = torch.matmul(Z_mean.T, Z_mean)
    cov_Z = cov_Z / (B - 1)
    cov_Z_prime = torch.matmul(Z_prime_mean.T, Z_prime_mean)
    cov_Z_prime = cov_Z_prime / (B - 1)

    diag = torch.eye(D, device=device)
    db = ~diag.bool()
    cov_z_loss = cov_Z[db].pow_(2).sum()
    cov_z_p_loss = cov_Z_prime[db].pow_(2).sum()
    return (cov_z_loss + cov_z_p_loss) / D


def combined_loss(Z, Z_prime, gamma=25, mu=25, v=1):
    inv_loss = gamma * invariance_loss(Z, Z_prime)
    var_loss = mu * variance_loss(Z, Z_prime)
    cov_loss = v * covariance_loss(Z, Z_prime)
    return inv_loss, var_loss, cov_loss



# load the encoders
encoder = Encoder()
encoder3 = Encoder()
encoder.load_state_dict(torch.load('model.pth'))


# eval the representations
encoder.eval()
representations = []
with torch.no_grad():
    for images, _ in dataloader:
        images = images.to(device)
        images = [train_transform(images[i]) for i in range(len(images))]
        images = torch.stack(images).to(device)
        features = encoder.encode(images)
        representations.append(features.cpu().numpy())
representations = representations1 = np.concatenate(representations, axis=0)
# train encoder no generate neighbors

n_neighbors = 4  # Including the image itself
neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(representations)
_, indices = neighbors.kneighbors(representations)

encoder3 = Encoder().to(device)
encoder3 = encoder3.train()
projector3 = Projector().to(device)
projector3 = projector3.train()
optimizer = optim.Adam(list(encoder3.parameters()) + list(projector3.parameters()), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-6)

for epoch in range(1):
    i = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # neighbor images init
        batch_indices = indices[len(images) * (i):len(images) * (i + 1), :]
        random_choice = np.random.randint(1, 4, size=(len(images),))
        batch_indices = batch_indices[np.arange(len(images)), random_choice]
        neighbor_images = torch.stack([dataset[idx][0] for idx in batch_indices], dim=0).to(device)
        # view 1 - the image itself
        images = [train_transform(images[i]) for i in range(len(images))]
        images = torch.stack(images).to(device)
        Y = encoder3(images)
        view1 = projector3(Y)
        # view 2 - the neighbor image
        augmented_images_prime = [train_transform(neighbor_images[i]) for i in range(len(neighbor_images))]
        augmented_images_prime = torch.stack(augmented_images_prime).to(device)
        Y_prime = encoder3(augmented_images_prime)
        view2 = projector3(Y_prime)
        # loss stage
        inv_loss, var_loss, cov_loss = combined_loss(view1, view2)
        loss = inv_loss + var_loss + cov_loss

        loss.backward()
        optimizer.step()
        i += 1

# eval no generate neighbors representations
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


### Q1

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

# activate original model
inverse_density_scores_q1 = find_density_estimation(encoder, representations1, "knn - density estimation of the original model")
# activate 'NO generated neighbors" model
inverse_density_scores_q6 = find_density_estimation(encoder3, representations3, "knn-density estimation no genereated neighbors model")

### Q2
def plot_ROC_curve(inverse_density_scores, msg_show):
    true_labels = np.concatenate((np.zeros(len(test_dataset)), np.ones(len(mnist_dataset))), axis=0)
    fpr, tpr, thresholds = roc_curve(true_labels, inverse_density_scores)
    auc_score = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='VICReg (AUC = %0.2f)' % auc_score)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('false positive eate')
    plt.ylabel('true positive rate')
    plt.title('ROC Curve - Anomaly Detection')
    plt.legend(loc="lower right")
    plt.show()

plot_ROC_curve(inverse_density_scores_q1, "ROC Curve of the first model")
plot_ROC_curve(inverse_density_scores_q6, "ROC Curve of no genereated neighbors model")

def present_anomalies(inverse_density_scores):
    top_indices = np.argsort(inverse_density_scores)[::-1][:7]
    top_images_success = []
    top_images_fails = []

    for idx in top_indices:
        if idx < 10000:
            top_images_fails.append(test_dataset[idx][0])
        else:
            top_images_success.append(mnist_dataset[idx-10000][0])
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

# Anomalies for the original model
present_anomalies(inverse_density_scores_q1)

# Anomalies for the "No generated neighbor model"
present_anomalies(inverse_density_scores_q6)