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

# ***Dataset*** prepare
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256
dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# init model
encoder = Encoder().to(device)
projector = Projector().to(device)


def plot_losses(sum_losses, inv_losses, var_losses, cov_losses, step = "epochs"):
    epochs = list(range(1, len(sum_losses) + 1))
    plt.plot(epochs, sum_losses, label='Overall Loss')
    plt.plot(epochs, inv_losses, label='Invariance Loss')
    plt.plot(epochs, var_losses, label='Variance Loss')
    plt.plot(epochs, cov_losses, label='Covariance Loss')
    plt.xlabel(step)
    plt.ylabel('Loss')
    plt.title(f"Losses over {step}")
    plt.legend()
    plt.show()

    epochs = list(range(1, len(sum_losses) + 1))
    plt.plot(epochs, inv_losses, label='Invariance Loss')
    plt.xlabel(step)
    plt.ylabel('Invariance Loss')
    plt.title(f"Invariance Loss over {step}")
    plt.legend()
    plt.show()

    epochs = list(range(1, len(sum_losses) + 1))
    plt.plot(epochs, var_losses, label='Variance Loss')
    plt.xlabel(step)
    plt.ylabel('Variance Loss')
    plt.title(f"Variance Losד over {step}")
    plt.legend()
    plt.show()


    epochs = list(range(1, len(sum_losses) + 1))
    plt.plot(epochs, cov_losses, label='Covariance Loss')
    plt.xlabel(step)
    plt.ylabel('Covariance Loss')
    plt.title(f"Covariance Loss over {step}")
    plt.legend()
    plt.show()

# train encoder

def train_ssl(encoder, projector, gamma=25, mu=25, v=1, epochs=25):
    sum_losses, inv_losses, var_losses, cov_losses = [], [], [], []
    sum_losses_test, inv_losses_test, var_losses_test, cov_losses_test = [], [], [], []
    encoder.train()
    projector.train()
    optimizer = optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=3e-4, betas=(0.9, 0.999),
                           weight_decay=1e-6)
    num_epochs = epochs
    for epoch in range(num_epochs):
        for batch in dataloader:
            images = batch[0]

            augmented_images = [train_transform(images[i]) for i in range(len(images))]
            augmented_images = torch.stack(augmented_images).to(device)
            Y = encoder(augmented_images)
            Z = projector(Y)
            augmented_images_prime = [train_transform(images[i]) for i in range(len(images))]
            augmented_images_prime = torch.stack(augmented_images_prime).to(device)
            Y_prime = encoder(augmented_images_prime)
            Z_prime = projector(Y_prime)

            inv_loss, var_loss, cov_loss = combined_loss(Z, Z_prime, gamma, mu, v)
            loss = inv_loss + var_loss + cov_loss

            sum_losses.append(loss.item())
            inv_losses.append(inv_loss.item())
            var_losses.append(var_loss.item())
            cov_losses.append(cov_loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        plot_losses(sum_losses, inv_losses, var_losses, cov_losses, step="batches")

        sum_sum_loss, sum_inv_loss, sum_var_loss, sum_cov_loss = 0, 0, 0, 0
        with torch.no_grad():
            for batch in test_dataloader:
                images = batch[0]

                augmented_images = [test_transform(images[i]) for i in range(len(images))]
                augmented_images = torch.stack(augmented_images).to(device)
                Y = encoder(augmented_images)
                Z = projector(Y)
                augmented_images_prime = [test_transform(images[i]) for i in range(len(images))]
                augmented_images_prime = torch.stack(augmented_images_prime).to(device)
                Y_prime = encoder(augmented_images_prime)
                Z_prime = projector(Y_prime)

                inv_loss, var_loss, cov_loss = combined_loss(Z, Z_prime, gamma, mu, v)
                loss = inv_loss + var_loss + cov_loss

                sum_sum_loss += loss.item()
                sum_inv_loss += inv_loss.item()
                sum_var_loss += var_loss.item()
                sum_cov_loss += cov_loss.item()

            loss = sum_sum_loss
            inv_loss = sum_inv_loss
            var_loss = sum_var_loss
            cov_loss = sum_cov_loss

            sum_losses_test.append(loss)
            inv_losses_test.append(inv_loss)
            var_losses_test.append(var_loss)
            cov_losses_test.append(cov_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Invariance Loss: {inv_loss}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Variance Loss: {var_loss}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Covariance Loss: {cov_loss}")
        plot_losses(sum_losses_test, inv_losses_test, var_losses_test, cov_losses_test)

train_ssl(encoder, projector)

####
##  Q2   ##
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def show_pca_tsne(encoder):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    encoder.eval()
    representations = []
    labels_list = []

    for batch in test_dataloader:
        images, labels = batch
        images = images.to(device)
        images = [test_transform(images[i]) for i in range(len(images))]
        images = torch.stack(images).to(device)

        with torch.no_grad():
            features = encoder.encode(images)
            representations.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

    representations = np.concatenate(representations, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    pca_result = PCA(n_components=2).fit_transform(representations)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis')
    class_values = np.unique(labels)
    scatter.set_cmap('tab10')
    scatter.set_clim(class_values.min(), class_values.max())
    plt.title('PCA 2D visualization')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    c = plt.colorbar(scatter, ticks=range(10), label='Class')
    c.ax.set_yticklabels(class_names)
    plt.show()

    t_res = TSNE(n_components=2, perplexity=28, random_state=40).fit_transform(representations)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(t_res[:, 0], t_res[:, 1], c=labels, cmap='viridis')
    class_values = np.unique(labels)
    scatter.set_cmap('tab10')
    scatter.set_clim(class_values.min(), class_values.max())
    plt.title('T-SNE 2D visualization')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    c = plt.colorbar(scatter, ticks=range(10), label='Class')
    c.ax.set_yticklabels(class_names)
    plt.show()

show_pca_tsne(encoder)


######
## Q3 - Linear probing
def linear_probing(encoder, num_epochs=10):
    epoch_losses = []
    epoch_acces = []

    encoder_dim = 128
    num_classes = 10
    learning_rate = 0.1
    momentum = 0.9

    classifier = nn.Linear(encoder_dim, num_classes).to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate, momentum=momentum)
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            images = [test_transform(images[i]) for i in range(len(images))]
            images = torch.stack(images).to(device)

            representations = encoder.encode(images)
            outputs = classifier(representations)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        epoch_losses.append(epoch_loss)
        epoch_acces.append(epoch_acc)
        print(f"Epoch [{epoch+1}/{num_epochs}]: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc*100:.2f}%")

    classifier.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            images = [test_transform(images[i]) for i in range(len(images))]
            images = torch.stack(images).to(device)

            representations = encoder.encode(images)
            outputs = classifier(representations).to(device)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    # plot loss
    epochs = list(range(1, len(epoch_losses) + 1))
    plt.plot(epochs, epoch_losses, label='Overall Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses over Epochs')
    plt.legend()
    plt.show()
    # plot accuracy
    epochs = list(range(1, len(epoch_acces) + 1))
    plt.plot(epochs, epoch_acces, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.show()

linear_probing(encoder)

#####
### Q4: Ablation 1 - No Variance Term
encoder_zero_var = Encoder().to(device)
projector_zero_var = Projector().to(device)
train_ssl(encoder_zero_var, projector_zero_var, mu=0, epochs=5)
show_pca_tsne(encoder_zero_var)
linear_probing(encoder_zero_var, num_epochs=20)

#####
### Q6: Ablation 3 - No Generated Neighbors
encoder.eval()
representations = []
with torch.no_grad():
    for images, _ in dataloader:
        images = images.to(device)
        images = [train_transform(images[i]) for i in range(len(images))]
        images = torch.stack(images).to(device)
        features = encoder.encode(images)
        representations.append(features.cpu().numpy())
representations = np.concatenate(representations, axis=0)

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

linear_probing(encoder3, num_epochs=20)

#####
### Q8
representations1 = representations

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

encoder.eval()
encoder3.eval()

class_indices = [[] for _ in range(10)]
for idx, label in enumerate(dataset.targets):
    class_indices[label].append(idx)
class_indices = [random.choice(indices) for indices in class_indices]

# print("class_indices", class_indices)

nearest_neighbors = []
farthest_neighbors = []

with torch.no_grad():
    for idx in class_indices:
        image, _ = dataset[idx]
        image = image.unsqueeze(0).to(device)

        neighbors1 = NearestNeighbors(n_neighbors=6).fit(representations1)
        _, nearest_indices1 = neighbors1.kneighbors(representations1)

        neighbors3 = NearestNeighbors(n_neighbors=6).fit(representations3)
        _, nearest_indices3 = neighbors3.kneighbors(representations3)
        nearest_images1 = [dataset[i] for i in nearest_indices1[idx].squeeze()]
        nearest_images3 = [dataset[i] for i in nearest_indices3[idx].squeeze()]
        nearest_neighbors.append((image, nearest_images1, nearest_images3))

        # find the 5 modt distance values
        farthest_indices1 = np.argsort(
            pairwise_distances(np.reshape(representations1[idx], (1, -1)), representations1, metric='euclidean'))[0][
                            -5:]
        farthest_indices3 = np.argsort(
            pairwise_distances(np.reshape(representations3[idx], (1, -1)), representations3, metric='euclidean'))[0][
                            -5:]
        farthest_images1 = [dataset[i] for i in farthest_indices1]
        farthest_images3 = [dataset[i] for i in farthest_indices3]
        farthest_neighbors.append((image, farthest_images1, farthest_images3))

for i, (image, nearest1, nearest3) in enumerate(nearest_neighbors):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 6, 1)
    plt.imshow(image.squeeze().cpu().numpy().transpose((1, 2, 0)))
    plt.title('Original - Q1')
    plt.axis('off')

    for j, neighbor in enumerate(nearest1):
        if j == 0:
            continue
        plt.subplot(1, 6, j + 1)
        plt.imshow(neighbor[0].squeeze().cpu().numpy().transpose((1, 2, 0)))
        plt.title(f"near {j + 1}")
        plt.axis('off')
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 6, 1)
    plt.imshow(image.squeeze().cpu().numpy().transpose((1, 2, 0)))
    plt.title('Original - Q3')
    plt.axis('off')

    for j, neighbor in enumerate(nearest3):
        if j == 0:
            continue
        plt.subplot(1, 6, j + 1)
        plt.imshow(neighbor[0].squeeze().cpu().numpy().transpose((1, 2, 0)))
        plt.title(f"near {j + 1}")
        plt.axis('off')
    plt.show()

# Plotting the farthest neighbors
for i, (image, farthest1, farthest3) in enumerate(farthest_neighbors):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 6, 1)
    plt.imshow(image.squeeze().cpu().numpy().transpose((1, 2, 0)))
    plt.title('Original - Q1')
    plt.axis('off')

    for j, neighbor in enumerate(farthest1):
        plt.subplot(1, 6, j + 2)
        plt.imshow(neighbor[0].squeeze().cpu().numpy().transpose((1, 2, 0)))
        plt.title(f"far {j + 1}")
        plt.axis('off')
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 6, 1)
    plt.imshow(image.squeeze().cpu().numpy().transpose((1, 2, 0)))
    plt.title('Original - Q3')
    plt.axis('off')

    for j, neighbor in enumerate(farthest3):
        plt.subplot(1, 6, j + 2)
        plt.imshow(neighbor[0].squeeze().cpu().numpy().transpose((1, 2, 0)))
        plt.title(f"far {j + 1}")
        plt.axis('off')

    plt.show()