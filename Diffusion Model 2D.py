import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


### Unconditional Diffusion Model
### Main function
def main_part_1_1_Unconditional_Diffusion_Model():
    denoiser, outputs, inputs, x0, dt, losses, n_points, T = Model_init_and_training_Unconditional_Diffusion_Model()
    visualize_the_model_results(outputs, inputs, x0)
    Q1_trajectory_for_a_chosen_point(denoiser, dt, T)
    Q2_plot_the_loss_function_over_trining(losses)
    Q3_plot_9_different_sampling_using_different_seeds(n_points, denoiser)
    Q4_plot_sampling_result_for_different_numbers_of_samplin_steps_T(denoiser, n_points)
    Q5_change_the_schedule_sampling_to_new_sigma()
    Q6_reverse_sampling_process(denoiser, T)


## Model
class DiffusionDenoiserUnconditional(nn.Module):
    def __init__(self):
        super(DiffusionDenoiserUnconditional, self).__init__()
        self.fc1 = nn.Linear(3, 1024)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x, t):
        t = t.repeat(x.size(0), 1)
        x = torch.cat([t, x], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def Model_init_and_training_Unconditional_Diffusion_Model():
    ## Dataset
    n_points = 1000
    x0 = np.random.uniform(low=-1, high=1, size=(n_points, 2))
    ## Init
    denoiser = DiffusionDenoiserUnconditional()
    optimizer = optim.Adam(denoiser.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_epochs = 6000
    batch_size = n_points
    T = 1000
    dt = -(1.0/T)  # Step size for sampling (negative)
    losses = []

    ## Training - Forward process
    inputs = torch.from_numpy(x0).float()
    for epoch in range(num_epochs):
        t = torch.randint(1, T, (1,)) * abs(dt)  # Sample random time t ∈ [0, 1]
        noise = torch.randn(batch_size, 2)  # Sample Gaussian noise ϵ ∼ N (0, I)
        sigma = torch.exp(5 * (t - 1))
        targets = inputs + sigma.unsqueeze(1) * noise
        outputs = denoiser(targets, t)

        loss = criterion(outputs, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
    return denoiser, outputs, inputs, x0, dt, losses, n_points, T


def visualize_the_model_results(outputs, inputs, x0):
    outputs = outputs + inputs
    outputs_np = outputs.detach().numpy()
    plt.scatter(x0[:, 0], x0[:, 1], label='Original Points')
    plt.scatter(outputs_np[:, 0], outputs_np[:, 1], label='Denoised Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Original Points vs Denoised Points')
    plt.legend()
    plt.show()


def Q1_trajectory_for_a_chosen_point(denoiser, dt, T):
    chosen_point = torch.tensor((3, 3), dtype=torch.float32).unsqueeze(0)
    trajectory = [chosen_point.detach().numpy()]
    sampling_steps = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512]  # max step is 1000
    with torch.no_grad():
        for t in range(0, T):
            t = torch.tensor(1 + dt*t)
            output = denoiser(chosen_point, t)
            chosen_point = chosen_point - output
            trajectory.append(chosen_point.detach().numpy())

    trajectory = np.array(trajectory)
    plt.figure(figsize=(8, 6))
    plt.scatter(trajectory[:, :,0], trajectory[:, :, 1], c=range(0, T+1), cmap='coolwarm')
    plt.colorbar(label='Time t')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory of Chosen Point')
    plt.show()


def Q2_plot_the_loss_function_over_trining(losses):
    plt.figure(figsize=(8, 6))
    plt.plot(range(0, len(losses), 10), losses[::10])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Function over Training Batches')
    plt.show()


def Q3_plot_9_different_sampling_using_different_seeds(n_points, model):
    seeds = range(1, 10)
    plt.figure(figsize=(12, 12))
    for seed in seeds:
        np.random.seed(seed)
        sample_points = ddim_sampling(1000, model, n_points).detach().numpy()
        plt.subplot(3, 3, seed)
        plt.scatter(sample_points[:, 0], sample_points[:, 1], s=5)
        plt.title('sampels')
        plt.xlabel('X')
        plt.ylabel('Y')
    plt.tight_layout()
    plt.show()

## DDIM Sampling
def reverse_process(score, z, t, dt):
    t = torch.full_like(z, t)
    dz = (torch.exp(5 * (t - 1)) * (5 * t * torch.exp(5 * (t - 1))) * score * -dt)
    return dz


def estimate_denoised(z, t, noise):
    expanded_t = torch.full_like(z, t)  # Expand t to match the shape of z and epsilon_hat
    x0 = z - (torch.exp(5 * (expanded_t - 1)) * noise)
    return x0


def ddim_sampling(T, denoiser, n_points):
    z = torch.randn(n_points, 2).float()  # sample from N(0, I)
    dt = -(1.0 / T)
    for t in range(0, T):
        t = torch.tensor(1 + dt * t)
        noise = denoiser(z, t)
        sample = estimate_denoised(z, t, noise)  # find denoised image
        sigma = torch.exp(5 * (torch.full_like(z, t) - 1)) ** 2
        score = (sample - z) / sigma  # gradient of log probability
        dz = reverse_process(score, z, t, dt)  # should use score! what is g?
        z += dz
    return z


def Q4_plot_sampling_result_for_different_numbers_of_samplin_steps_T(denoiser, n_points):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    T_values = [5, 25, 200, 500, 1000, 2000]

    for i, T_val in enumerate(T_values):
        curr_z = ddim_sampling(T_val, denoiser, n_points)  # Enter num of steps and get samples
        curr_z = curr_z.detach().numpy()
        row = i // 3
        col = i % 3
        axs[row, col].scatter(curr_z[:, 0], curr_z[:, 1], s=5, alpha=0.5)
        axs[row, col].set(xlabel='X', ylabel='Y')
        axs[row, col].set_title(f'T = {T_val}')
    plt.tight_layout()
    plt.show()


def Q5_change_the_schedule_sampling_to_new_sigma():
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue', 'red']
    T = 1000
    dt = 1.0 / T
    sample_times = np.arange(0, 1, dt)
    original_sigma = torch.exp(5 * (torch.tensor(sample_times) - 1))
    new_sigma = torch.exp(2 * (torch.tensor(sample_times) - 1))
    ax.plot(sample_times, original_sigma, color=colors[0], label=f'sigma = exp(5 * (t-1))')
    ax.plot(sample_times, new_sigma, color=colors[1], label=f'sigma = exp(2 * (t-1))')
    ax.set_xlabel('Time')
    ax.set_ylabel('Sigma')
    ax.set_title('Sampling Schedule')
    ax.legend()
    plt.show()


def Q6_reverse_sampling_process(denoiser, T):
    ## Q6 - ddim sampling
    outputs = [list() for i in range(10)]
    z = torch.randn(1, 2)
    z = z.repeat(10, 1)
    z2 = torch.tensor(z)  # save a copy for second run
    dt = -(1.0 / T)
    for t in range(0, T):
        t = torch.tensor(1 + dt*t)
        noise = denoiser(z, t)
        sample = estimate_denoised(z, t, noise)  # find denoised image
        sigma = torch.exp(5 * (torch.full_like(z, t) - 1))**2
        score = (sample - z) / sigma  # gradient of log probability
        dz = reverse_process(score, z, t, dt)
        z += dz
        for i in range(10):
            outputs[i].append(torch.tensor(z[i]))
    # plot the results on the same graph - only one line should be seen because they are the same
    plt.figure(figsize=(8, 6))
    for i in range(10):  # same value ten times!
        outputs[i] = torch.stack(outputs[i], dim=0).detach().numpy()
    for i in range(10):
        plt.scatter(outputs[i][:, 0], outputs[i][:, 1], c=range(0, T), cmap='coolwarm')
    plt.colorbar(label='Time t')
    labels = [f'Line {i+1}' for i in range(10)]
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Outputs')
    plt.show()

    ## second try - now add some noise while denoising:
    outputs = [list() for i in range(10)]
    z = z2
    dt = -(1.0 / T)
    for t in range(0, T):
        t = torch.tensor(1 + dt*t)
        noise = denoiser(z, t)
        sample = estimate_denoised(z, t, noise)  # find denoised image
        sample += 0.1 * torch.randn(sample.shape[0], 2)
        sigma = torch.exp(5 * (torch.full_like(z, t) - 1))**2
        score = (sample - z) / sigma  # gradient of log probability
        dz = reverse_process(score, z, t, dt)
        z += dz
        for i in range(4):
            outputs[i].append(torch.tensor(z[i]))
    # plot 4 results on the same graph - only one line should be seen because they are the same
    plt.figure(figsize=(8, 6))
    for i in range(4):
        outputs[i] = torch.stack(outputs[i], dim=0).detach().numpy()
    for i in range(4):
        plt.scatter(outputs[i][:, 0], outputs[i][:, 1], c=range(0, T), cmap='coolwarm')
    plt.colorbar(label='Time t')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Outputs When add noise while denoising')
    plt.show()


"""
##################################### PART 2 ##################################################
"""
### PART 1.2 Conditional Diffuusion Model
### Main function
def main_part_1_2_Conditional_Diffusion_Model():
    Udenoiser, outputs, inputs, dataset, labels, dt = Model_init_and_training_Conditional_Diffusion_Model()
    visualize_the_model_results(outputs, inputs, dataset)
    Q1_plot_classes(dataset, labels)
    Q3_trajectory_point_for_each_class(Udenoiser, dataset, labels, dt)
    Q4_plot_a_sampling_of_1000_points(Udenoiser, dt)
    Q6_find_probability(Udenoiser, dt)


class DiffusionDenoiserConditional(nn.Module):
    def __init__(self):
        super(DiffusionDenoiserConditional, self).__init__()
        # embedding dimension is 64
        # num of classes are five
        self.embedding = nn.Embedding(5, 64)
        self.fc1 = nn.Linear(3 + 64, 1024)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x, t, class_label):
        embedded_class = self.embedding(class_label)
        t = t.repeat(x.size(0), 1)
        x = torch.cat([t, x, embedded_class], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def Model_init_and_training_Conditional_Diffusion_Model():
    # Dataset of 5 classes odf rings near center
    n_points = 1000
    dataset = np.random.uniform(low=-1, high=1, size=(n_points, 2))
    labels = np.zeros(n_points, dtype=int)
    distances = np.linalg.norm(dataset, axis=1)

    # Assign labels based on distances from the origin
    labels[(distances <= 0.283)] = 0
    labels[(distances > 0.283) & (distances <= 0.283 * 2)] = 1
    labels[(distances > 0.283 * 2) & (distances <= 0.283 * 3)] = 2
    labels[(distances > 0.283 * 3) & (distances <= 0.283 * 4)] = 3
    labels[(distances > 0.283 * 4) & (distances <= 0.283 * 5)] = 4
    labels = torch.tensor(labels)

    # Add noise to the dataset
    dataset += np.random.normal(loc=0, scale=0.05, size=(n_points, 2))
    # Init
    Udenoiser = DiffusionDenoiserConditional()
    optimizer = optim.Adam(Udenoiser.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    num_epochs = 3000
    batch_size = n_points
    T = 1000
    dt = -(1.0/T)
    losses = []

    # Training
    inputs = torch.from_numpy(dataset).float()
    for epoch in range(num_epochs):
        t = torch.randint(1, T, (1,)) * abs(dt)  # Sample random time t ∈ [0, 1]
        noise = torch.randn(batch_size, 2)  # Sample Gaussian noise ϵ ∼ N (0, I)
        sigma = torch.exp(5 * (t - 1))
        targets = inputs + sigma.unsqueeze(1) * noise
        outputs = Udenoiser(targets, t, labels)

        loss = criterion(outputs, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    return Udenoiser, outputs, inputs, dataset, labels, dt


def Q1_plot_classes(dataset, labels):
    plt.figure(figsize=(8, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    class_names = ['Circle', 'Inner Ring', 'Second Ring', 'Third Ring', 'Fourth Ring']

    for i in range(5):
        class_points = dataset[labels == i]
        plt.scatter(class_points[:, 0], class_points[:, 1], color=colors[i], label=class_names[i])

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Input Dataset')
    plt.legend()
    plt.show()


def Q3_trajectory_point_for_each_class(Udenoiser, dataset, labels, dt):
    trajectories = [list() for _ in range(5)]
    T = int(1 / abs(dt))
    with torch.no_grad():
        for class_label in range(5):
            z = torch.randn(1, 2).float()  # sample from N(0, I)
            for t in range(0, T):
                t = torch.tensor(1 + dt * t)
                noise = Udenoiser(z, t, torch.tensor([class_label]))
                sample = estimate_denoised(z, t, noise)  # find denoised image
                sigma = torch.exp(5 * (torch.full_like(z, t) - 1)) ** 2
                score = (sample - z) / sigma  # gradient of log probability
                dz = reverse_process(score, z, t, dt)
                z += dz
                trajectories[class_label].append(torch.tensor(z).detach().numpy())
    colors = ['Blues', 'Reds', 'Greens', 'Oranges', 'Purples']
    class_names = ['First Ring', 'Second Ring', 'Third Ring', 'Fourth Ring', "Fifth Ring"]
    plt.figure(figsize=(8, 8))
    visualize_classes()
    for i, trajectory in enumerate(trajectories):
        trajectory = np.array(trajectory)
        plt.scatter(trajectory[:, :, 0], trajectory[:, :, 1], c=range(0, T), cmap=colors[i], label=class_names[i])
    plt.colorbar(label='Time - light is start and dark is end.')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectories of Chosen Point')
    plt.show()


def Q4_plot_a_sampling_of_1000_points(Udenoiser, dt):
    samples = [list() for _ in range(5)]
    T = int(1 / abs(dt))
    with torch.no_grad():
        for class_label in range(5):
            z = torch.randn(200, 2).float()  # sample from N(0, I)
            for t in range(0, T):
                t = torch.tensor([1 + dt * t])
                noise = Udenoiser(z, t, torch.tensor([class_label]*200))
                sample = estimate_denoised(z, float(t), noise)  # find denoised image
                sigma = torch.exp(5 * (torch.full_like(z, float(t)) - 1)) ** 2
                score = (sample - z) / sigma  # gradient of log probability
                dz = reverse_process(score, z, float(t), dt)
                z += dz
            samples[class_label].append(torch.tensor(z).detach().numpy())
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    class_names = ['First Ring', 'Second Ring', 'Third Ring', 'Fourth Ring', "Fifth Ring"]
    plt.figure(figsize=(8, 8))
    visualize_classes()
    for class_label in range(5):
        class_samples = np.array(samples[class_label])
        plt.scatter(class_samples[:, :, 0], class_samples[:, :, 1], color=colors[class_label], label=class_names[class_label])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Samples points')
    plt.show()


def visualize_classes():
    square = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1], [-1, -1]])
    circle = plt.Circle((0, 0), 0.283, color='blue', linestyle='-', fill=False)
    plt.gca().add_patch(circle)
    circle = plt.Circle((0, 0), 0.566, color='red', linestyle='-', fill=False)
    plt.gca().add_patch(circle)
    circle = plt.Circle((0, 0), 0.849, color='green', linestyle='-', fill=False)
    plt.gca().add_patch(circle)
    circle = plt.Circle((0, 0), 1.122, color='orange', linestyle='-', fill=False)
    plt.gca().add_patch(circle)
    circle = plt.Circle((0, 0), 1.415, color='purple', linestyle='-', fill=False)
    plt.gca().add_patch(circle)
    plt.plot(square[:, 0], square[:, 1], color='black', linestyle='-')


def visualize_the_points():
    plt.figure(figsize=(8, 8))
    points_within_distribution = list(
        [[0.1, 0.1, "blue"], [0.3, 0.3, "red"], [0.5, 0.5, "green"], [0.7, 0.7, "orange"], [0.9, 0.9, "purple"]])
    for point in points_within_distribution:
        plt.scatter(point[0], point[1], color=point[2])
    points_outside_distribution = np.array([[2.0, 2.0], [1.5, -1.5], [-3.0, -0.5]])
    plt.scatter(points_outside_distribution[:, 0], points_outside_distribution[:, 1], color='gray')
    visualize_classes()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points on Top of Data')
    plt.show()
    return points_within_distribution, points_outside_distribution

def sigma(x, t):
    return torch.exp(5 * (torch.full_like(x, t) - 1))

def snr(x, t):
    return 1 / (sigma(x, t) ** 2)


def Q6_find_probability(Udenoiser, dt):
    T = int(1 / abs(dt))
    points_within_distribution, points_outside_distribution = visualize_the_points()
    # estimate probability of points within distribution
    for class_label, point in enumerate(points_within_distribution):
        x = torch.tensor([point[0], point[1]])
        snr_value = 0
        for i in range(1000):
            t = torch.randint(1, T, (1,)) * abs(dt)  # Sample random time t ∈ [0, 1]
            noise = torch.randn(1, 2)
            x_t = x + sigma(x, float(t)) * noise
            denoised_sample = Udenoiser(x_t, t, torch.tensor([class_label]))
            x_0 = x_t - sigma(x, float(t)) * denoised_sample
            snr_t_dt = snr(x, float(t) + dt)
            snr_value += float((snr_t_dt - snr(x, float(t)))[0] * (torch.norm(x - x_0, dim=1) ** 2))
        probs = -(T / 2) * snr_value / 1000
        print(f"The estimated probabilty for the point: {point} is {probs}")

    # estimate probability for a point with wrong class
    point = points_within_distribution[-1]
    x = torch.tensor([point[0], point[1]])
    snr_value = 0
    for i in range(1000):
        t = torch.randint(1, T, (1,)) * abs(dt)  # Sample random time t ∈ [0, 1]
        noise = torch.randn(1, 2)
        x_t = x + sigma(x, float(t)) * noise
        fake_class_label = 0  # the real one is 4
        denoised_sample = Udenoiser(x_t, t, torch.tensor([fake_class_label]))
        x_0 = x_t - sigma(x, float(t)) * denoised_sample
        snr_t_dt = snr(x, float(t) + dt)
        snr_value += float((snr_t_dt - snr(x, float(t)))[0] * (torch.norm(x - x_0, dim=1) ** 2))
    probs = -(T / 2) * snr_value / 1000
    print(f"The estimated probabilty for the point: {point} with the wrong class: 'blue' is {probs}")

    # estimate probability for points outside the distribution
    points = points_outside_distribution
    for point in points:
        x = torch.tensor([point[0], point[1]], dtype=torch.float32)
        snr_value = 0
        for i in range(1000):
            t = torch.randint(1, T, (1,)) * abs(dt)  # Sample random time t ∈ [0, 1]
            noise = torch.randn(1, 2)
            x_t = x + sigma(x, float(t)) * noise
            most_outer_label = 4  # no label is correct - so I chose the closest one
            denoised_sample = Udenoiser(x_t, t, torch.tensor([most_outer_label]))
            x_0 = x_t - sigma(x, float(t)) * denoised_sample
            snr_t_dt = snr(x, float(t) + dt)
            snr_value += float((snr_t_dt - snr(x, float(t)))[0] * (torch.norm(x - x_0, dim=1) ** 2))
        probs = -(T / 2) * snr_value / 1000
        print(f"The estimated probability for the point: {point} which is out of distribution is {probs}")


if __name__ == '__main__':
    main_part_1_1_Unconditional_Diffusion_Model()
    main_part_1_2_Conditional_Diffusion_Model()