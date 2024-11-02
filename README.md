# Advanced-ML-Course

# Generative Modelling
Diffusion Models, Auto Regressive Models (GPT-2)

## Training a 2D Diffusion Model:
Developed a diffusion model composed for 2D points with forward and reverse processes, implementing stochastic differential equations (SDE) to gradually convert inputs from their natural distribution to Gaussian distribution and a reverse process, denoising samples back to their original distribution. Using Stochastic Differential Equations (SDEs) and assuming a Variance Exploding (VE) process, the forward process involves shrinking inputs and adding noise, while the reverse process trains a denoiser to reconstruct the original inputs. The denoiser predicts added noise instead of the original points, and the loss function is based on the prediction of noise rather than its product with the variance. Additionally, the reverse sampling process employs DDIM (Denosing Diffusion Implicit Models) sampling, and the probability of given inputs is estimated for probabilistic inference.

Examples:<br/>
<img
  src="resources/Diffusion Models/Original Point vs denoised Points.png"
  style="display: inline-block; margin: 0 auto;" width="350" height="200"><br/>
<img
  src="resources/Diffusion Models/Trajecotry of chosen point.png"
  style="display: inline-block; margin: 0 auto;" width="350" height="200"><br/>
<img
  src="resources/Diffusion Models/Outputs when add noise while denoising.png"
  style="display: inline-block; margin: 0 auto;" width="350" height="200"><br/>
<img
  src="resources/Diffusion Models/Screenshot from 2024-05-20 17-26-37.png"
  style="display: inline-block; margin: 0 auto;" width="350" height="200">
<img
  src="resources/Diffusion Models/Screenshot from 2024-05-20 17-26-43.png"
  style="display: inline-block; margin: 0 auto;" width="350" height="200"><br/>


## Auto-Regressive Text Model
Trained a GPT-2 model on "Alice in Wonderland" text using Byte Pair Encoder, treating text generation as a classification problem. Split text into blocks for training, optimized with Cross-Entropy loss. Explored inversion process to find input vector matching a given sentence, enhancing understanding of transformer architecture and limitations of auto-regressive models.

Examples:<br/>
<img
  src="resources/Auto Regressive model/AR1.png"
  style="display: inline-block; margin: 0 auto;" width="600" height="300"><br/>
<img
  src="resources/Auto Regressive model/AR2.png"
  style="display: inline-block; margin: 0 auto;" width="500" height="400"><br/>
<img
  src="resources/Auto Regressive model/AR4.png"
  style="display: inline-block; margin: 0 auto;" width="500" height="400"><br/>
<img
  src="resources/Auto Regressive model/AR3.png"
  style="display: inline-block; margin: 0 auto;" width="370" height="80"><br/>

# Self-Supervised &amp; Unsupervised Learning
VICReg algorithm, Probing, Anomaly Detection.


## VICReg algorithm
To Do


## General Q&A on ML

### Q1) What GPU contains?
GPU is separated in something called Streaming Multiprocessors (SMs). Around ~100 SM in each GPU.
Each SM contains a set of CUDA cores (~64), tensor cores (~4), shared memory (~164KB), registers(~40MB Each), and other components.

GPU Contains thousands of CUDA cores (~5000-10000), which are smaller processors built for for basic arithmetic and logical operations which are good for matrix multiplications, additions, and element-wise operations.

Above that GPU has his own global memory (~40GB) it's called VRAM. This is where the CPU's transfer the ready for training batches.

Moreover Every GPU contains around ~432 tensor cores, each SM has ~4. They faster than CUDA cores but uses FP16 (Floating Points) or Int8 which mean the NN must supports low precision.


### Q2) What CPU contains?
Each CPU core has its own control unit and arithmetic logic unit (ALU), which handles mathematical and logical operations. Unlike GPU cores, CPU cores are complex, allowing for a wide range of calculations and functions.

The CPU has an integrated memory controller that manages data flow between the CPU and RAM. While GPUs also have memory controllers, CPUs typically have more sophisticated controllers to handle complex memory hierarchies and ensure data consistency across cores.

L3 Cache: Larger and shared among all cores, used for data that multiple cores may need to access.

Clock Speed: CPUs are typically clocked higher than GPUs, with speeds commonly ranging from 2 GHz to 5 GHz. Higher clock speeds mean they can complete individual tasks faster, which is advantageous for sequential tasks.

### Q3) CPU advantages over GPU:
Parallelism: CPUs have fewer, more powerful cores designed for sequential processing, while GPUs have thousands of simpler cores optimized for parallel tasks.

Cache Hierarchy: CPU caches are larger and more sophisticated, catering to complex workflows with frequent branching and varied data access patterns.

Instruction Complexity: CPUs support a broad range of instructions, including complex control-flow mechanisms, making them versatile for general-purpose computing.

### Q4) How exactly the weights changes in trainning:
This is how each step is look like:

```
optimizer.zero_grad()
output = model(data)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

The line ` optimizer.zero_grad() ` simply clear the .grad element of the weights from the last step.

The line ` output = model(data) ` is the forward step which takes the data via the net (multiply by the weights) and return the output.

The line ` loss = criterion(output, target) ` the criterion measure the difference of output and target via the loss function. The result is scalar value representing the error for the batch.

The line ` loss.backward ` initiate the backpropogation process which calculates the gradients of the loss with respect to the each weight in the net. More specificaly the network computes the partial derivative of the loss with respect to each weight. Then weight.grad get a value for each weight. So this process actually move on the net from the end to the start with the loss tensor as input and calculate the gradients via the chain rule.

The last line ` optimizer.step() ` update each weight in relation to the optimization algorithm such as
SGD: w = w - learning_rate * w.grad
Or Adam which uses momentum and adaptive learning rates.

### Q5) What is the use of Gradient in all those learinning?
The gradient shows the direction of the steepest increase in loss, so by moving in the opposite direction (down the gradient), we get closer to the minimum loss and, therefore, a better solution.

### Q6) Basic steps to improve GPU training:
1) Increase batch size - a larger batch size allows more data to be processed simultaneously on the GPU, improving throughput, we need to be know sometimes it's not possible since we have memory limit and also it can effect the training results.

2) Use parallel data loaders, num_workers, and prefetching to reduce the CPU bottleneck in feeding data to the GPU. The thumb rule is num_workers=cpu_cores.

3) Mixed Precision Training: Use mixed-precision to reduce memory usage and increase throughput, while maintaining model accuracy. Quantization is also possible in some cases.

4) Model Parallelism and Distributed Training: For very large models, split the model across multiple GPUs (model parallelism) or use data parallelism to train on multiple GPUs at once.

5) Using GPU for dataloader: we can use the GPU himself to load data to the device like this:
```
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
inputs, labels = data[0].to(device=device, non_blocking=True), data[1].to(device=device, non_blocking=True)
```
1 - The pin memory is a flag that make the data to stay on the physical memory and give direct memory access to the GPU without involving the CPU for the entire transfer duration.
2 - When using the .to(device) we give the flag non_blocking=True which allows the transfer of data to the GPU to happen asynchronously, meaning that the main thread doesn’t have to wait for the transfer to finish before moving on. This can speed up training by overlapping the data transfer and the computation if there are no dependencies on the data transfer to complete immediately.

6) Apply data type changes on the GPU: In case of changing from small to big data type changes like int8 to float32 than it would be wise first to move the data into the GPU and than convert him.

### Q7) Basic steps to improve CPU training:
1) Multi-threading: Utilize multiple worker threads for data loading. In PyTorch, you can specify num_workers in the DataLoader to load batches in parallel.
2) Data Preprocessing: Ensure data is preprocessed efficiently. Use libraries like NumPy for vectorized operations instead of loops.
3) right batch size: first we don't get the benefit of maximize the utilization of GPU memory so we don't have any need to use bigger batch size. CPU memory management is complicated than GPU so we need to try to get the right balance.
4) Use mixed precision (16-bit floating point) if your CPU supports it. This can lead to faster computations and reduced memory usage.
5) If you have access to multiple CPU cores, consider distributing your model across them. This is more complex but can yield speed improvements. If you have access to multiple machines, consider using distributed training frameworks (like Horovod) to parallelize the training across several CPUs.