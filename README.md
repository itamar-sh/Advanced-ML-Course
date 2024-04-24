# Advanced-ML-Course

## Generative Modelling
Diffusion Models, Auto Regressive Models (GPT-2)

#### Training a 2D Diffusion Model:
Developed a diffusion model composed of forward and reverse processes, implementing stochastic differential equations (SDE) to gradually convert inputs from their natural distribution to Gaussian distribution and vice versa. Utilized Variance Exploding (VE) process and trained a denoiser to reverse the diffusion process.


## Self-Supervised &amp; Unsupervised Learning
VICReg algorithm, Probing, Anomaly Detection.

#### Training an Auto-Regressive Text Model:
Trained a GPT-2 model on "Alice in Wonderland" text using Byte Pair Encoder, treating text generation as a classification problem. Split text into blocks for training, optimized with Cross-Entropy loss. Explored inversion process to find input vector matching a given sentence, enhancing understanding of transformer architecture and limitations of auto-regressive models.
