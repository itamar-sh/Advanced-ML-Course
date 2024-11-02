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
