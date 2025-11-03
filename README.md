# VAE MNIST - Variational Autoencoder for Handwritten Digits

A PyTorch implementation of a Variational Autoencoder (VAE) trained on the MNIST dataset for unsupervised learning of handwritten digit representations and generation.

## Overview

This project implements a VAE that learns to:
- **Encode** handwritten digits into a compact 32-dimensional latent space
- **Decode** latent representations back to realistic digit images
- **Generate** new digit-like images by sampling from the learned latent space
- **Reconstruct** input images with high fidelity

## Model Architecture

### Encoder
- Input: 784-dimensional flattened MNIST images (28×28)
- Hidden layer: 512 neurons with ReLU activation
- Output: 32-dimensional latent space (μ and log σ²)

### Decoder
- Input: 32-dimensional latent vector
- Hidden layer: 512 neurons with ReLU activation
- Output: 784-dimensional reconstruction with sigmoid activation

### Loss Function
Combines two components:
- **Reconstruction Loss**: Binary Cross-Entropy between input and reconstruction
- **KL Divergence**: Regularizes latent space to follow standard normal distribution

## Key Features

- **Unsupervised Learning**: No digit labels used during training
- **Mixed Precision Training**: Uses AMP for faster computation
- **Comprehensive Logging**: Tracks BCE and KLD components separately
- **Visual Outputs**: Generates sample grids and reconstruction comparisons
- **Checkpointing**: Saves model state after each epoch

## Requirements

```bash
torch
torchvision
tqdm
matplotlib
```

## Usage

### Training
```bash
python vae_mnist.py
```

### Configuration
Key hyperparameters (modify in `vae_mnist.py`):
```python
batch_size = 128      # Training batch size
epochs = 20           # Number of training epochs
lr = 1e-3            # Learning rate
latent_dim = 32      # Latent space dimensionality
hidden_dim = 512     # Hidden layer size
```

## Output Structure

After training, the following files are generated in `vae_outputs/`:

```
vae_outputs/
├── samples/                    # Generated digit samples
│   ├── sample_epoch_001.png   # 8×8 grid of generated digits
│   ├── sample_epoch_002.png
│   └── ...
├── recons/                     # Reconstruction comparisons
│   ├── recon_epoch_001.png    # Original vs reconstructed images
│   ├── recon_epoch_002.png
│   └── ...
├── vae_epoch_001.pt           # Model checkpoints
├── vae_epoch_002.pt
└── ...
```

## Results

### Training Progress
- Loss decreases from ~162 to ~103 over 20 epochs
- Reconstruction error (BCE): ~115 → ~75
- KL divergence stabilizes around ~25-27
- Test loss: ~128 → ~103

### Capabilities
1. **Generation**: Sample random latent codes to create new digit-like images
2. **Reconstruction**: Faithfully reconstruct input digits
3. **Interpolation**: Smooth transitions between different digit styles
4. **Representation**: Learn meaningful 32D embeddings of digit structure

## Technical Details

### Reparameterization Trick
Enables backpropagation through stochastic sampling:
```python
z = μ + σ * ε, where ε ~ N(0,1)
```

### Loss Components
- **BCE Loss**: Treats each pixel as independent Bernoulli variable
- **KL Loss**: Encourages latent distribution to match N(0,1)

### Data Preprocessing
- Images normalized to [0,1] range for compatibility with BCE loss
- No data augmentation (focuses on learning core digit structure)

## Example Usage

### Load Trained Model
```python
import torch
from vae_mnist import VAE

# Load checkpoint
checkpoint = torch.load('vae_outputs/vae_epoch_020.pt')
model = VAE(latent_dim=checkpoint['latent_dim'])
model.load_state_dict(checkpoint['model_state'])

# Generate new samples
with torch.no_grad():
    z = torch.randn(16, 32)  # 16 random latent codes
    samples = model.decode(z)
```

### Interpolation Between Digits
```python
# Interpolate between two latent points
z1, z2 = torch.randn(1, 32), torch.randn(1, 32)
interpolated = interpolate(model, z1, z2, steps=8)
```

## Performance Notes

- Trains in ~3 seconds per epoch on CPU
- Uses mixed precision for efficiency (disabled on non-CUDA systems)
- Memory efficient with batch processing
- No multiprocessing for DataLoader (for compatibility)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Extensions

Potential improvements:
- β-VAE with adjustable KL weighting
- Convolutional architecture for better image modeling
- Conditional VAE using digit labels
- Disentangled representation learning
- Higher resolution image generation