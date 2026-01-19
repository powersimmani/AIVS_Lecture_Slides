# Lecture 15 - Part 6: Hands-on & Applications

## Overview
This part covers practical GAN implementation and real-world applications.

## Key Topics

### 1. Basic GAN Implementation
```python
# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

### 2. Training Loop
```python
for epoch in range(epochs):
    for real_imgs, _ in dataloader:
        # Train Discriminator
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z)

        d_loss_real = criterion(discriminator(real_imgs), ones)
        d_loss_fake = criterion(discriminator(fake_imgs.detach()), zeros)
        d_loss = d_loss_real + d_loss_fake

        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z)
        g_loss = criterion(discriminator(fake_imgs), ones)

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()
```

### 3. GAN Applications - Image Generation
**Face Generation**: StyleGAN, PGGAN
**Art Generation**: Creative adversarial networks
**High-Resolution**: Progressive growing techniques

### 4. GAN Applications - Image-to-Image
**Pix2Pix**: Paired image translation
- Edges → Photo
- Sketch → Color image
- Day → Night

**CycleGAN**: Unpaired translation
- Horse ↔ Zebra
- Photo ↔ Painting
- Summer ↔ Winter

### 5. GAN Applications - Other Domains
**Super Resolution**: SRGAN, ESRGAN
- Enhance image resolution
- Photo restoration

**Inpainting**: Fill missing regions
- Remove objects
- Reconstruct damaged images

**Text-to-Image**: Early approaches
- StackGAN
- AttnGAN

**Data Augmentation**:
- Generate training samples
- Balance imbalanced datasets

### 6. Best Practices
**Training Tips**:
- Use two-timescale update rule
- Monitor both losses
- Save checkpoints frequently
- Visualize generations regularly

**Common Hyperparameters**:
- Learning rate: 0.0002
- Adam: β1=0.5, β2=0.999
- Batch size: 32-128
- Latent dim: 100-512

**Evaluation**:
- FID score for quality
- Visual inspection
- Diversity metrics

## Important Takeaways
1. GAN training requires careful loss monitoring
2. Image-to-image translation is powerful application
3. CycleGAN enables unpaired translation
4. GANs useful for augmentation and enhancement
5. Regular checkpointing essential
6. FID is standard evaluation metric

