# simpleMHVAE

Simple one-off implementation of markovian hierarchical variational auto-encoders. I made these modules to understand diffusion a little better. Perhaps useful for you too. Licence is MIT, so do whatever you want. This is a pedagogic exercise and will not be optimal computationally, but rather focus simplicity.

The notebooks folder will contain examples on the MNIST dataset, FashionMNIST, and another non-image dataset.

The implementation is based in PyTorch and is largely based on Calvin Luo's "Understanding Diffusion Models: A Unified Perspective" (https://arxiv.org/pdf/2208.11970.pdf)

## Prerequisites

The implementation is based on vanilla PyTorch

If you have a GPU:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

If you have a CPU only:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```