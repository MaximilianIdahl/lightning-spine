import torch


class CappedRELU(torch.nn.Module):
    def __init__(self, cap=1):
        super().__init__()
        self.cap = cap

    def forward(self, x):
        return x.clamp(min=0, max=self.cap)


class EmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, vectors, noise=None):
        self.vectors = torch.tensor(vectors)
        self.noise = noise
        self.mean_val_per_dim = self.vectors.mean(0)
        self.std_per_dim = self.vectors.std(0)

    def __len__(self):
        return self.vectors.shape[0]

    def __getitem__(self, idx):
        y = self.vectors[idx]
        if self.noise:
            x = y + self.noise * torch.normal(mean=self.mean_val_per_dim, std=self.std_per_dim)
        else:
            x = y
        return x, y
