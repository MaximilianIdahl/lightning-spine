import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as ptl
from utils import CappedRELU, EmbeddingsDataset
from tqdm import tqdm

word2idx, vectors = np.load('embeddings/glove.42B.300d.npy', allow_pickle=True)
print(f'Loaded vectors of dim {vectors.shape}')

limit = 15000
idx2word = {v: k for k, v in word2idx.items()}
word2idx = {idx2word[i]: i for i in range(0, limit)}
vectors = vectors[:limit]
print(f'Limited vectors of dim {vectors.shape}')

# hparams TODO argparse, cuda switch
input_dim = vectors.shape[1]
hidden_dim = 1000
sparsity_perc = 0.9
noise = 0.4
lr = 1e-3
batch_size = 64


class SPINE(ptl.LightningModule):
    def __init__(self, input_dim, hidden_dim, sparsity_perc, lambda_asl=1.0, lambda_psl=1.0):
        super(SPINE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_perc = sparsity_perc
        self.lambda_asl = lambda_asl
        self.lambda_psl = lambda_psl
        self.reconstr_loss_fn = nn.functional.mse_loss
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                     CappedRELU(cap=1))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h), h

    def average_sparsity_loss(self, h):
        return (torch.mean(h, dim=0) - (1 - self.sparsity_perc)).clamp(min=0).pow(2).sum() / self.hidden_dim

    def partial_sparsity_loss(self, h):
        return torch.sum(h * (1 - h)) / (self.hidden_dim * h.shape[0])  # average over hidden vector and batch

    def training_step(self, batch, batch_num):
        x, y = batch
        out, h = self.forward(x)
        rl = self.reconstr_loss_fn(out, y)
        asl = self.average_sparsity_loss(h)
        psl = self.partial_sparsity_loss(h)
        loss = rl + self.lambda_asl * asl + self.lambda_psl * psl
        avg_num_active_dims = (h > 0).sum(1).float().mean()
        prog_dict = {'total_loss': loss, 'rl': rl, 'asl': asl, 'psl': psl, 'avg_num_active_dims': avg_num_active_dims}
        prog_dict = {k: f'{v.item():.6f}' for k, v in prog_dict.items()}
        return {'loss': loss, 'prog': prog_dict}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=lr, eps=1e-3)

    @ptl.data_loader
    def tng_dataloader(self):
        return DataLoader(EmbeddingsDataset(vectors, noise), pin_memory=True, batch_size=batch_size, shuffle=True)

    def get_sparse_vectors(self):
        test_dl = DataLoader(EmbeddingsDataset(vectors), pin_memory=True, batch_size=batch_size)
        batch_vecs = []
        with torch.no_grad():
            for x, y in tqdm(test_dl, desc='Creating embedding matrix'):
                _, sparse_vectors = self.forward(x.cuda())
                batch_vecs.append(sparse_vectors.cpu().detach())
        return torch.cat(batch_vecs).numpy()


model = SPINE(input_dim, hidden_dim, sparsity_perc)
trainer = ptl.Trainer(max_nb_epochs=10, gpus=[0])
trainer.fit(model)
sparse_vectors = model.get_sparse_vectors()
print(sparse_vectors.shape)
print('Dumping to disk...')
np.save('embeddings/spine.npy', [word2idx, sparse_vectors])
