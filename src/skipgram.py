import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.u_embeddings = nn.Embedding(vocab_size, emb_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, emb_dim, sparse=True)

        self.u_embeddings.weight.data.uniform_(-0.5 / emb_dim, 0.5 / emb_dim)
        self.v_embeddings.weight.data.uniform_(-0.5 / emb_dim, 0.5 / emb_dim)

    def forward(self, u_idxs, v_idxs):
        u_emb = self.u_embeddings(u_idxs)
        v_emb = self.v_embeddings(v_idxs)
        score = torch.mul(u_emb, v_emb).sum(dim=1)
        return score
