from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=False)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=False)

        # 초기화 범위: 차원에 무관하게 적절한 스케일 유지
        # Xavier/Glorot uniform 변형: sqrt(6 / (emb_dimension + emb_dimension))
        initrange = (6.0 / (2 * self.emb_dimension)) ** 0.5
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.uniform_(self.v_embeddings.weight.data, -initrange, initrange)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)

        # Handle both 1D and 2D tensors
        if neg_score.dim() == 1:
            neg_score = -F.logsigmoid(-neg_score)
        else:
            neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)
