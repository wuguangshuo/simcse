import torch
import torch.nn.functional as F

def SimCSE_loss(pred, tau=0.05):
    ids = torch.arange(0, pred.shape[0])
    y_true = ids + 1 - ids % 2 * 2
    similarities = F.cosine_similarity(pred.unsqueeze(1), pred.unsqueeze(0), dim=2)
    # 屏蔽对角矩阵，即自身相等的loss为0
    similarities = similarities - torch.eye(pred.shape[0]) * 1e12
    similarities = similarities / tau
    loss=F.cross_entropy(similarities, y_true)
    return torch.mean(loss)

pred = torch.tensor([[0.3, 0.2, 2.1, 3.1,1.5],
        [0.3, 0.2, 2.1, 3.1,1.6],
        [-1.79, -3, 2.11, 0.89,0.5],
        [-1.79, -3, 2.11, 0.89,0.3]])
res=SimCSE_loss(pred)
print(res)