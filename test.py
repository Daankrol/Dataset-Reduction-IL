import numpy as np
import torch
import torch.nn.functional as F


p = []
for c in range(3):
    p.append(np.array([]))
    print(p)





# k = 3
# aa = [0.3, 0.2, 0.5]
#
# bb = [
#     [0.2 - 0.0001, 0.0001, 0.8],
#     [0.4, 0.1, 0.5],
#     [np.nan, np.nan,np.nan] # empty neighbour
# ]
# # print(aa)
# aa = torch.from_numpy(np.expand_dims(aa, axis=1).repeat(k, axis=1))
# # bb = bb[0]
# bb = torch.from_numpy(np.array(bb))
# print(aa)
# print(bb, '\n=====')
# #
# divergence = (aa * torch.log(aa/bb))
# print(divergence)
# print(divergence.sum(dim=0))
# print(divergence.sum(dim=1).mean(dim=0), '\n====')
#
# divergence = F.kl_div(bb.log(), aa, None, None, 'sum')
# print(divergence)
