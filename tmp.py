import torch
t2 = torch.tensor([[1,2,3], [4,5,6], [7, 8, 9]])
t1 = torch.tensor([1,2,3])
print(t1)

# args.logit_weight = 5 ** -2.477043166563927 * 0.03
# args.emb_weight = 5 ** 1 * 1
# args.struc_weight = 5 ** 1 * 0.45
# args.gt_weight = 5 ** -0.08501119246436462 * 0.03

# wandb: 	emb_weight: 1
# wandb: 	gt_weight: -0.08501119246436462
# wandb: 	logit_weight: -2.477043166563927
# wandb: 	struc_weight: 1