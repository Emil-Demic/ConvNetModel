import torch
import torch.nn.functional as F


def calculate_accuracy_alt(query_feature_all, image_feature_all):
    query_feature_all = torch.tensor(query_feature_all)
    image_feature_all = torch.tensor(image_feature_all)

    rank = torch.zeros(len(query_feature_all))
    for idx, query_feature in enumerate(query_feature_all):
        distance = F.pairwise_distance(query_feature.unsqueeze(0), image_feature_all)
        target_distance = F.pairwise_distance(
            query_feature.unsqueeze(0), image_feature_all[idx].unsqueeze(0))
        rank[idx] = distance.le(target_distance).sum()

    rank1 = rank.le(1).sum().numpy()
    rank5 = rank.le(5).sum().numpy()
    rank10 = rank.le(10).sum().numpy()
    rankM = rank.mean().numpy()

    return rank1, rank5, rank10, rankM
