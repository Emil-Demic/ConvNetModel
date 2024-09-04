import cv2
import numpy as np
import torch
import torch.nn.functional as F
from bresenham import bresenham


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


def drawPNG(vector_images, side=256, time_frac=None, skip_front=False):
    raster_image = np.ones((side, side), dtype=np.uint8)
    prevX, prevY = None, None
    begin_time = vector_images[0]['timestamp']
    start_time = vector_images[0]['timestamp']
    end_time = vector_images[-1]['timestamp']

    if time_frac:
        if skip_front:
            start_time = (end_time - start_time) * time_frac
        else:
            end_time -= (end_time - start_time) * time_frac

    for points in vector_images:
        time = points['timestamp'] - begin_time
        if not (start_time <= time <= end_time):
            continue

        x, y = map(float, points['coordinates'])
        x = int(x * side);
        y = int(y * side)
        pen_state = list(map(int, points['pen_state']))
        if not (prevX is None or prevY is None):
            cordList = list(bresenham(prevX, prevY, x, y))
            for cord in cordList:
                if (cord[0] > 0 and cord[1] > 0) and (cord[0] < side and cord[1] < side):
                    raster_image[cord[1], cord[0]] = 0
            if pen_state == [0, 1, 0]:
                prevX = x
                prevY = y
            elif pen_state == [1, 0, 0]:
                prevX = None
                prevY = None
            else:
                raise ValueError('pen_state not accounted for')
        else:
            prevX = x
            prevY = y
    # invert black and white pixels and dialate
    raster_image = (1 - cv2.dilate(1 - raster_image, np.ones((3, 3), np.uint8), iterations=1)) * 255
    return raster_image
