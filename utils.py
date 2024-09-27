import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from bresenham import bresenham
import scipy.spatial.distance as ssd

from config import args

def compute_view_specific_distance(sketch_feats, image_feats):
    return ssd.cdist(sketch_feats, image_feats, 'sqeuclidean')


def outputHtml(sketchindex, indexList, file_names):
    imageNameList = shuffleListTest
    sketchPath = os.path.join("fscoco", "raster_sketches")
    imgPath = os.path.join("fscoco", "images")

    tmpLine = "<tr>"

    tmpLine += "<td><image src='%s' width=256 /></td>" % (
        os.path.join(sketchPath, str(shuffleListTest[sketchindex]).zfill(12) + ".jpg"))
    for i in indexList:
        if i != sketchindex:
            tmpLine += "<td><image src='%s' width=256 /></td>" % (
                os.path.join(imgPath, str(imageNameList[i]).zfill(12) + ".jpg"))
        else:
            tmpLine += "<td ><image src='%s' width=256   style='border:solid 2px red' /></td>" % (
                os.path.join(imgPath, str(imageNameList[i]).zfill(12) + ".jpg"))

    return tmpLine + "</tr>"


def calculate_accuracy(dist, file_names):
    top1 = 0
    top5 = 0
    top10 = 0
    top20 = 0
    tmpLine = ""
    for i in range(dist.shape[0]):
        rank = dist[i].argsort()
        if rank[0] == i:
            top1 = top1 + 1
        if i in rank[:5]:
            top5 = top5 + 1
        if i in rank[:10]:
            top10 = top10 + 1
        if i in rank[:20]:
            top20 = top20 + 1
        tmpLine += outputHtml(i, rank[:10], file_names) + "\n"
    # num = dist.shape[0]
    # print(' top1: ' + str(top1 / float(num)))
    # print(' top5: ' + str(top5 / float(num)))
    # print('top10: ' + str(top10 / float(num)))

    htmlContent = """
       <html>
       <head></head>
       <body>
       <table>%s</table>
       </body>
       </html>""" % (tmpLine)
    with open(r"result.html", 'w+') as f:
        f.write(htmlContent)
    return top1, top5, top10


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


def drawPNG(vector_images, side=256, time_frac=None, skip_front=False, add_stroke=False):
    if args.colormap:
        raster_image = np.ones((side, side), dtype=np.float32)
    else:
        raster_image = np.ones((side, side), dtype=np.uint8)
    prevX, prevY = None, None
    begin_time = vector_images[0]['timestamp']
    start_time = vector_images[0]['timestamp']
    end_time = vector_images[-1]['timestamp']
    full_time = end_time - begin_time

    if time_frac:
        if skip_front:
            start_time = (end_time - start_time) * time_frac
        else:
            end_time -= (end_time - start_time) * time_frac

    if add_stroke:
        noise_points = []
        for _ in range(3):
            idx = random.randint(0, len(vector_images) - 1)
            noise_points.append(vector_images[idx])

        noise_points[0]['pen_state'] = [1, 0, 0]
        noise_points[-1]['pen_state'] = [1, 0, 0]

        vector_images = vector_images + noise_points

    for points in vector_images:
        time = points['timestamp'] - begin_time
        if not (start_time <= time <= end_time):
            continue

        x, y = map(float, points['coordinates'])
        x = int(x * side)
        y = int(y * side)
        pen_state = list(map(int, points['pen_state']))
        if not (prevX is None or prevY is None):
            cordList = list(bresenham(prevX, prevY, x, y))
            for cord in cordList:
                if (cord[0] > 0 and cord[1] > 0) and (cord[0] < side and cord[1] < side):
                    if args.colormap:
                        raster_image[cord[1], cord[0]] = time / full_time
                    else:
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

    if args.colormap:
        raster_image = raster_image.astype(np.uint8)
        mask = raster_image == 255
        raster_image = cv2.applyColorMap(raster_image, cv2.COLORMAP_TURBO)
        raster_image[mask] = 255

    # cv2.imshow('raster_image', raster_image)
    # cv2.waitKey(0)

    return raster_image
