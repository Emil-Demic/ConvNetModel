import os
import torch
import torch.nn.functional as F
import scipy.spatial.distance as ssd


def compute_view_specific_distance(sketch_feats, image_feats):
    return ssd.cdist(sketch_feats, image_feats, 'sqeuclidean')


def outputHtml(sketchindex, indexList, file_names):
    imageNameList = file_names
    sketchPath = os.path.join("fscoco", "raster_sketches")
    imgPath = os.path.join("fscoco", "images")

    tmpLine = "<tr>"

    tmpLine += "<td><image src='%s' width=256 /></td>" % (
        os.path.join(sketchPath, str(file_names[sketchindex])))
    for i in indexList:
        if i != sketchindex:
            tmpLine += "<td><image src='%s' width=256 /></td>" % (
                os.path.join(imgPath, str(imageNameList[i])))
        else:
            tmpLine += "<td ><image src='%s' width=256   style='border:solid 2px red' /></td>" % (
                os.path.join(imgPath, str(imageNameList[i])))

    return tmpLine + "</tr>"


def calculate_accuracy(dist, file_names):
    top1 = 0
    top5 = 0
    top10 = 0
    tmpLine = ""
    for i in range(dist.shape[0]):
        rank = dist[i].argsort()
        if rank[0] == i:
            top1 = top1 + 1
        if i in rank[:5]:
            top5 = top5 + 1
        if i in rank[:10]:
            top10 = top10 + 1
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
