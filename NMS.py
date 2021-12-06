# TODO: given bounding boxes and corresponding scores, perform non max suppression
def nms(bounding_boxes, confidence_score, threshold=0.05):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    sorted_index = np.argsort(confidence_score)[::-1]
    for i in range(len(sorted_index)):
        if confidence_score[i] < threshold:
            sorted_index = sorted_index[:i]

    indices = set()

    if len(sorted_index) == 0:
        return indices

    while len(sorted_index) > 0:
        box1 = bounding_boxes[sorted_index[0], :]
        indices.add(sorted_index[0])

        if len(sorted_index) == 1:
            break
        sorted_index = sorted_index[1:]

        keep_index = []
        for i in range(len(sorted_index)):
            index = sorted_index[i]
            box2 = bounding_boxes[index]
            IoU = iou(box1, box2)
            if IoU <= 0.3:
                keep_index.append(i)
        sorted_index = sorted_index[keep_index]
    return indices


# TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-8)

    return iou
