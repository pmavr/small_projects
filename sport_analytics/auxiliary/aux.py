import cv2
import numpy as np
from auxiliary import ColorClusters as cc,\
    ObjectDetector as od, \
    HoughLines as hl


def show_image(img, msg=''):
    cv2.imshow(msg, img)
    while 1:
        k = cv2.waitKey(0)
        if k == 27:
            break
    cv2.destroyWindow(msg)


def remove_white_dots(image, iterations=1):
    # do connected components processing
    for j in range(iterations):
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, None, None, None, 8, cv2.CV_32S)
        # get CC_STAT_AREA component as stats[label, COLUMN]
        areas = stats[1:, cv2.CC_STAT_AREA]

        result = np.zeros((labels.shape), np.uint8)

        for i in range(0, nlabels - 1):
            if areas[i] >= 100:  # keep
                result[labels == i + 1] = 255

        image = result
        image = cv2.bitwise_not(image, image)

    return result


def object_detector_pipeline(image, yolo, team_predictor):
    frame_o = np.copy(image)
    img_for_object_detector = od.image_preprocess(frame_o)

    output = yolo.predict(img_for_object_detector)
    objects = yolo.extract_objects(output)

    refined_objects = yolo.merge_overlapping_boxes(objects)

    final_objects = yolo.kmeans_determine_team(refined_objects, team_predictor)
    return  final_objects


def court_detector_pipeline(image):
    frame_c = np.copy(image)
    img_c = hl.image_preprocess(frame_c)
    lines, img_with_hough_lines = hl.houghLines(img_c, frame_c)

    hor_lines = []
    ver_lines = []

    if lines is not None:
        for line in lines:
            rho, theta = line
            if hl.is_horizontal(theta):
                hor_lines.append(line)
            elif hl.is_vertical(theta):
                ver_lines.append(line)

    ref_hor_lines = hl.refine_lines(hor_lines, rtol=.125)
    ref_ver_lines = hl.refine_lines(ver_lines, rtol=.125)

    lines = []
    for line in ref_hor_lines:
        lines.append(line)
    for line in ref_ver_lines:
        lines.append(line)

    intersection_points = hl.get_intersection_points(lines)
    return [p for p in intersection_points if p is not None and p[0] >= 0 and p[1] >= 0]