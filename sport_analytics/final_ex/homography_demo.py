import cv2, \
    numpy as np
from sport_analytics.auxiliary import ColorClusters as cc, HoughLines as hl, ObjectDetector as od
from sport_analytics.auxiliary.aux import object_detector_pipeline, court_detector_pipeline, show_image


input_file = "../clips/belgium_japan.mp4"

training_frames = 2
yolo = od.ObjectDetector()

vs = cv2.VideoCapture(input_file)

boxes = []
idx = 0
for j in range(training_frames):
    success, frame = vs.read()
    frame = cv2.resize(frame, (1280, 720))

    img = od.image_preprocess(frame)
    output = yolo.predict(img)
    objects = yolo.extract_objects(output)

    for (b, _, _, _) in objects:
        box = yolo.to_image(b)
        if box.shape[0] > box.shape[1]:
            boxes.append(box)

team_predictor = cc.kmeans_train_clustering(boxes, n_clusters=3)
vs.release()

frame = cv2.imread('../clips/frame4.jpg')
frame_resized = cv2.resize(frame, (1280, 720))

points = court_detector_pipeline(frame_resized)

identified_objects = object_detector_pipeline(frame_resized, yolo, team_predictor)

frame_with_boxes = yolo.draw_bounding_boxes(frame_resized, identified_objects)

court_intersection_points = hl.get_court_intersection_points()

court_image = cv2.imread('../clips/court.jpg')

src = np.array([[914., 0.], [1091., 0.], [914., 557.7078881563991], [1091., 129.50011366219587]], np.float32)

dst = np.array([[459.0645161112684, 143.2217852219327], [765.7319784379197, 121.76624489962595],
                [1110.6788355551557, 433.11638665653606], [899.7333270184508, 162.6960601898524]], np.float32)

homography_matrix = cv2.getPerspectiveTransform(src, dst)

im_out = cv2.warpPerspective(court_image, homography_matrix, (frame_with_boxes.shape[1], frame_with_boxes.shape[0]))

im_out = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(im_out, 10, 255, cv2.THRESH_BINARY)
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

final_image = hl.blend_images(frame_with_boxes, mask)

show_image(final_image)
