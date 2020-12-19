import cv2
from auxiliary import ColorClusters as cc, ObjectDetector as od
from auxiliary.aux import object_detector_pipeline, court_detector_pipeline


# input_file = "../clips/france_belgium.mp4"
# input_file = "../clips/chelsea_manchester.mp4"
# input_file = "../clips/aris_aek.mp4"
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

# Uncomment in order to output to video
# writer = cv2.VideoWriter('../clips/video_with_predictions.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15.0, (1280, 720), True)

success, frame = vs.read()

while success:

    frame_resized = cv2.resize(frame, (1280, 720))

    points = court_detector_pipeline(frame_resized)

    identified_objects = object_detector_pipeline(frame_resized, yolo, team_predictor)

    frame_with_boxes = yolo.draw_bounding_boxes(frame_resized, identified_objects)

    for p in points:
        if p is not None:
            cv2.line(frame_with_boxes, (int(p[0]), int(p[1])), (int(p[0]), int(p[1])), (255, 255, 0), 10)

    cv2.imshow('Match Detection', frame_with_boxes)
    # writer.write(frame_with_boxes)
    # video play - pause - quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1)

    success, frame = vs.read()

print("[INFO] cleaning up...")
vs.release()
# writer.release()
cv2.destroyAllWindows()
