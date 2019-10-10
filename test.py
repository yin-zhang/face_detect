import numpy as np
import cv2
from face_detector import FaceDetector

det = FaceDetector('models/face_detection_front.tflite',
                   'data/anchors.csv')
in_bgr = cv2.imread('data/test_img.jpg')
in_rgb = in_bgr[:,:,::-1]
list_keypoints, list_bbox = det(in_rgb)


print('bbox:', list_bbox)
print('keyp:', list_keypoints)

out_img = np.copy(in_bgr)

# point size
ps = int(np.ceil(min(out_img.shape[0], out_img.shape[1]) / 256))

if list_keypoints is not None:
    for idx in range(len(list_keypoints)):
        keypoints = list_keypoints[idx]
        bbox = list_bbox[idx]
        x0 = int(np.round(bbox[0]-bbox[2]/2))
        y0 = int(np.round(bbox[1]-bbox[3]/2))
        x1 = int(np.round(bbox[0]+bbox[2]/2))
        y1 = int(np.round(bbox[1]+bbox[3]/2))
        cv2.rectangle(out_img, (x0,y0), (x1,y1), (0,0,255), ps)
        for i in range(keypoints.shape[0]):
            p = (int(keypoints[i,0]+0.5),int(keypoints[i,1]+0.5))
            cv2.circle(out_img, p, ps, (255,0,0), ps)
    cv2.imwrite('out.jpg', out_img)
