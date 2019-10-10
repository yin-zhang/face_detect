import csv
import cv2
import numpy as np
import tensorflow as tf

class FaceDetector():
    r"""
    Class to use Google's Mediapipe FaceDetector pipeline from Python.
    Any any image size and aspect ratio supported.  Multiple
    faces are supported.

    Args:
        face_model: path to the face_detection_front.tflite
        anchors_path: path to the csv containing SSD anchors
    Ourput:
        (6,2) array of keypoints and bounding boxes
    Examples::
        >>> det = FaceDetector(path1, path2)
        >>> input_img = np.random.randint(0,255, 128*128*3).reshape(128,128,3)
        >>> list_keypoints, list_bbox = det(input_img)
    """

    def __init__(self, face_model, anchors_path):

        self.interp_face = tf.lite.Interpreter(face_model)
        self.interp_face.allocate_tensors()
        
        # reading the SSD anchors
        with open(anchors_path, "r") as csv_f:
            self.anchors = np.r_[
                [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]

        # reading tflite model paramteres
        output_details = self.interp_face.get_output_details()
        input_details = self.interp_face.get_input_details()
        print('face.output_details:', output_details)
        print('face.input_details:', input_details)

        self.in_idx = input_details[0]['index']
        self.out_reg_idx = output_details[0]['index']
        self.out_clf_idx = output_details[1]['index']
            
    @staticmethod
    def _im_normalize(img):
         return np.ascontiguousarray(
             2 * ((img / 255) - 0.5
        ).astype('float32'))
       
    @staticmethod
    def _sigm(x):
        return 1 / (1 + np.exp(-x) )
    
    @staticmethod
    def _pad1(x):
        return np.pad(x, ((0,0),(0,1)), constant_values=1, mode='constant')
    
    # compute IoU similarity
    def calc_iou(self, box0, box1):
        xmin = max(box0[0]-box0[2]/2, box1[0]-box1[2]/2)
        ymin = max(box0[1]-box0[3]/2, box1[1]-box1[3]/2)
        xmax = min(box0[0]+box0[2]/2, box1[0]+box1[2]/2)
        ymax = min(box0[1]+box0[3]/2, box1[1]+box1[3]/2)
        area0 = box0[2] * box0[3]
        area1 = box1[2] * box1[3]
        i = max(0, xmax - xmin) * max(0, ymax - ymin)
        u = area0 + area1 - i
        return i / u

    def non_maximum_suppression(self, reg, anchors, probs, weighted=True):

        sorted_idxs = probs.argsort()[::-1].tolist()

        abs_bbox = np.copy(reg)
        
        # turn relative bbox/keyp into absolute bbox/keyp
        for idx in sorted_idxs:
            center = anchors[idx,:2] * 128
            for j in range(2):
                abs_bbox[idx,j] = center[j] + abs_bbox[idx,j]
                abs_bbox[idx,(j+4)::2] = center[j] + abs_bbox[idx,(j+4)::2]

        remain_idxs = sorted_idxs
        output_bbox = abs_bbox[0:0,:]

        while len(remain_idxs) > 0:
            # separate remain_idxs into candids and remain
            candids = []
            remains = []
            idx0 = remain_idxs[0]
            for idx in remain_idxs:
                iou = self.calc_iou(abs_bbox[idx0,:], abs_bbox[idx,:])
                if iou >= 0.3:
                    candids.append(idx)
                else:
                    remains.append(idx)

            # compute weighted bbox/keyp
            if not weighted:
                weighted_bbox = abs_bbox[idx0,:]
            else:
                weighted_bbox = abs_bbox[0,:] * 0
                weight_sum = 0
                for idx in candids:
                    w = probs[idx]
                    weight_sum = weight_sum + w
                    weighted_bbox = weighted_bbox + w * abs_bbox[idx,:]
                weighted_bbox = weighted_bbox / weight_sum

            # add a new instance
            output_bbox = np.concatenate((output_bbox, weighted_bbox.reshape(1,-1)), axis=0)
            
            remain_idxs = remains

        return output_bbox

    def detect_face(self, img_norm):
        assert -1 <= img_norm.min() and img_norm.max() <= 1,\
        "img_norm should be in range [-1, 1]"
        assert img_norm.shape == (128, 128, 3),\
        "img_norm shape must be (128, 128, 3)"

        # predict face location and 6 initial landmarks
        self.interp_face.set_tensor(self.in_idx, img_norm[None])
        self.interp_face.invoke()

        out_reg = self.interp_face.get_tensor(self.out_reg_idx)[0]
        out_clf = self.interp_face.get_tensor(self.out_clf_idx)[0,:,0]
        out_prb = self._sigm(out_clf)

        # finding the best prediction
        detection_mask = out_prb > 0.75
        filtered_detect = out_reg[detection_mask]
        filtered_anchors = self.anchors[detection_mask]
        filtered_probs = out_prb[detection_mask]

        if filtered_detect.shape[0] == 0:
            print("No faces found")
            return None, None

        # perform non-maximum suppression
        candidate_detect = self.non_maximum_suppression(filtered_detect, filtered_anchors, filtered_probs)

        bboxs = []
        keyps = []

        for idx in range(candidate_detect.shape[0]):

            # bounding box center offsets, width and height
            bbox = candidate_detect[idx, :4]

            # 6 initial keypoints
            keyp = candidate_detect[idx,4:].reshape(-1,2)
        
            bboxs.append(bbox)
            keyps.append(keyp)
            
        return bboxs, keyps

    def preprocess_img(self, img):
        # fit the image into a 128x128 square
        shape = np.r_[img.shape]
        pad = (shape.max() - shape[:2]).astype('uint32') // 2
        img_pad = np.pad(
            img,
            ((pad[0],pad[0]), (pad[1],pad[1]), (0,0)),
            mode='constant')
        img_small = cv2.resize(img_pad, (128, 128))
        img_small = np.ascontiguousarray(img_small)
        
        img_norm = self._im_normalize(img_small)
        return img_pad, img_norm, pad


    def __call__(self, img):
        img_pad, img_norm, pad = self.preprocess_img(img)
        
        bboxs, keyps = self.detect_face(img_norm)
        if bboxs is None:
            return None, None

        scale = max(img.shape) / 128

        list_keyp = []
        list_bbox = []
        for i in range(len(bboxs)):

            bbox = bboxs[i]
            keyp = keyps[i]

            bbox *= scale
            keyp *= scale
            bbox[:2] -= pad[::-1]
            keyp -= pad[::-1]

            list_keyp.append(keyp)
            list_bbox.append(bbox)
            
        return list_keyp, list_bbox
