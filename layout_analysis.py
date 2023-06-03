import cv2
import os, glob
import time
import random
import numpy as np
import argparse
import onnxruntime as ort
labels= ['header', 'table', 'text']

def create_predictor(args, mode):
    model_dir = args.model
    # print(mode)
    model_file_path = model_dir + '/inference.onnx'
    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(
            model_file_path))
    providers = ['CPUExecutionProvider']
    print("--------------------CPU---------------------")
    sess = ort.InferenceSession(model_file_path, providers=providers)
    return sess, sess.get_inputs()[0], None, None

class LayoutAnalysis:

    def __init__(self, args, conf_thres=0.4, iou_thres=0.4):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            create_predictor(args, 'layout')
        # Get model info
        self.get_input_details()
        self.get_output_details()

    

    def get_input_details(self):
        model_inputs = self.predictor.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
    
    def get_output_details(self):
        model_outputs = self.predictor.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
    
    def prepare_input(self, input_img):
        self.img_height, self.img_width = input_img.shape[:2]

        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        # Scale input pixel values to 0 to 1
        input_img, ratio, dwdh = self.letterbox(input_img, auto=False)
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.expand_dims(input_img, 0)
        input_img = np.ascontiguousarray(input_img)
        input_img = input_img.astype(np.float32)
        input_img = input_img / 255.0
        input = {self.input_names[0]:input_img}
        
        return input, ratio, dwdh
    
    def process_output(self, outputs, ratio, dwdh ):
        boxes = []
        outlabels = []
        scores = []
        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            boxes.append(box)
            cls_id = int(cls_id)
            score = round(float(score),3)
            scores.append(score)
            outlabels.append(cls_id)
       
        return boxes, outlabels, scores
    
    def postprocess(self, ori_images, outputs, ratio, dwdh, img_idx):
        results = []
        boxes = []
        outlabels = []
        scores = []
        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            label = labels[int(cls_id)]
            score = round(float(score),3)
            result = {'label': label, 'score' : score, 'bbox': box}
            results.append(result)
            boxes.append(box)
            scores.append(score)
            outlabels.append(int(cls_id))
        print(outlabels)
        combined_img = self.draw_detections(ori_images, boxes, scores, outlabels)
        # cv2.imshow("Detected Objects", combined_img)
        # cv2.waitKey(0)
        out_folder = 'output/layout/'
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        
        outname = out_folder + str(img_idx) + str(boxes) + '.jpg' 
        cv2.imwrite(outname, combined_img)
        return results
    
    def draw_detections(self, image, boxes, scores, class_ids, mask_alpha=0.3):
        colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(labels)}
        mask_img = image.copy()
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        # Draw bounding boxes and labels of detections
        for box, score, class_id in zip(boxes, scores, class_ids):
            # print(class_id)
            name = labels[class_id]
            color = colors[name]

            x1, y1, x2, y2 = box

            # Draw rectangle
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

            # Draw fill rectangle in mask image
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

            # label = labels[int(class_id)]
            caption = f'{name} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(det_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)
            cv2.rectangle(mask_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)
            cv2.putText(det_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

            cv2.putText(mask_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)
    
    def obiect_detect(self,image):
        
        ori_images = image.copy()
        input, ratio, dwdh = self.prepare_input(ori_images)

        # ONNX inference
        outputs = self.predictor.run(self.output_names, input)[0]
        boxes, outlabels, scores = self.process_output(outputs, ratio, dwdh )
        return boxes, outlabels, scores
    
    def __call__(self,image, img_idx):
        ori_images = image.copy()
        starttime = time.time()
        input, ratio, dwdh = self.prepare_input(ori_images)

        # ONNX inference
        outputs = self.predictor.run(self.output_names, input)[0]
        results = self.postprocess(ori_images, outputs, ratio, dwdh, img_idx)
        time_layout = time.time() - starttime
        
        return results, time_layout



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', type=str, default='yolov7.onnx', help='model.onnx path(s)')
    parser.add_argument('--image', type=str, default='/images', help='image dir')  # file/folder, 0 for webcam
    args = parser.parse_args()
    
    # Initialize layout analysis model
    layout_analysis = LayoutAnalysis(args, conf_thres=0.35, iou_thres=0.65)
    labels= ['header', 'table', 'text']

    cv2.namedWindow("input", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
    image_src= cv2.imread(args.image)                    # Read image
    imS = cv2.resize(image_src, (960, 540))                # Resize image
    cv2.imshow("input", imS)                       # Show image
    cv2.waitKey(0) 
    boxes, scores, class_ids = layout_analysis.obiect_detect(image_src)

    combined_img = layout_analysis.draw_detections(image_src, boxes, class_ids, scores  )
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions 
    imO = cv2.resize(combined_img, (540, 960))                # Resize image
    cv2.imshow("output", imO)                       # Show image
    cv2.waitKey(0) 
