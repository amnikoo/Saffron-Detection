import cv2
import json
import numpy as np
import os
import time
import glob
from matplotlib import pyplot as plt
import csv

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes


def sind(x):
    return np.sin(x / 180*np.pi)
    
def cosd(x):
    return np.cos(x / 180*np.pi)


def draw_line_segment(image, center, angle, color, length=30, thickness=5):
    x1 = center[0] - cosd(angle) * length / 2
    x2 = center[0] + cosd(angle) * length / 2
    y1 = center[1] - sind(angle) * length / 2
    y2 = center[1] + sind(angle) * length / 2

    cv2.line(image, (int(x1 + .5), int(y1 + .5)), (int(x2 + .5), int(y2 + .5)), color, thickness)

def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    phi = 0
    weighted_bifpn = False
    model_path = 'csv_479_0.1197.h5'
    saf = '0478'
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    # coco classes
    # classes = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15',16:'16',17:'17',18:'18',19:'19',20:'20',21:'21',22:'22',23:'23',24:'24',25:'25',26:'26',27:'27',28:'28',29:'29',30:'30',31:'31',32:'32',33:'33',34:'34',35:'35'}
    classes = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10',11:'11',12:'12',13:'13',14:'14',15:'15',16:'16',17:'17'}
    num_classes = 18
    score_threshold = 0.05
    colors =  [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    _, model = efficientdet(phi=phi,
                            weighted_bifpn=weighted_bifpn,
                            num_classes=num_classes,
                            score_threshold=score_threshold)

    model.load_weights(model_path, by_name=True)
    flag = 0
    for image_path in glob.glob('/content/drive/My Drive/Test/*.jpg'):
        flag+=1
        image = cv2.imread(image_path)
   
        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
     
        # print(boxes)
     
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
   
        boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)
    

        # select indices which have a score above the threshold
        indices = np.where(scores[:] > score_threshold)[0]

        print(labels)
        # select those detections
        boxes = boxes[indices]
        labels = labels[indices]
        counter = 0
        max = 0
        for i in range (len(boxes)):
          for j in range (len(boxes)):
            IOU = get_iou(boxes[i],boxes[j])
            if IOU > 0 and j!=i:
              if scores[i]>scores[j]:
                boxes[j] = [0,0,0,0]
              else:
                boxes[i] = [0,0,0,0]
        lines = []
        for i in range (len(boxes)):
          w = boxes[i][2] - boxes[i][0]
          h = boxes[i][3] - boxes[i][1]
          if(boxes[i][0] != 0):
            boxes[i][0] =  boxes[i][0] + (1/2)*w
            boxes[i][1] =  boxes[i][1] + (1/2)*h
            lines.append([boxes[i][0],boxes[i][1],labels[i]*10])
            boxes[i][2] =  boxes[i][2] - (1/3)*w
            boxes[i][3] =  boxes[i][3] - (1/3)*h
        print(scores)
        
        with open('/content/drive/MyDrive/test'+str(flag) +'.csv', mode='w') as employee_file:
          employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
       
          for i in range(len(lines)):
            
            employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
            employee_writer.writerow([lines[i][0],lines[i][1],lines[i][2],scores[i]])







        draw = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        for i in range(len(lines)):
          draw_line_segment(draw, lines[i][:2], lines[i][2], (0, 0, 255))


        cv2.imwrite("SelfSupervised-Test-18Label-3143-500 epoch-High Thickness-dot-30*30-lines-TH 0.05-"+str(flag)+".jpg",draw)


        




if __name__ == '__main__':
    main()
