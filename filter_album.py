
import face_recognition
import os
import torch
import torchvision.transforms as transforms
import cv2
from pathlib import Path

# from basiccode.ssd_resnet import get_all_bbox


def extend_bbox_normalized(bbox, height_percent, width_percent):
    """
    Extend the normalized bounding box by a given percentage of height and width.

    Parameters:
    - bbox (tuple): Normalized bounding box coordinates in the format (x_min, y_min, x_max, y_max).
    - height_percent (float): Percentage to extend the height of the bounding box.
    - width_percent (float): Percentage to extend the width of the bounding box.

    Returns:
    - tuple: Extended normalized bounding box coordinates in the format (x_min, y_min, x_max, y_max).
    """
    x_min, y_min, x_max, y_max = bbox

    # Calculate height and width adjustments
    height_delta = (y_max - y_min) * (height_percent / 100)
    width_delta = (x_max - x_min) * (width_percent / 100)

    # Extend the bounding box while ensuring it does not exceed the image boundaries
    extended_bbox = (
        max(0, x_min - width_delta),
        max(0, y_min - height_delta),
        min(1, x_max + width_delta),
        min(1, y_max + height_delta)
    )

    return extended_bbox


def get_all_bbox(image, results, height_extend=10, width_extend=0):
    all_bboxes = []
    for image_idx in range(len(results)):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get the original height and width of the image to resize the ...
        # ... bounding boxes to the original size size of the image
        orig_h, orig_w = image.shape[0], image.shape[1]
        # get the bounding boxes, classes, and confidence scores
        bboxes, classes, confidences = results[image_idx]
        print(len(bboxes))
        for idx in range(len(bboxes)):
            # get the bounding box coordinates in xyxy format
            x1, y1, x2, y2 = bboxes[idx]
            bbox_norm = bboxes[idx]
            x1, y1, x2, y2 = extend_bbox_normalized(bbox_norm,height_extend,width_extend)
            # resize the bounding boxes from the normalized to 300 pixels
            x1, y1 = int(x1*300), int(y1*300)
            x2, y2 = int(x2*300), int(y2*300)
            # resizing again to match the original dimensions of the image
            x1, y1 = int((x1/300)*orig_w), int((y1/300)*orig_h)
            x2, y2 = int((x2/300)*orig_w), int((y2/300)*orig_h)
            all_bboxes.append([x1, y1, x2, y2])
        
        return all_bboxes


def percentage_of_bbox_within(bbox1, bbox2):
    """
    Calculate the percentage of bbox1 that is within bbox2.

    Parameters:
    - bbox1 (tuple): Bounding box coordinates in the format (x1, y1, x2, y2).
    - bbox2 (tuple): Bounding box coordinates in the format (x1, y1, x2, y2).

    Returns:
    - float: Percentage of bbox1 that is within bbox2.
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    # Calculate coordinates of intersection
    x_intersection = max(x1, x3)
    y_intersection = max(y1, y3)
    x_intersection_max = min(x2, x4)
    y_intersection_max = min(y2, y4)

    # Calculate area of intersection
    intersection_width = max(0, x_intersection_max - x_intersection)
    intersection_height = max(0, y_intersection_max - y_intersection)
    intersection_area = intersection_width * intersection_height

    # Calculate area of bbox1
    bbox1_area = (x2 - x1) * (y2 - y1)

    # Calculate percentage of bbox1 within bbox2
    percentage_within = (intersection_area / bbox1_area) * 100 if bbox1_area > 0 else 0.0

    return percentage_within


def get_face_encoding_for_ref(ref_image_path):
    ref_image = face_recognition.load_image_file(ref_image_path)
    ref_face_encoding = face_recognition.face_encodings(ref_image)[0]
    return ref_face_encoding


class CleanPersonData():
    def __init__(self,root_dir, ref_path, output_path):
        self.root_dir = root_dir
        self.ref_path = ref_path
        self.ref_encoding = None
        self.ssd_model = None
        self.utils = None
        self.device = None
        self.initialize_models()
        self.output_path = output_path
        self.initialize_ref_encoding()
        
    def initialize_models(self):
        self.ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ssd_model.to(device)
        self.device = device
        self.ssd_model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

    def initialize_ref_encoding(self):
        ref_image = face_recognition.load_image_file(self.ref_path)
        self.ref_encoding = face_recognition.face_encodings(ref_image)[0]

    def get_person_bbox(self,image_path):
        image = cv2.imread(image_path)
        # keep the original height and width for resizing of bounding boxes
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply the image transforms
        transformed_image = self.transform(image)
        # convert to torch tensor
        tensor = torch.tensor(transformed_image, dtype=torch.float32)
        # add a batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)

        # get the detection results
        with torch.no_grad():
            detections = self.ssd_model(tensor)
        # the PyTorch SSD `utils` help get the detection for each input if...
        # ... there are more than one image in a batch
        # for us there is only one image per batch
        results_per_input = self.utils.decode_results(detections)
        # get all the results where detection threshold scores are >= 0.45
        # SSD `utils` help us here as well
        best_results_per_input = [self.utils.pick_best(results, 0.45) for results in results_per_input]
        allbboxes = get_all_bbox(image,best_results_per_input)
        return allbboxes
    
    def get_reference_face_locations(self,image_path):
        unknown_image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        for i,face_encoding in enumerate(face_encodings):
            is_match = face_recognition.compare_faces([self.ref_encoding], face_encoding)[0]
            if is_match:
            
                top, right, bottom, left = face_locations[i]
                x1,y1,x2,y2 = left,top,right,bottom
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                return [x1,y1,x2,y2]
        return None

    def get_output_path(self,img_path):
        ref_name = Path(self.ref_path).stem
        img_stem = Path(img_path).stem
        return os.path.join(self.output_path,f'{img_stem}-ref-{ref_name}.jpg')
        
    def process_file(self,img_path):
        person_bboxes = self.get_person_bbox(img_path)
        face_bbox = self.get_reference_face_locations(img_path)
        for bbox in person_bboxes:
            
            bbox_within = percentage_of_bbox_within(face_bbox,bbox)
            if bbox_within > 80:
                img =  cv2.imread(img_path)
                x1,y1,x2,y2 = bbox
                cropped_image = img[y1:y2, x1:x2]
                cv2.imwrite(self.get_output_path(img_path),cropped_image)

    def gen_person_data(self,ref_path, all_images_dir):
        all_image_names = [path for path in os.listdir(all_images_dir) if path.lower().endswith(".jpg")]
        ref_encoding = get_face_encoding_for_ref(ref_path)
        for image_name in all_image_names:
            image_path = os.path.join(all_images_dir,image_name)
            self.process_file(image_path)


def main():
    root_dir = "/root/workspace/data/IsolateGroupDataset/data/"
    ref_path = "/root/workspace/data/IsolateGroupDataset/data/IMG_0705.jpg"
    new_image_path = "/root/workspace/data/IsolateGroupDataset/data/IMG_2531.jpg"
    output_path = "/root/workspace/data/IsolateGroupDataset/output"
    cleaner = CleanPersonData(root_dir,ref_path, output_path)
    # cleaner.process_file(new_image_path)
    cleaner.gen_person_data(ref_path,root_dir)


if __name__ == "__main__":
    main()
