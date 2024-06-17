import torch
import torchvision
import numpy as np
import torchvision.transforms.functional as F
import cv2

from models import load_freezed_model

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    """
    x1_tl, y1_tl, x1_br, y1_br = box1
    x2_tl, y2_tl, x2_br, y2_br = box2
    
    # Calculate the coordinates of the intersection rectangle
    x_left = max(x1_tl, x2_tl)
    y_top = max(y1_tl, y2_tl)
    x_right = min(x1_br, x2_br)
    y_bottom = min(y1_br, y2_br)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    box1_area = (x1_br - x1_tl) * (y1_br - y1_tl)
    box2_area = (x2_br - x2_tl) * (y2_br - y2_tl)

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

def calculate_distance(keypoint1, keypoint2):
    """
    Calculate Euclidean distance between two keypoints.
    """
    return np.sqrt((keypoint1[0] - keypoint2[0])**2 + (keypoint1[1] - keypoint2[1])**2)

def evaluate_keypoints(predictions, ground_truths, iou_threshold=0.7):
    """
    Evaluate keypoint predictions against ground truths.
    """
    num_predictions = len(predictions["boxes"])
    num_ground_truths = len(ground_truths["bbox"])


    keypoint_labels_allowed = ["ejection beginning", "mid upstroke", "maximum velocity", "mid deceleration point", "ejection end"]
    distances = []
    distances_by_type_of_keypoint = {}
    for l in keypoint_labels_allowed:
        distances_by_type_of_keypoint[l] = []

    # Initialize TP, FP, FN counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    if num_predictions == 0:
        return distances, distances_by_type_of_keypoint, 0, 0, num_ground_truths
    
    # IMPORTANT: the NMS built in FasterRCNN only eliminates overlapping boxes of the same label
    label = np.bincount(predictions["labels"]).argmax() # Returns the most common box label
    valid_boxes = predictions["labels"]==label # To mask the boxes that match the label
    # Sort the predicted bounding boxes by x coordinate
    x_coordinates = predictions["boxes"][valid_boxes][:, 0]
    sorted_indices = np.argsort(x_coordinates)

    num_predictions = len(predictions["boxes"][valid_boxes]) # Update num of predictions to be only the valid ones

    # Iterate through predicted bounding boxes
    for bbox_id in range(len(predictions["boxes"][valid_boxes][sorted_indices])):
        pred_box = predictions["boxes"][valid_boxes][sorted_indices][bbox_id]
        pred_keypoints = predictions["keypoints"][valid_boxes][sorted_indices][bbox_id]

        if predictions["scores"][valid_boxes][sorted_indices][bbox_id] < 0.3:
            num_predictions -= 1 # if predicted bbox does not pass the score threshold, it is not even a prediction
            continue
        # Check if any ground truth bounding box matches the predicted bounding box
        matched_gt_id = None
        for gt_bbox_id in range(len(ground_truths["bbox"])):
            gt_bbox = ground_truths["bbox"][gt_bbox_id]
            iou = calculate_iou(pred_box, gt_bbox)
            if iou >= iou_threshold:
                matched_gt_id = gt_bbox_id
                true_positives += 1
                break
        
        if matched_gt_id:
            # If a match is found, check keypoint accuracy
            gt_keypoints = ground_truths["kpts"][matched_gt_id]
            gt_keypoints = np.array(gt_keypoints)
            gt_keypoint_labels = ground_truths["kpts_labels"][matched_gt_id]
            # Filter keypoints
            mask = np.isin(gt_keypoint_labels, keypoint_labels_allowed)
            filtered_keypoints = gt_keypoints[mask]
            filtered_labels = gt_keypoint_labels[mask]
            gt_keypoints = filtered_keypoints.copy()

            correct_keypoints = 0
            for pred_kp, gt_kp, kp_label in zip(pred_keypoints, gt_keypoints, filtered_labels):
                distance = calculate_distance(pred_kp, gt_kp)
                distances.append(distance.item())
                distances_by_type_of_keypoint[kp_label].append(distance.item())

    
    # TODO: IT WOULD BE VERY USEFUL TO COMPUTE:
    # - True Positives: THE AMOUNT OF BBOXES CORRECTLY IDENTIFIED (THOSE WITH GOOD IoU) ,
    # - False positives: WHERE A BBOX WAS PREDICTED BUT NO REAL BBOX HAS A GOOD IOU, 
    # - True Negatives: -
    # - False Negatives: # of real BBOXES minus (-) # of True Positives
    false_positives = num_predictions - true_positives
    false_negatives = num_ground_truths - true_positives
    
    return distances, distances_by_type_of_keypoint, true_positives, false_positives, false_negatives

def evaluate(model: torch.nn.Module, device):

    tr = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean = [.5, .5, .5], std = [.5, .5, .5])
        ])

    # Iterate over the test filenames
    TEST_DATASET_LIST_TXT = "/home/shernandez/data/SpectralDopplerNetPreprocessed/filenames/doppler_test_filenames.txt"
    DATA_ROOT = "/home/shernandez/data/SpectralDopplerNetPreprocessed"


    img_list_from_file = []
    with open(TEST_DATASET_LIST_TXT) as f:
        img_list_from_file.extend(f.read().splitlines())

    errors = []
    errors_by_label = {
        "ejection beginning": [],
        "mid upstroke": [],
        "maximum velocity": [],
        "mid deceleration point": [],
        "ejection end": [],
    }
    precisions = []
    recalls = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
        
    for filename in img_list_from_file:
        print("Processing...", filename)
        img_path = DATA_ROOT+"/frames/"+filename
        annotations_path = img_path.replace(".png", ".npy").replace("/frames/", "/annotations/")
        metadata_path = img_path.replace(".png", ".npy").replace("/frames/", "/metadata/")

        if "Umbilical Artery" not in filename and "Uterine Artery" not in filename and "Middle Cerebral Artery" not in filename:
            continue

        img_original = cv2.imread(img_path)
        data = np.load(annotations_path, allow_pickle=True).item()
        metadata = np.load(metadata_path, allow_pickle=True).item()

        image = F.to_tensor(img_original)
        image = tr(image)
        image = image.unsqueeze(0)

        image = image.to(device)
        model.eval()
        output = model(image)

        predictions = output[0]
        predictions = {k: v.cpu().detach().numpy() for k, v in predictions.items()}

        distances, distance_by_label, tp, fp, fn = evaluate_keypoints(predictions, data)

        errors.extend(distances)
        for k in errors_by_label.keys():
            errors_by_label[k].extend(distance_by_label[k])
        if tp+fp != 0:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
        else:
            precision = 0
            recall = 0
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        precisions.append(precision)
        recalls.append(recall)


    print("Mean kpt error in pixels:", np.mean(errors))
    return errors_by_label, np.mean(errors)



if __name__ == "__main__":
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_gpu = True if device=="cuda" else False
    model = load_freezed_model("/home/shernandez/projects/DopplerNet/storage/logs/SpectralDopplerNet_IMPACT_Preprocessed_SERGI/KeypointRCNN/resnet_50/520192344/epoch_1_weights_SpectralDopplerNet_IMPACT_Preprocessed_SERGI_KeypointRCNN_best_kptsErr_0.pth", is_gpu)
    errors_by_label, mean_error = evaluate(model, device)