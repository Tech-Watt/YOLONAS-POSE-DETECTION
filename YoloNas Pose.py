import super_gradients
from super_gradients.training import models
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization
import cv2
import torch

video = r'C:\Users\Admin\Desktop\data\neutral/10.mp4'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = super_gradients.training.models.get("yolo_nas_pose_l",
                            pretrained_weights="coco_pose").to(device)


cap = cv2.VideoCapture(video)

while True:
    rt,video = cap.read()
    video = cv2.resize(video,(700,500))
    model_predictions  = model.predict(video, conf=0.5)


    prediction = model_predictions[0].prediction
    bboxes = prediction.bboxes_xyxy
    poses  = prediction.poses
    scores = prediction.scores
    edge_links = prediction.edge_links
    edge_colors  = prediction.edge_colors
    keypoint_colors = prediction.keypoint_colors


    video = keleton_image = PoseVisualization.draw_poses(
        image=video,
        poses = poses,
        boxes= bboxes,
        scores =  scores,
        is_crowd=None,
        edge_links= edge_links,
        edge_colors= edge_colors,
        keypoint_colors= keypoint_colors,
        joint_thickness=2,
        box_thickness=2,
        keypoint_radius=5
    )



    cv2.imshow('frame', video)
    cv2.waitKey(1)