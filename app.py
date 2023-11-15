import cv2
import numpy as np
import gradio as gr
from mbnet import load_model, detect_objects, get_box_dimensions, draw_labels, load_img
from yolov3 import load_image, load_yolo, detect_objects_yolo, get_box_dimensions_yolo, draw_labels_yolo


# Image Inference

def img_inf(img,model):
    if model=="MobileNet-SSD":
        model, classes, colors = load_model()
        image, height, width, channels = load_img(img)
        blob, outputs = detect_objects(image, model)
        boxes, class_ids = get_box_dimensions(outputs, height, width)
        image1 = draw_labels(boxes, colors, class_ids, classes, image)
        return cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    else:
        model, classes, colors, output_layers = load_yolo()
        image, height, width, channels = load_image(img)
        blob, outputs = detect_objects_yolo(image, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions_yolo(outputs, height, width)
        image=draw_labels_yolo(boxes, confs, colors, class_ids, classes, image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    

     
model_name = gr.Radio(["MobileNet-SSD", "YOLOv3"], value="YOLOv3", label="Model", info="choose your model")
inputs_image = gr.Image(type="filepath", label="Input Image")
outputs_image = [
    gr.Image(type="numpy", label="Output Image"),
]
interface_image = gr.Interface(
    fn=img_inf,
    inputs=[inputs_image,model_name],
    outputs=outputs_image,
    title="Image Detection",
    description="upload your photo and select one model and see the resuls!",
    examples=[["dog.jpg"]],
    cache_examples=False,
)

#Video Inference

def vid_inf(vid,model_type):
    if model_type=="MobileNet-SSD":
        cap = cv2.VideoCapture(vid)
        model, classes, colors = load_model()
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                height, width, channels = frame.shape
                blob, outputs = detect_objects(frame, model)
                boxes, class_ids = get_box_dimensions(outputs, height, width)
                frame=draw_labels(boxes, colors, class_ids, classes, frame)
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        cap = cv2.VideoCapture(vid)
        model, classes, colors, output_layers = load_yolo()
        while(cap.isOpened()):
            ret1,frame1 = cap.read()
            if ret1:
                height, width, channels = frame1.shape
                blob, outputs = detect_objects_yolo(frame1, model, output_layers)
                boxes, confs, class_ids = get_box_dimensions_yolo(outputs, height, width)
                frame=draw_labels_yolo(boxes, confs, colors, class_ids, classes, frame1)
                yield cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                


model_name = gr.Radio(["MobileNet-SSD", "YOLOv3"], value="YOLOv3", label="Model", info="choose your model")
inputs_video = gr.Video(sources=None, label="Input Video")
outputs_video = [
    gr.Image(type="numpy", label="Output Video"),
]
interface_video = gr.Interface(
    fn=vid_inf,
    inputs=[inputs_video,model_name],
    outputs=outputs_video,
    title="Video Detection",
    description="upload your video and select one model and see the resuls!",
    examples=[["video_1.mp4"]],
    cache_examples=False,
)

gr.TabbedInterface(
    [interface_image, interface_video],
    tab_names=['Image inference', 'Video inference'],
    title='Object Detection(MobileNet-SSDxYOLOv3)'
).queue().launch()