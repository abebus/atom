import requests
import cv2
import torch
from time import sleep
from PIL import Image
from typing import Union, Dict, List
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection, AutoImageProcessor

extractor = AutoFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

# my_token = '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'
# API_URL = "https://api-inference.huggingface.co/models/hustvl/yolos-tiny"
# headers = {"Authorization": f"Bearer {my_token}"}
stream_url = 'https://flussonic2.powernet.com.ru:8081/user83831/tracks-v1/mono.m3u8?token=dont-panic-and-carry-a-towel'

# def query(frame: cv2.Mat) -> Union[Dict,List]:
#     data = cv2.imencode('.jpg', frame)[1].tobytes()
#     response = requests.post(API_URL, headers=headers, data=data)
#     return response.json()

# def get_output(filename):
#     output = query(filename)
#     if isinstance(output, dict) and 'error' in output.keys():
#         sleep(1)
#         get_output(filename)
#     else:
#         return output


class ContextVideoCapture(cv2.VideoCapture): # я люблю так делать потому что я крутой
    def __enter__(self, *args):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        cv2.destroyAllWindows()


with ContextVideoCapture(stream_url) as cap:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        image = Image.fromarray(frame)

        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])

        results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            xmin, ymin, xmax, ymax = [int(i) for i in box.tolist()]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.config.id2label[label.item()]} Detected', (xmax + 10, ymax), 0, 0.3, (0, 255, 0))
            print(f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)} at location {box}")

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

