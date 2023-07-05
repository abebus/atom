'''
извлечение стрима из повернета
'''
import cv2

stream_url = 'https://flussonic2.powernet.com.ru:8081/user83831/tracks-v1/mono.m3u8?token=dont-panic-and-carry-a-towel'

cap = cv2.VideoCapture(stream_url)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

exit()
'''
детекция объектов через йоло
'''
import requests
import cv2

my_token = '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'
API_URL = "https://api-inference.huggingface.co/models/hustvl/yolos-tiny"
headers = {"Authorization": f"Bearer {my_token}"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("D:/Desktop/72499026.0.0.jpg")
print(output)