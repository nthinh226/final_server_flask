import os
import shutil

import cv2
import time
import math
import json
import pickle
import warnings
import numpy as np
# import mediapipe as mp
import face_recognition
from imutils import paths

warnings.filterwarnings('ignore')


def make_480p(cap):
    """Chuyển đổi độ lớn khung hình (resolution) về 640x480

    Args:
        cap (_type_): _description_
    """
    cap.set(3, 640)
    cap.set(4, 480)


# class FaceDetector:
#     """Lớp FaceDetector: phát hiện khuôn mặt bằng thư viện MediaPipe
#     """
#
#     def __init__(self, min_conf=0.75) -> None:
#         self.mp_face_detection = mp.solutions.face_detection
#         self.mp_draw = mp.solutions.drawing_utils
#         self.face_detection = self.mp_face_detection.FaceDetection(
#             min_detection_confidence=min_conf
#         )
#
#     def detect_faces(self, frame, draw=True):
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         self.results = self.face_detection.process(rgb)
#
#         bboxes = []
#
#         if self.results.detections:
#             for id, detection in enumerate(self.results.detections):
#                 bboxC = detection.location_data.relative_bounding_box
#                 f_height, f_width, f_channels = frame.shape
#                 bbox = (
#                     int(bboxC.xmin * f_width),
#                     int(bboxC.ymin * f_height),
#                     int(bboxC.width * f_width),
#                     int(bboxC.height * f_height)
#                 )
#                 bboxes.append([id, bbox, detection.score])
#
#                 if draw:
#                     frame = self.draw_rectangle(frame, bbox)
#
#         return frame, bboxes
#
#     def draw_rectangle(self, frame, bbox, length=30, thickness=5):
#         x, y, w, h = bbox
#         x1, y1 = x + w,  y + h
#
#         cv2.rectangle(frame, bbox, (0, 255, 0), 1)
#
#         # Top left: x, y
#         cv2.line(frame, (x, y), (x + length, y), (0, 255, 0), thickness)
#         cv2.line(frame, (x, y), (x, y + length), (0, 255, 0), thickness)
#
#         # Top right: x1, y
#         cv2.line(frame, (x1, y), (x1 - length, y), (0, 255, 0), thickness)
#         cv2.line(frame, (x1, y), (x1, y + length), (0, 255, 0), thickness)
#
#         # Bottom left: x, y1
#         cv2.line(frame, (x, y1), (x + length, y1), (0, 255, 0), thickness)
#         cv2.line(frame, (x, y1), (x, y1 - length), (0, 255, 0), thickness)
#
#         # Bottom right: x1, y1
#         cv2.line(frame, (x1, y1), (x1 - length, y1), (0, 255, 0), thickness)
#         cv2.line(frame, (x1, y1), (x1, y1 - length), (0, 255, 0), thickness)
#
#         return frame


# Hàm tạo dataset (nên tạo folder `datasets` sẵn). Tạo ảnh xám để thời gian encode nhanh
# def generate_dataset(device=0, c=30, email="abc@gmail.com"):
#     """Tạo dataset có dạng
#     ├── datasets
#         │   ├── votuanan@gmail.com    [20 tấm ảnh]
#
#     Args:
#         device (int, optional): Thiết bị đầu vào camera (mỗi máy có thể khác). Defaults to 0.
#         c (int, optional): Số lượng ảnh muốn chụp. Defaults to 30.
#         email (str, optional): Email người dùng. Defaults to "abc@gmail.com".
#     """
#     # Create dataset folder for each person
#     dataset_path = os.path.join("datasets", f"{email}")
#     os.mkdir(dataset_path)
#
#     print("[INFO] Initializing face capture. Look at the camera/webcam and wait...")
#
#     cap = cv2.VideoCapture(device)  # args["device"]
#
#     if not cap.isOpened():
#         print("Camera/webcam cannot be opened or video file corrupted.")
#         exit()
#
#     # Initialize individual sampling face count
#     count = 0
#
#     make_480p(cap)
#
#     detector = FaceDetector()
#
#     print("[INFO] Video is streaming...")
#     while True:
#         ret, frame = cap.read()
#
#         if ret:
#             frame = cv2.flip(frame, 1, 1)  # Flip to act as a mirror
#
#             frame, bboxes = detector.detect_faces(frame, False)
#
#             cv2.imshow("Frame", frame)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#             for bbox in bboxes:
#                 (x, y, w, h) = bbox[1]
#
#                 if count >= c:  # args["count"]:
#                     print(
#                         f"[INFO] {count} images saved. You can stop streaming now.")
#                     break
#
#                 count += 1
#
#                 file_name = f"{count}.jpg"
#                 cv2.imwrite(
#                     filename=os.path.join(dataset_path, file_name),
#                     # img=frame[y-60:y+h+50, x-50:x+w+50]  # face area
#                     img=gray[y-60:y+h+50, x-50:x+w+50]  # face area
#                 )
#
#             k = cv2.waitKey(1) & 0xFF
#             if k == ord('q') or k == 27:
#                 print("[INFO] Face capturing finished...")
#                 break
#         else:
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()


# Hàm lưu encodings của khuôn mặt / training
def encode_face(email="abc@gmail.com"):
    """Lưu encodings của các khuôn mặt gắn với email tương ứng để sau này load lên để nhận diện

    Args:
        dataset_path (str, optional): Folder chứa dataset ảnh. Defaults to "datasets".
        model_type (str, optional): Model nhận dạng khuôn mặt. Dùng `cnn` nếu có GPU. Defaults to "hog".
        encoding_file (str, optional): Tên file encodings để lưu. Defaults to "encodings.pickle".
    """
    # Grab the paths to the input images in our dataset
    start = time.time()
    print("[INFO] Quantifying faces...")
    dataset_path = os.path.join("src\datasets", email)
    print(dataset_path)
    image_paths = list(paths.list_images(dataset_path))

    # Initialize the list of known encodings and known names
    known_encodings = []
    known_emails = []

    # Loop over the image paths
    for (i, img_path) in enumerate(image_paths):
        # Extract the person email from the image path
        print(f"[INFO] Processing image {i + 1}/{len(image_paths)}")
        email = img_path.split(os.path.sep)[-2]

        # Load the input image and convert it from BGR (OpenCV ordering) to dlib ordering (RGB)
        image = cv2.imread(img_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect the (x, y) coordinates of the bounding boxes corresponding to each face in the input image
        model = "cnn"
        boxes = face_recognition.face_locations(img=rgb, model=model)

        # Compute the facial embeddings for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Loop over the encodings
        for encoding in encodings:
            # Add each encoding + email to our set of known emails and encodings
            known_encodings.append(encoding)
            known_emails.append(email)

    # Dump the facial encodings + emails to disk
    print("[INFO] Serializing encodings...")
    data = {"encodings": known_encodings, "emails": known_emails}

    encoding_file = os.path.join("src/encodings", f"{email}.pickle")
    print(encoding_file)
    with open(encoding_file, "wb") as f:
        f.write(pickle.dumps(data))

    print("[INFO] Done !!!")

    seconds = time.time() - start
    print('Time Taken:', time.strftime("%H:%M:%S", time.gmtime(seconds)))
    # shutil.rmtree(dataset_path);
    # print("Deleted folder " + dataset_path)


def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    """Chuyển khoảng cách giữa 2 khuôn mặt thành phần trăm

    Args:
        face_distance (_type_): Khoảng cách 
        face_match_threshold (float, optional): _description_. Defaults to 0.6 or 0.4718.

    Returns:
        Percentage: Phần trăm
    """
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))


def resize_image_with_aspect_ratio(img, width=None, height=None, inter=cv2.INTER_AREA):
    """Điều chỉnh độ lớn của hình ảnh

    Args:
        img (_type_): Hình ảnh
        width (_type_, optional): Độ rộng mong muốn của hình ảnh. Defaults to None.
        height (_type_, optional): Chiều cao mong muốn của hình ảnh. Defaults to None.
        inter (_type_, optional): Nội suy. Defaults to cv2.INTER_AREA.

    Returns:
        image: Hình ảnh được điều chỉnh
    """
    dim = None
    (h, w) = img.shape[:2]

    if width is None and height is None:
        return img

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(src=img, dsize=dim, interpolation=inter)


def export_to_json(email="Unknown", score=0):
    """Lưu thông tin vào file json

    Args:
        email (str, optional): Email. Defaults to "Unknown".
        score (int, optional): Độ chính xác khuôn mặt. Defaults to 0.
    """
    # Data to processed
    json_dict = {
        "email": email,
        "score": score,
    }
    if email == "Unknown":
        pass
    else:
        # Serializing json
        json_obj = json.dumps(json_dict, indent=4)

        # Writing to sample.json
        with open("sample.json", "w") as json_file:
            json_file.write(json_obj)


# Hàm nhận diện khuôn mặt qua ảnh
def reg_image(img_path="", encoding_file=""):
    """Nhận vào một tấm ảnh và xuất ra màn hình

    Args:
        img_path (str, optional): Đường dẫn đến hình ảnh. VD: 'images/photo.jpg'. Defaults to "".
        model_type (str, optional): Model nhận dạng khuôn mặt. Dùng `cnn` nếu có GPU. Defaults to "hog".
        encoding_file (str, optional): File encodings đã tạo. Defaults to "encodings.pickle".

    Returns:
        Image: Tấm ảnh chứa thông tin email và độ chính xác của khuôn mặt
    """
    # Load the known faces and embeddings
    print("[INFO] Loading encodings...")
    try:
        print("SRC Pickle: " + encoding_file)
        data = pickle.loads(open(encoding_file, "rb").read())
    except Exception as e:
        # print("Facial Embeddings file " + encoding_file + " may not exist.")
        print("Error: " + e)

    # Load the input image and convert it from BGR (OpenCV ordering) to dlib ordering (RGB)
    image = cv2.imread(img_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the (x, y) coordinates of the bounding boxes corresponding to each face in the input image
    print("[INFO] Recognizing faces...")
    model = "hog"
    boxes = face_recognition.face_locations(img=rgb, model=model)

    # Compute the facial embeddings for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Initialize the list of emails for each face detected
    emails = []
    scores = []

    # Loop over the encodings
    for encoding in encodings:
        # Attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        email = "Unknown"
        score = 0

        # # Check to see if we have found a match
        # if True in matches:
        #     # Find the indexes of all matched faces then initialize a dictionary to count the total number of times each face was matched
        #     matched_indices = [i for (i, b) in enumerate(matches) if b]
        #     counts = {}

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(
            data["encodings"],
            encoding
        )
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            email = data["emails"][best_match_index]
        emails.append(email)

        score = face_distance_to_conf(np.min(face_distances))
        score = np.round(score * 100, 1)
        scores.append(score)

    # Loop over the recognized faces
    for ((top, right, bottom, left), email, score) in zip(boxes, emails, scores):
        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, f"{email}", (left, bottom + 25),
                    font, 0.7, (255, 255, 255), 1)
        cv2.putText(image, f"{score}%", (left + 6, top - 20),
                    font, 0.7, (255, 255, 255), 1)

    # # Export result to json
    # export_to_json(email, score)
    #
    # # Show the output image
    # print("[INFO] Finished...")
    # resized = resize_image_with_aspect_ratio(img=image, width=600)
    # cv2.imshow("Image", resized)
    # cv2.waitKey()

    return score


if __name__ == "__main__":
    # Tạo file encodings
    # encode_face(email="votuanan1309@gmail.com")

    # Nhận diện khuôn mặt
    reg_image(img_path="src/4.jpg", encoding_file="src/encodings/b@gmail.com.pickle")
