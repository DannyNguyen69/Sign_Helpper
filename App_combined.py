import os
import sys
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter

# Lấy đường dẫn đúng cho MLP_model.p
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'MLP_model.p')

if not os.path.exists(model_path):
    print(f"❌ Không tìm thấy file mô hình tại: {model_path}")
    sys.exit()

with open(model_path, 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']

# Khai báo nhãn
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
    19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

DESIRED_ASPECT_RATIO = 1.3333
PADDING = 10
HISTORY_LENGTH = 10
COOLDOWN_SECONDS = 2.5  # thời gian chờ giữa 2 ký tự

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
prev_frame_time = time.time()

# Thêm dòng này để cố định kích thước cửa sổ
cv2.namedWindow('Hand Sign Recognition + Detected Sentence', cv2.WINDOW_AUTOSIZE)

# Các biến lưu trữ
prediction_history = deque(maxlen=HISTORY_LENGTH)
recognized_text = ""
last_added_time = 0
last_added_char = ""

# Biến nút Reset
reset_button_clicked = False
reset_button_rect = (10, 0, 120, 40)  # Sẽ cập nhật y sau

# Hàm callback chuột

def mouse_callback(event, x, y, flags, param):
    global recognized_text, reset_button_rect, reset_button_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # x, y là tọa độ trên combined_img
        # Kiểm tra nếu click vào vùng nút Reset (bên phải)
        frame_width = param['frame_width']
        text_img_height = param['text_img_height']
        btn_x, btn_y, btn_w, btn_h = reset_button_rect
        # Dịch x sang vùng text_img
        if x >= frame_width:
            rel_x = x - frame_width
            rel_y = y
            if btn_x <= rel_x <= btn_x + btn_w and btn_y <= rel_y <= btn_y + btn_h:
                recognized_text = ""
                reset_button_clicked = True

def calculate_bounding_box(hand_landmarks, frame_shape):
    h, w, _ = frame_shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x, x_max)
        y_max = max(y, y_max)
    x_min = max(0, x_min - PADDING)
    y_min = max(0, y_min - PADDING)
    x_max = min(w, x_max + PADDING)
    y_max = min(h, y_max + PADDING)
    return x_min, y_min, x_max, y_max

def enforce_aspect_ratio(x_min, y_min, x_max, y_max, frame_shape, desired_aspect_ratio):
    h, w, _ = frame_shape
    box_width = x_max - x_min
    box_height = y_max - y_min
    current_aspect_ratio = box_height / box_width
    if current_aspect_ratio < desired_aspect_ratio:
        new_height = int(box_width * desired_aspect_ratio)
        y_center = (y_min + y_max) // 2
        y_min = max(0, y_center - new_height // 2)
        y_max = min(h, y_center + new_height // 2)
    elif current_aspect_ratio > desired_aspect_ratio:
        new_width = int(box_height / desired_aspect_ratio)
        x_center = (x_min + x_max) // 2
        x_min = max(0, x_center - new_width // 2)
        x_max = min(w, x_center + new_width // 2)
    return x_min, y_min, x_max, y_max

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time)
    prev_frame_time = current_time

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label != 'Left':
                continue

            x_min, y_min, x_max, y_max = calculate_bounding_box(hand_landmarks, frame.shape)
            x_min, y_min, x_max, y_max = enforce_aspect_ratio(x_min, y_min, x_max, y_max, frame.shape, DESIRED_ASPECT_RATIO)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

            x_, y_, data_aux = [], [], []
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_index = int(prediction[0])
            prediction_history.append(predicted_index)

            most_common = Counter(prediction_history).most_common(1)[0]
            if most_common[1] > HISTORY_LENGTH * 0.7:
                predicted_character = labels_dict.get(most_common[0], "???")
                cv2.putText(frame, f'Ki hieu: {predicted_character}', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)

                if predicted_character != "???" and (time.time() - last_added_time > COOLDOWN_SECONDS or predicted_character != last_added_char):
                    recognized_text += predicted_character
                    last_added_char = predicted_character
                    last_added_time = time.time()

    # Hiển thị FPS
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2)

    # Tạo box câu chữ 
    text_img = np.zeros((frame.shape[0], 400, 3), dtype=np.uint8)

    # Hàm wrap_text mới 
    def wrap_text(text, font, font_scale, thickness, max_width):
        lines = []
        current_line = ""
        for char in text:
            test_line = current_line + char
            (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if w > max_width and current_line:
                lines.append(current_line)
                current_line = char
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)
        return lines

    wrapped_lines = wrap_text(recognized_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2, 380)
    y0 = 60
    dy = 40
    for i, line in enumerate(wrapped_lines):
        y = y0 + i * dy
        if y > text_img.shape[0] - 60:
            break
        cv2.putText(text_img, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Vẽ nút Reset 
    btn_x, btn_y, btn_w, btn_h = 10, text_img.shape[0] - 50, 120, 40
    reset_button_rect = (btn_x, btn_y, btn_w, btn_h)
    cv2.rectangle(text_img, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (255, 255, 255), 2)
    cv2.putText(text_img, "Reset", (btn_x + 15, btn_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # cam bên trái, câu chữ bên phải
    combined_img = np.hstack((frame, text_img))

    # Đăng ký callback chuột
    cv2.setMouseCallback('Hand Sign Recognition + Detected Sentence', mouse_callback, param={'frame_width': frame.shape[1], 'text_img_height': text_img.shape[0]})

    cv2.imshow('Hand Sign Recognition + Detected Sentence', combined_img)

    # Xử lý phím bấm
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 8:  # Backspace
        recognized_text = recognized_text[:-1]
    elif key == 13:  # Enter
        recognized_text = ""
    elif key == 32:  # Spacebar
        recognized_text += " "

cap.release()
cv2.destroyAllWindows() 