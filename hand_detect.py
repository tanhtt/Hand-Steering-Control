import cv2
import mediapipe as mp
import numpy as np
import socket
import time

# Khởi tạo MediaPipe Hands với cấu hình tối ưu hơn
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Kết nối UDP tới Unity
UDP_IP = "127.0.0.1"
UDP_PORT = 8050
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Mở webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Tham số điều chỉnh
STEERING_SENSITIVITY = 2.0    # Độ nhạy của vô lăng
THROTTLE_SENSITIVITY = 2.5    # Độ nhạy của ga/phanh
SMOOTHING_FACTOR = 0.5        # Hệ số làm mượt (0-1)

# Cấu hình khoảng cách tay
MIN_HAND_DISTANCE = 0.15      # Khoảng cách tối thiểu giữa 2 tay
MAX_HAND_DISTANCE = 0.45      # Khoảng cách tối đa giữa 2 tay
NEUTRAL_MIN = 0.25            # Khoảng cách bắt đầu vùng trung hòa
NEUTRAL_MAX = 0.35            # Khoảng cách kết thúc vùng trung hòa

# Biến lưu trữ giá trị trước đó để làm mượt
prev_horizontal = 0.0
prev_vertical = 0.0
prev_depth = 0.5

# Khởi tạo kích thước căn chỉnh vị trí tay
initial_hand_distance = None
calibration_frames = 0
CALIBRATION_REQUIRED_FRAMES = 30

# Hàm kiểm tra tay nắm lại (cải tiến)
def is_fist(hand_landmarks):
    # Sử dụng các ngón tay để phát hiện nắm đấm
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    # Điểm giữa lòng bàn tay
    wrist = hand_landmarks.landmark[0]
    middle_base = hand_landmarks.landmark[9]
    palm_center_x = (wrist.x + middle_base.x) / 2
    palm_center_y = (wrist.y + middle_base.y) / 2
    
    # Tính khoảng cách từ đầu ngón tay đến lòng bàn tay
    fingers_extended = 0
    
    # Kiểm tra ngón cái (thumb)
    thumb_dist = np.sqrt((thumb_tip.x - palm_center_x)**2 + (thumb_tip.y - palm_center_y)**2)
    if thumb_dist > 0.1:
        fingers_extended += 1
        
    # Kiểm tra ngón trỏ (index)
    index_dist = np.sqrt((index_tip.x - palm_center_x)**2 + (index_tip.y - palm_center_y)**2)
    if index_dist > 0.15:
        fingers_extended += 1
        
    # Kiểm tra ngón giữa (middle)
    middle_dist = np.sqrt((middle_tip.x - palm_center_x)**2 + (middle_tip.y - palm_center_y)**2)
    if middle_dist > 0.15:
        fingers_extended += 1
        
    # Kiểm tra ngón áp út (ring)
    ring_dist = np.sqrt((ring_tip.x - palm_center_x)**2 + (ring_tip.y - palm_center_y)**2)
    if ring_dist > 0.15:
        fingers_extended += 1
        
    # Kiểm tra ngón út (pinky)
    pinky_dist = np.sqrt((pinky_tip.x - palm_center_x)**2 + (pinky_tip.y - palm_center_y)**2)
    if pinky_dist > 0.15:
        fingers_extended += 1
    
    # Nếu có ít hơn 2 ngón tay duỗi ra, coi là nắm đấm
    return fingers_extended < 2

# Hàm xác định tay nào là trái/phải dựa trên nhãn từ MediaPipe
def identify_hands(multi_handedness):
    left_hand = None
    right_hand = None
    
    if multi_handedness:
        for idx, hand_handedness in enumerate(multi_handedness):
            handedness = hand_handedness.classification[0].label
            if handedness == "Left":
                left_hand = idx
            elif handedness == "Right":
                right_hand = idx
                
    return left_hand, right_hand

# Hàm tính góc xoay vô lăng (cải tiến)
def get_steering_angle(left_hand, right_hand):
    if left_hand is None or right_hand is None:
        return 0.0
        
    # Sử dụng điểm giữa của bàn tay để tính góc
    left_wrist = left_hand.landmark[0]
    left_middle = left_hand.landmark[9]
    left_center_x = (left_wrist.x + left_middle.x) / 2
    left_center_y = (left_wrist.y + left_middle.y) / 2
    
    right_wrist = right_hand.landmark[0]
    right_middle = right_hand.landmark[9]
    right_center_x = (right_wrist.x + right_middle.x) / 2
    right_center_y = (right_wrist.y + right_middle.y) / 2
    
    # Tính độ nghiêng
    dx = right_center_x - left_center_x
    dy = right_center_y - left_center_y
    angle = np.arctan2(dy, dx) * 180 / np.pi
    
    # Chuyển đổi góc thành giá trị lái (-1 đến 1)
    # 0 độ = tay ngang = lái thẳng
    # -45 độ = xoay trái tối đa
    # +45 độ = xoay phải tối đa
    steering = np.clip(angle / 45.0 * STEERING_SENSITIVITY, -1.0, 1.0)
    
    return steering

# Hàm tính khoảng cách tới camera (cải tiến)
def get_hand_distance(left_hand, right_hand):
    if left_hand is None or right_hand is None:
        return None
    
    # Sử dụng khoảng cách giữa hai tay
    left_wrist = left_hand.landmark[0]
    right_wrist = right_hand.landmark[0]
    
    # Đo khoảng cách giữa hai tay
    hand_distance = np.sqrt((right_wrist.x - left_wrist.x)**2 + (right_wrist.y - left_wrist.y)**2)
    
    return hand_distance

# Thời gian FPS
prev_time = 0
curr_time = 0

# Để người dùng nhận biết trạng thái hiệu chuẩn
calibration_active = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ camera")
        break

    # Lật ngang hình ảnh để hiển thị dạng gương
    frame = cv2.flip(frame, 1)
    
    # Chuyển đổi màu để xử lý MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Giá trị mặc định
    horizontal = 0.0
    vertical = 0.0
    jump = 0.0
    depth = 0.5
    hand_distance = None
    calibration_text = ""
    
    # Biến lưu tay trái và tay phải
    left_hand_landmarks = None
    right_hand_landmarks = None

    if results.multi_hand_landmarks and results.multi_handedness:
        # Xác định tay trái và tay phải từ nhãn của MediaPipe
        left_idx, right_idx = identify_hands(results.multi_handedness)
        
        # Vẽ landmarks lên hình ảnh
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Gán landmarks cho tay trái và tay phải
            if i == left_idx:
                left_hand_landmarks = hand_landmarks
            elif i == right_idx:
                right_hand_landmarks = hand_landmarks

        # Xử lý các cử chỉ nếu phát hiện được cả hai tay
        if left_hand_landmarks and right_hand_landmarks:
            left_fist = is_fist(left_hand_landmarks)
            right_fist = is_fist(right_hand_landmarks)
            
            # Lấy khoảng cách giữa hai tay
            current_hand_distance = get_hand_distance(left_hand_landmarks, right_hand_landmarks)
            hand_distance = current_hand_distance
            
            # Hiệu chuẩn tự động khoảng cách tay
            if initial_hand_distance is None and left_fist and right_fist:
                if not calibration_active:
                    calibration_active = True
                    calibration_frames = 0
                    
                calibration_frames += 1
                if calibration_frames <= CALIBRATION_REQUIRED_FRAMES:
                    calibration_text = f"Giữ vô lăng ổn định để hiệu chuẩn: {calibration_frames}/{CALIBRATION_REQUIRED_FRAMES}"
                    if calibration_frames == CALIBRATION_REQUIRED_FRAMES:
                        initial_hand_distance = current_hand_distance
                        NEUTRAL_MIN = initial_hand_distance * 0.9
                        NEUTRAL_MAX = initial_hand_distance * 1.05
                        MIN_HAND_DISTANCE = initial_hand_distance * 0.7
                        MAX_HAND_DISTANCE = initial_hand_distance * 1.2
            
            if left_fist and right_fist:  # Cả 2 tay nắm lại = cầm vô lăng
                # Tính góc xoay vô lăng
                raw_horizontal = get_steering_angle(left_hand_landmarks, right_hand_landmarks)
                
                # Áp dụng làm mượt cho vô lăng
                horizontal = prev_horizontal * SMOOTHING_FACTOR + raw_horizontal * (1 - SMOOTHING_FACTOR)
                prev_horizontal = horizontal
                
                # Xử lý ga/phanh dựa vào khoảng cách tay
                if initial_hand_distance is not None:  # Đã hiệu chuẩn
                    normalized_distance = (current_hand_distance - MIN_HAND_DISTANCE) / (MAX_HAND_DISTANCE - MIN_HAND_DISTANCE)
                    normalized_distance = np.clip(normalized_distance, 0.0, 1.0)
                    
                    # Tính giá trị ga/phanh
                    if current_hand_distance > NEUTRAL_MAX:  # Tay xa nhau = tiến
                        raw_vertical = ((current_hand_distance - NEUTRAL_MAX) / (MAX_HAND_DISTANCE - NEUTRAL_MAX)) * THROTTLE_SENSITIVITY
                        raw_vertical = np.clip(raw_vertical, 0.0, 1.0)
                    elif current_hand_distance < NEUTRAL_MIN:  # Tay gần nhau = lùi
                        raw_vertical = ((NEUTRAL_MIN - current_hand_distance) / (NEUTRAL_MIN - MIN_HAND_DISTANCE)) * -THROTTLE_SENSITIVITY
                        raw_vertical = np.clip(raw_vertical, -1.0, 0.0)
                    else:  # Vùng trung lập
                        raw_vertical = 0.0
                    
                    # Áp dụng làm mượt cho ga/phanh
                    vertical = prev_vertical * SMOOTHING_FACTOR + raw_vertical * (1 - SMOOTHING_FACTOR)
                    prev_vertical = vertical
            else:
                # Phát hiện được 2 tay nhưng không nắm = không lái
                horizontal = prev_horizontal * SMOOTHING_FACTOR  # Giảm dần về 0
                vertical = prev_vertical * SMOOTHING_FACTOR      # Giảm dần về 0
                prev_horizontal = horizontal
                prev_vertical = vertical

    # Đảm bảo các giá trị nằm trong khoảng [-1, 1]
    horizontal = np.clip(horizontal, -1.0, 1.0)
    vertical = np.clip(vertical, -1.0, 1.0)
    
    # Tính và hiển thị FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    
    # Hiển thị thông tin lên hình ảnh
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Hiển thị trạng thái hiệu chuẩn
    if calibration_text:
        cv2.putText(frame, calibration_text, (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Hiển thị các giá trị điều khiển
    control_text = f"Lái: {horizontal:.2f}, Ga/Phanh: {vertical:.2f}"
    cv2.putText(frame, control_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Hiển thị trạng thái nắm tay và khoảng cách
    if left_hand_landmarks and right_hand_landmarks:
        left_fist = is_fist(left_hand_landmarks)
        right_fist = is_fist(right_hand_landmarks)
        status = "Cầm vô lăng" if left_fist and right_fist else "Không cầm vô lăng"
        cv2.putText(frame, status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if hand_distance is not None:
            cv2.putText(frame, f"Khoảng cách tay: {hand_distance:.3f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if initial_hand_distance is not None:
                cv2.putText(frame, f"Chuẩn: {initial_hand_distance:.3f}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Lấy kích thước frame
    height, width, _ = frame.shape
        
    # Vẽ trực quan vô lăng và ga/phanh
    center_x, center_y = width - 150, height - 150
    
    # Vẽ vô lăng
    cv2.circle(frame, (center_x, center_y), 80, (255, 255, 255), 2)
    # Vẽ mốc chính giữa vô lăng
    cv2.line(frame, 
             (center_x, center_y - 80), 
             (center_x, center_y - 60), 
             (255, 255, 255), 2)
    
    # Vẽ hướng lái
    angle_rad = horizontal * np.pi / 2
    end_x = int(center_x + 70 * np.sin(angle_rad))
    end_y = int(center_y - 70 * np.cos(angle_rad))
    cv2.line(frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3)
    
    # Vẽ thanh ga/phanh
    throttle_x = 50
    throttle_y = height - 150
    throttle_height = 200
    cv2.rectangle(frame, (throttle_x - 20, throttle_y - throttle_height//2), 
                 (throttle_x + 20, throttle_y + throttle_height//2), (255, 255, 255), 2)
    
    # Vạch giữa
    cv2.line(frame, (throttle_x - 25, throttle_y), (throttle_x + 25, throttle_y), (255, 255, 255), 2)
    
    # Điền màu cho ga/phanh
    throttle_val = int(vertical * throttle_height//2)
    if throttle_val > 0:  # Ga (xanh)
        cv2.rectangle(frame, (throttle_x - 15, throttle_y), 
                     (throttle_x + 15, throttle_y - throttle_val), (0, 255, 0), -1)
    elif throttle_val < 0:  # Phanh (đỏ)
        cv2.rectangle(frame, (throttle_x - 15, throttle_y), 
                     (throttle_x + 15, throttle_y - throttle_val), (0, 0, 255), -1)
    
    # Gửi dữ liệu đến Unity
    message = f"[{horizontal:.2f},{vertical:.2f},{jump:.1f}]".encode('utf-8')
    sock.sendto(message, (UDP_IP, UDP_PORT))
    
    # Hiển thị hình ảnh
    cv2.imshow("Hand Steering Control", frame)
    
    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
sock.close()