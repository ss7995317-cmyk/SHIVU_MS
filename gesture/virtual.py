import cv2
import mediapipe as mp
import pyautogui
import util
import random
from pynput.mouse import Button, Controller
import time

mouse = Controller()

# Screen size for mapping hand coordinates
screen_width, screen_height = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Cooldown variables to prevent rapid triggering
last_gesture_time = 0
gesture_cooldown = 0.5  # 500ms cooldown

# ---------------- Gesture Definitions ---------------- #

def is_thumb_closed(landmarks_list):
    """Check if thumb is closed (tip near palm)."""
    dist = util.get_distance([landmarks_list[4], landmarks_list[5]])
    return dist < 30  # threshold for closed thumb

def is_left_click(landmarks_list, thumb_index_dist):
    return (
        util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 40 and
        util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) > 100 and
        thumb_index_dist > 60
    )

def is_right_click(landmarks_list, thumb_index_dist):
    return (
        util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 40 and
        util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 100 and
        thumb_index_dist > 60
    )

def is_double_click(landmarks_list, thumb_index_dist):
    return (
        util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 40 and
        util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 40 and
        thumb_index_dist > 60
    )

def is_screenshot(landmarks_list, thumb_index_dist):
    return (
        util.get_angle(landmarks_list[13], landmarks_list[14], landmarks_list[16]) < 40 and
        thumb_index_dist > 60
    )

def can_perform_gesture():
    """Check if enough time has passed since last gesture"""
    global last_gesture_time
    current_time = time.time()
    if current_time - last_gesture_time >= gesture_cooldown:
        last_gesture_time = current_time
        return True
    return False

def move_cursor(landmarks_list):
    """Move cursor using index finger tip position"""
    index_tip = landmarks_list[8]  # index finger tip landmark
    x = int(index_tip[0] * screen_width)
    y = int(index_tip[1] * screen_height)
    pyautogui.moveTo(x, y)

# ---------------- Gesture Detection ---------------- #

def detect_gestures(frame, landmarks_list):
    if len(landmarks_list) < 21:
        return

    thumb_index_dist = util.get_distance([landmarks_list[4], landmarks_list[5]])

    # Cursor movement only when thumb is closed
    if is_thumb_closed(landmarks_list):
        move_cursor(landmarks_list)
        cv2.putText(frame, "Moving Cursor", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    # Click gestures
    if is_double_click(landmarks_list, thumb_index_dist) and can_perform_gesture():
        pyautogui.doubleClick()
        cv2.putText(frame, "Double Click", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    elif is_left_click(landmarks_list, thumb_index_dist) and can_perform_gesture():
        mouse.press(Button.left)
        mouse.release(Button.left)
        cv2.putText(frame, "Left Click", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif is_right_click(landmarks_list, thumb_index_dist) and can_perform_gesture():
        mouse.press(Button.right)
        mouse.release(Button.right)
        cv2.putText(frame, "Right Click", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    elif is_screenshot(landmarks_list, thumb_index_dist) and can_perform_gesture():
        img = pyautogui.screenshot()
        label = random.randint(1, 1000)
        img.save(f'my_screenshot_{label}.png')
        cv2.putText(frame, "Screenshot Saved", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# ---------------- Main Loop ---------------- #

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Hand Gesture Control Started!")
    print("Available Gestures:")
    print("- Cursor movement (index finger, only when thumb closed)")
    print("- Left click")
    print("- Right click")
    print("- Double click")
    print("- Screenshot")
    print("Press 'q' to quit")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            landmarks_list = []
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    landmarks_list.append((lm.x, lm.y))

                detect_gestures(frame, landmarks_list)

            # ✅ Fixed typo here
            cv2.putText(frame, "Press 'q' to quit", (50, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Hand Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()