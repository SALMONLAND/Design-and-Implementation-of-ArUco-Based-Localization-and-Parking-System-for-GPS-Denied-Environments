import cv2
import cv2.aruco as aruco
import numpy as np

FRAME_W, FRAME_H = 640, 480

camera_matrix = np.array([
    [FRAME_W,       0, FRAME_W / 2],
    [      0, FRAME_W, FRAME_H / 2],
    [      0,       0,           1]
], dtype=np.float32)

dist_coeffs = np.zeros((4, 1), dtype=np.float32)

MARKER_SIZE = 0.04  # 마커 실제 크기 (미터) ← 본인 마커 크기로 수정!

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

half = MARKER_SIZE / 2
obj_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0]
], dtype=np.float32)

def find_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            return cap
        cap.release()
    return None

def rvec_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    yaw   = np.degrees(np.arctan2( R[1, 0], R[0, 0]))
    roll  = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
    return yaw, pitch, roll

def detect_and_estimate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        print("Aruco ID : None | (yaw, pitch, roll) : (None, None, None) | dist : None")
        return frame

    aruco.drawDetectedMarkers(frame, corners, ids)

    for i, marker_id in enumerate(ids.flatten()):
        success, rvec, tvec = cv2.solvePnP(
            obj_points, corners[i], camera_matrix, dist_coeffs
        )
        if not success:
            continue

        yaw, pitch, roll = rvec_to_euler(rvec)

        # ── 거리 계산 ──────────────────────────────────
        dist_m  = np.linalg.norm(tvec)          # 직선 거리 (미터)
        dist_cm = dist_m * 100                  # 센티미터 변환

        print(f"Aruco ID : {marker_id} | "
              f"(yaw, pitch, roll) : ({yaw:+.1f}, {pitch:+.1f}, {roll:+.1f}) | "
              f"dist : {dist_cm:.1f} cm")

    return frame

cap = find_camera()
if cap is None:
    print("사용 가능한 카메라가 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detect_and_estimate(frame)

cap.release()
