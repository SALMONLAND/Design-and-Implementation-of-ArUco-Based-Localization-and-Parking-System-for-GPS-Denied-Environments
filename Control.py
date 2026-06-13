import cv2
import cv2.aruco as aruco
import numpy as np
import lgpio
import time
import signal
import sys

# ─────────────────────────────────────────
#  설정
# ─────────────────────────────────────────
MARKER_SIZE       = 0.05    # 마커 실제 크기 (m)
FRAME_W, FRAME_H  = 640, 480
BASE_YAW          = -92.0   # 수평 기준 yaw 오프셋

# 거리 구간 (cm)
DIST_STOP         = 5.0   # 정지
DIST_SLOW         = 40.0    # 저속
DIST_MID          = 70.0    # 중속
# DIST_MID 초과 → 고속

# 고정 모터값 (0.0 ~ 1.0)
SPEED_FAST        = 0.65
SPEED_MID         = 0.50
SPEED_SLOW        = 0.35

# 조향 보정량 (yaw 오차가 임계값 초과 시 한쪽을 이만큼 감산)
YAW_THRESH        = 10.0    # 이 각도 이하면 직진 간주
STEER_CORR        = 0.15    # 보정 강도

# 마커 ID 역할
ID_FORWARD  = 0
ID_LEFT     = 1
ID_RIGHT    = 2
ID_STOP     = 3
ID_PARK     = 10   # 정밀 주차 진입

# 마커 미감지 연속 프레임 수 초과 시 정지
NO_MARKER_LIMIT   = 30

# ─────────────────────────────────────────
#  GPIO 설정
# ─────────────────────────────────────────
GPIOCHIP = 4
LFA, LRA = 17, 27
LFB, LRB = 22, 23
RFA, RRA = 24, 25
RFB, RRB =  4, 18
ALL_PINS  = [LFA, LRA, LFB, LRB, RFA, RRA, RFB, RRB]

h = lgpio.gpiochip_open(GPIOCHIP)
for p in ALL_PINS:
    lgpio.gpio_claim_output(h, p, 0)

def _set_single(fwd_pin, rev_pin, d):
    """단일 모터 1개 제어: d>0 전진, d<0 후진, d=0 정지"""
    lgpio.gpio_write(h, fwd_pin, 1 if d > 0 else 0)
    lgpio.gpio_write(h, rev_pin, 1 if d < 0 else 0)

def drive(l, r):
    """
    직진/조향용 4모터 제어. 앞뒤 같은 방향.
    l, r: -1(후진) ~ +1(전진)
    """
    lv = max(-1.0, min(1.0, l))
    rv = max(-1.0, min(1.0, r))
    _set_single(LFA, LRA, lv)   # 왼쪽 앞
    _set_single(LFB, LRB, lv)   # 왼쪽 뒤
    _set_single(RFA, RRA, rv)   # 오른쪽 앞
    _set_single(RFB, RRB, rv)   # 오른쪽 뒤

def drive_turn(l, r):
    """
    회전 전용 4모터 제어.
    사선으로 맞물리는 문제 해결:
      오른쪽 앞(RFA/RRA), 왼쪽 뒤(LFB/LRB) 만 부호 반전
    """
    lv = max(-1.0, min(1.0, l))
    rv = max(-1.0, min(1.0, r))
    _set_single(LFA, LRA,  lv)  # 왼쪽 앞  → 정상
    _set_single(LFB, LRB, -lv)  # 왼쪽 뒤  → 반전
    _set_single(RFA, RRA, -rv)  # 오른쪽 앞 → 반전
    _set_single(RFB, RRB,  rv)  # 오른쪽 뒤 → 정상

def motor_test():
    """
    바퀴별 개별 동작 테스트.
    실행하면 각 바퀴를 1.2초씩 +0.6 값으로 구동.
    각 바퀴가 '전진 방향'으로 도는지 직접 눈으로 확인.
    반대로 돌면 해당 핀의 fwd/rev를 swap하거나 부호 반전 필요.
    """
    tests = [
        ("왼쪽 앞  (LFA=17 / LRA=27)", LFA, LRA),
        ("왼쪽 뒤  (LFB=22 / LRB=23)", LFB, LRB),
        ("오른쪽 앞 (RFA=24 / RRA=25)", RFA, RRA),
        ("오른쪽 뒤 (RFB=4  / RRB=18)", RFB, RRB),
    ]
    print("\n=== 바퀴 방향 테스트 (python3 aruco_drive.py test) ===")
    for name, fwd, rev in tests:
        print(f"▶ {name}  → +0.6 인가 중 (전진 방향으로 돌아야 정상)")
        _set_single(fwd, rev, 0.6)
        time.sleep(1.2)
        stop()
        time.sleep(0.4)
    print("=== 완료 — 반대로 돈 바퀴 번호를 알려주세요 ===\n")

def stop():
    drive(0, 0)

def cleanup(*_):
    stop()
    for p in ALL_PINS:
        lgpio.gpio_write(h, p, 0)
    lgpio.gpiochip_close(h)
    cap.release()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

# ─────────────────────────────────────────
#  ArUco 설정
# ─────────────────────────────────────────
camera_matrix = np.array([
    [FRAME_W,       0, FRAME_W / 2],
    [      0, FRAME_W, FRAME_H / 2],
    [      0,       0,           1]
], dtype=np.float32)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

half = MARKER_SIZE / 2
obj_points = np.array([
    [-half,  half, 0], [ half,  half, 0],
    [ half, -half, 0], [-half, -half, 0]
], dtype=np.float32)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

def find_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            print(f"카메라 인덱스 {i} 사용")
            return cap
        cap.release()
    return None

def rvec_to_yaw(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return np.degrees(np.arctan2(R[1, 0], R[0, 0]))

def detect(frame):
    """
    Returns: (marker_id, yaw, dist_cm) for the closest marker, or None
    """
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None

    best = None
    for i, marker_id in enumerate(ids.flatten()):
        ok, rvec, tvec = cv2.solvePnP(
            obj_points, corners[i], camera_matrix, dist_coeffs
        )
        if not ok:
            continue
        yaw     = rvec_to_yaw(rvec)
        dist_cm = np.linalg.norm(tvec) * 100
        if best is None or dist_cm < best[2]:
            best = (int(marker_id), yaw, dist_cm)

    return best

# ─────────────────────────────────────────
#  단순 구간 로직
# ─────────────────────────────────────────
def decide_speed(dist_cm):
    """거리 구간에 따라 기본 속도 반환 (후진이므로 음수)"""
    if dist_cm > DIST_MID:
        return -SPEED_FAST
    elif dist_cm > DIST_SLOW:
        return -SPEED_MID
    else:
        return -SPEED_SLOW

def decide_steer(yaw, base_speed):
    """
    yaw 오차로 좌우 모터값 계산.
    yaw > BASE_YAW + thresh → 오른쪽으로 치우침 → 좌측 감속
    yaw < BASE_YAW - thresh → 왼쪽으로 치우침  → 우측 감속
    """
    err = yaw - BASE_YAW
    l, r = base_speed, base_speed

    if err > YAW_THRESH:        # 오른쪽 편향 → 좌측 줄여서 우회전
        l = base_speed + STEER_CORR   # 후진값이므로 + = 더 느리게
    elif err < -YAW_THRESH:     # 왼쪽 편향 → 우측 줄여서 좌회전
        r = base_speed + STEER_CORR

    return l, r

def handle_id(marker_id, yaw, dist_cm):
    """
    마커 ID별 동작 분기.
    반환: True = 계속 주행, False = 이 프레임 종료(정지 등)
    """
    # ── 정지 마커 ──────────────────────────────
    if marker_id == ID_STOP:
        stop()
        print(f"[ID {marker_id}] 정지 마커 → 정지")
        return False

    # ── 목표 거리 도달 ─────────────────────────
    if dist_cm <= DIST_STOP:
        stop()
        print(f"[ID {marker_id}] {dist_cm:.1f} cm → 목표 도달! 정지")
        return False

    # ── 주차 마커 (ID 10): 정면 정렬 우선 ──────
    if marker_id == ID_PARK:
        base  = decide_speed(dist_cm)
        l, r  = decide_steer(yaw, base)
        drive(l, r)
        print(f"[PARK] dist={dist_cm:.1f}cm  yaw_err={yaw-BASE_YAW:+.1f}°  L={l:+.2f} R={r:+.2f}")
        return True

    # ── 경로 안내 마커 (ID 0/1/2): ID 방향 우선 ─
    base = decide_speed(dist_cm)

    if marker_id == ID_LEFT:
        # 좌회전: 좌측 후진 + 우측 전진 → 4바퀴 탱크 턴
        drive_turn(base, -base)
        print(f"[LEFT ] dist={dist_cm:.1f}cm  → 좌회전 (탱크턴)")

    elif marker_id == ID_RIGHT:
        # 우회전: 좌측 전진 + 우측 후진 → 4바퀴 탱크 턴
        drive_turn(-base, base)
        print(f"[RIGHT] dist={dist_cm:.1f}cm  → 우회전 (탱크턴)")

    else:  # ID_FORWARD 또는 미정의 ID → 직진+조향 보정
        l, r = decide_steer(yaw, base)
        drive(l, r)
        print(f"[FWD ] dist={dist_cm:.1f}cm  yaw_err={yaw-BASE_YAW:+.1f}°  L={l:+.2f} R={r:+.2f}")

    return True

# ─────────────────────────────────────────
#  메인 루프
# ─────────────────────────────────────────
cap = find_camera()
if cap is None:
    print("카메라를 찾을 수 없습니다.")
    sys.exit(1)

# 테스트 모드: python3 aruco_drive.py test
if len(sys.argv) > 1 and sys.argv[1] == "test":
    motor_test()
    cleanup()
    sys.exit(0)

print("=" * 50)
print("단순 구간 제어 주행 시작")
print(f"  정지 거리  : {DIST_STOP} cm")
print(f"  저속 구간  : {DIST_SLOW} cm 이하")
print(f"  중속 구간  : {DIST_MID} cm 이하")
print(f"  yaw 임계값 : ±{YAW_THRESH}°")
print("Ctrl+C 로 종료")
print("=" * 50)

no_marker_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    result = detect(frame)

    if result is None:
        no_marker_count += 1
        if no_marker_count >= NO_MARKER_LIMIT:
            stop()
            print(f"[마커 없음] {NO_MARKER_LIMIT}프레임 연속 미감지 → 정지")
        else:
            print(f"[마커 없음] 탐색 중... ({no_marker_count}/{NO_MARKER_LIMIT})")
        continue

    no_marker_count = 0
    marker_id, yaw, dist_cm = result
    handle_id(marker_id, yaw, dist_cm)
