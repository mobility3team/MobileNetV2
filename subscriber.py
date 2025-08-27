## subscriber.py
# -*- coding: utf-8 -*-
import paho.mqtt.client as mqtt
import json
import time
import threading

# ===== 모터 제어 스텁 함수 (TODO: 실제 코드로 교체) =====
def motor_stop():
    print(">>> [동작] 모터 정지")

def motor_bypass():
    print(">>> [동작] 회피 주행")

def motor_set_speed(speed):
    print(f">>> [동작] 속도 {speed} km/h로 설정")

def motor_resume():
    print(">>> [동작] 차량 진행 재개")

def motor_slowdown():
    print(">>> [동작] 차량 서행")

def show_on_lcd(text):
    # 실제 LCD/OLED에 표시하는 코드가 들어갈 자리
    print(f"[LCD 출력] {text}")

# ===== 상태 관리 =====
last_action = None           # 최근 action
last_msg_time = time.time()  # 마지막 수신 시각
timeout_sec = 5              # 5초 동안 메시지 없으면 fail-safe (보류)

# ===== [보류] 주기적으로 메시지 수신 여부 확인 =====
# def watchdog():
#     while True:
#         now = time.time()
#         if now - last_msg_time > timeout_sec:
#             print("[안전] 최근 메시지 없음, fail-safe 정지 수행")
#             motor_stop()
#         time.sleep(1)
#
# watchdog_thread = threading.Thread(target=watchdog, daemon=True)
# watchdog_thread.start()

# ===== 메시지 처리 로직 =====
def handle_message(data):
    global last_action, last_msg_time

    # JSON 필드 추출
    msg_id = data.get("id", "")
    timestamp = data.get("timestamp", "")
    sender = data.get("sender", "")
    msg_type = data.get("type", "")
    action = data.get("action", "")
    confidence = data.get("confidence", 0.0)
    priority = data.get("priority", "medium")

    # 기본 로그
    print(f"\n\n[수신] id={msg_id}, sender={sender}, type={msg_type}, action={action}, priority={priority}, confidence={confidence}")

    # 신뢰도 필터링
    if confidence < 0.5:
        # print(">>> 신뢰도 낮음, 무시")
        return

    # 중복 방지
    if action == last_action:
        # print(">>> 같은 action 반복 수신, 무시")
        return
    last_action = action

    # 우선도 표시
    if priority == "high":
        print(">>> [우선도] HIGH: 즉각적 반응")
    elif priority == "medium":
        print(">>> [우선도] MEDIUM: 상황에 따라 반응")
    else:
        print(">>> [우선도] LOW: 참고용")

    # LCD/OLED 출력 (지금은 print)
    show_on_lcd(f"{action.upper()} ({priority})")

    # action에 따른 제어
    if action.startswith("speed_"):
        speed_val = action.split("_")[1]
        motor_set_speed(speed_val)
    elif action == "stop":
        motor_stop()
    elif action == "go":
        motor_resume()
    elif action == "slowdown":
        motor_slowdown()
    elif action == "bypass":
        motor_bypass()
    else:
        print(f">>> 정의되지 않은 action: {action}")

    # type에 따른 상황 로그
    if msg_type == "construction":
        print(">>> [상황] 공사 구간 감지")
    elif msg_type == "obstacle":
        print(">>> [상황] 장애물 감지")
    elif msg_type == "signal_light":
        print(">>> [상황] 신호등 감지")
    elif msg_type == "sign":
        print(">>> [상황] 표지판 감지")
    elif msg_type == "dinosour":
        print(">>> [상황] 돌발 장애물 감지")
    else:
        print(">>> [상황] 정의되지 않은 위험 타입")

    # 마지막 메시지 시각 갱신
    last_msg_time = time.time()

# ===== MQTT 콜백 =====
def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        handle_message(data)
    except Exception as e:
        print("[에러] 메시지 파싱 실패:", e)

# ===== [보류] MQTT 연결 끊김 시 fail-safe =====
# def on_disconnect(client, userdata, rc):
#     print("[MQTT] 브로커 연결 끊김! fail-safe 정지 수행")
#     motor_stop()

# ===== MQTT 구독 =====
client = mqtt.Client()
client.on_message = on_message
# client.on_disconnect = on_disconnect  # [보류]

client.connect("localhost", 1883, 60)  # 실제 테스트 시 라즈베리파이 IP로 변경
client.subscribe("v2v/hazard")
print("[시작] MQTT Subscriber 대기 중...")
client.loop_forever()

