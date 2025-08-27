import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import cv2
import json
import time
import numpy as np
import paho.mqtt.client as mqtt

# === MQTT 설정 ===
broker_address = "localhost"
topic = "v2v/hazard"
mqtt_client = mqtt.Client()
mqtt_client.connect(broker_address, 1883, 60)

# === 클래스-액션 매핑 ===
class_action_map = {
    "obstacle": "bypass",
    "red": "stop",
    "yellow": "slowdown",
    "green": "go"
}

# === 클래스 인덱스 ===
class_names = ["green", "obstacle", "red", "yellow"]

# === CUDA 디바이스 정보 출력 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] 현재 사용 중인 디바이스: {device}")
if device.type == 'cuda':
    print(f"[INFO] CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
else:
    print("[INFO] CUDA 사용 불가, CPU로 실행됩니다.")

# === MobileNetV2 모델 로드 ===
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load("./dataset/models/model_mobilenetv2.pth", map_location=device))
model = model.to(device)
model.eval()

print("[INFO] MobileNetV2 모델 로드 완료, 웹캠 시작합니다...")

# === 웹캠 설정 ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
if not cap.isOpened():
    print("[ERROR] 웹캠 열기 실패")
    exit()

# === 마지막 전송 class/action 기억 ===
last_class = None
last_action = None

# === 정규화 텐서 (GPU 전용) ===
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] 프레임 읽기 실패")
        continue

    # === 전처리 ===
    resized = cv2.resize(frame, (224, 224))
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    tensor = (tensor.to(device) - mean) / std
    input_tensor = tensor.unsqueeze(0)

    # === 추론 ===
    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output[0], dim=0)
        pred_idx = torch.argmax(prob).item()
        confidence = prob[pred_idx].item()

    class_name = class_names[pred_idx]
    action = class_action_map.get(class_name, "unknown")

    # === 중복 전송 방지 ===
    if class_name == last_class and action == last_action:
        continue
    last_class = class_name
    last_action = action

    # === 신뢰도 기준 통과 시 MQTT 전송 ===
    if confidence >= 0.5:
        msg = {
            "id": "limo_0001",
            "timestamp": int(time.time()),
            "sender": "limo",
            "type": class_name,
            "action": action,
            "confidence": round(confidence, 4),
            "priority": "high"
        }
        mqtt_client.publish(topic, json.dumps(msg))
        print("[전송]", json.dumps(msg, indent=2))

    # === 디버깅용 화면 표시 ===
    cv2.putText(frame, f"{class_name} ({confidence:.2f})", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("MobileNetV2 Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === 종료 처리 ===
cap.release()
cv2.destroyAllWindows()

