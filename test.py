import cv2

# 0은 첫 번째 카메라, 1은 두 번째 카메라일 수 있음
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("카메라를 열 수 없습니다. 카메라 번호를 확인하세요.")
    exit()

# 1080p 요청: 실제 적용 여부는 카메라와 연결 환경에 따라 다름
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 화면 비율 기준의 임시 홈 구역. 실제 카메라 위치에 맞춰 조정할 수 있음.
slot_count = 9
left_ratio, right_ratio = 0.1, 0.9
top_ratio, bottom_ratio = 0.2, 0.8

print("카메라 화면이 실행됩니다. 종료하려면 q를 누르세요.")

while True:
    success, frame = camera.read()

    if not success:
        print("카메라 영상을 가져오지 못했습니다.")
        break

    height, width = frame.shape[:2]
    for index in range(slot_count):
        x1 = int(width * (left_ratio + (right_ratio - left_ratio) * index / slot_count))
        x2 = int(width * (left_ratio + (right_ratio - left_ratio) * (index + 1) / slot_count))
        y1, y2 = int(height * top_ratio), int(height * bottom_ratio)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, str(index + 1), (x1 + 5, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Closet Camera", frame)

    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
