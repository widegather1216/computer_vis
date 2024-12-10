from ultralytics import YOLO
import cv2  # OpenCV를 사용하여 이미지를 처리하고 표시
import os

# YOLO 모델 불러오기
model = YOLO("yolo11n-seg.pt")
input_folder = "inputs"  # 이미지가 저장된 폴더 경로
output_folder = "outputs"  # 결과 이미지를 저장할 폴더 경로
os.makedirs(output_folder, exist_ok=True)  # 출력 폴더 생성 (존재하지 않을 경우)

for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # 지원하는 이미지 확장자
        # 이미지 경로
        img_path = os.path.join(input_folder, file_name)
        img = cv2.imread(img_path)  # OpenCV로 이미지를 읽어옴
        results = model(img)
        annotated_frame = results[0].plot()

        # 이미지 크기
        height, width, _ = annotated_frame.shape  # 이미지 크기

        # 9분할 격자선 그리기
        # 가로, 세로 3등분 선을 그립니다.
        for i in range(1, 3):  # 3등분을 위해 1, 2번째 선 추가
            # 가로 선 그리기
            cv2.line(annotated_frame, (0, i * height // 3), (width, i * height // 3), (255, 255, 255),1)
            # 세로 선 그리기
            cv2.line(annotated_frame, (i * width // 3, 0), (i * width // 3, height), (255, 255, 255),1)

        # 세그멘테이션 결과 시각화
        #cv2.imshow("Segmented Image", annotated_frame)

        # 결과 저장
        output_path = os.path.join(output_folder, f"segmented_{file_name}")
        cv2.imwrite(output_path, annotated_frame)  # 분석 결과 이미지 저장

        # OpenCV 창이 종료되기 전까지 기다립니다.
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
