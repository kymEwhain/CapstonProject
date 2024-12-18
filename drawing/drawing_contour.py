# 외곽선 시각화 함수
def draw_segmentation_with_contours(image, mask, color=(255, 0, 0), thickness=2):
    """
    이미지에 마스크 외곽선을 추가하는 함수.
    """
    image = image.copy()
    mask = mask.astype(np.uint8)  # 마스크를 이진화 (0과 1로만 구성)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness)  # 외곽선 그리기
    return image



# 네온 효과 함수
def draw_neon_contours(image, mask, neon_color=(255, 0, 255), thickness=3, blur_size=21, glow_strength=3):
    """
    네온 효과를 외곽선에 적용하는 함수.
    Parameters:
        image (numpy.ndarray): 입력 이미지
        mask (numpy.ndarray): 이진화된 마스크 (0과 1로만 구성)
        neon_color (tuple): 네온 색상 (B, G, R)
        thickness (int): 외곽선 두께
        blur_size (int): 흐림 효과의 강도
        glow_strength (int): 흐림을 여러 번 반복해 네온 효과 강화
    """
    output = np.zeros_like(image)
    mask = mask.astype(np.uint8)

    # 외곽선 그리기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, neon_color, thickness, lineType=cv2.LINE_AA)

    # 네온 빛 흐림 효과 반복 적용
    neon_glow = np.zeros_like(output)
    for _ in range(glow_strength):
        output = cv2.GaussianBlur(output, (blur_size, blur_size), 0)
        neon_glow = cv2.addWeighted(neon_glow, 1.0, output, 0.5, 0)

    # 최종 네온 효과와 이미지 합성
    result = cv2.addWeighted(image, 1.0, neon_glow, 0.6, 0)
    cv2.drawContours(result, contours, -1, neon_color, thickness, lineType=cv2.LINE_AA)

    return result



# 비디오 파일 열기
video_path = "/content/drive/MyDrive/Colab Notebooks/vividiva/segmentation.mp4"  # 원본 비디오 경로
cap = cv2.VideoCapture(video_path)

# 비디오 프레임 크기와 FPS 정보 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 비디오 저장을 위한 설정
output_video_path = "/content/drive/MyDrive/Colab Notebooks/vividiva/contour.mp4"  # 저장할 비디오 경로

# VideoWriter 객체 생성 (XVID 코덱을 사용하여 mp4 파일로 저장)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 비디오 프레임 처리
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 비디오 프레임 개수

for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break  # 더 이상 읽을 프레임이 없으면 종료

    # Segmentation 마스크를 프레임에 추가 (투명 배경으로 설정)
    overlay_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # 'video_segments'에서 해당 프레임의 마스크를 가져와서 외곽선 그리기
    if frame_idx in video_segments:  # 비디오 세그멘테이션이 존재하는 경우에만 처리
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            tmp_mask = out_mask.squeeze()  # 마스크 차원 조정
            # 외곽선 그리기
            overlay_image = draw_neon_contours(
                overlay_image, tmp_mask, neon_color=(255, 0, 255), thickness=2
            )

    # 이미지를 BGR로 변환 (OpenCV는 BGR 포맷을 사용)
    overlay_image_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)

    # 비디오에 프레임 추가
    out.write(overlay_image_bgr)

# 비디오 파일 저장 종료
out.release()
cap.release()
