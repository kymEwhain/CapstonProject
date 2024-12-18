# mediapipe 실시간 반영 사용자 평가 코드 입니다.

import cv2
import mediapipe as mp
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import time

# Mediapipe Pose 모델 초기화
mp_pose = mp.solutions.pose

# 전문가 영상에서 키포인트 추출 함수
def extract_keypoints_from_video(video_path):
    keypoints_list = []
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)
            if result.pose_landmarks:
                keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in result.pose_landmarks.landmark])
                keypoints_list.append(keypoints)
    cap.release()
    return np.array(keypoints_list)

# 포즈 정규화 함수 (기준점: 골반)
def normalize_keypoints(keypoints):
    pelvis = keypoints[24]  # 골반 좌표 (Assuming landmark 24 is pelvis)
    return keypoints - pelvis

# 실시간 웹캠 입력과 전문가 포즈 비교 함수
def process_webcam_and_compare(expert_video_path):
    # 전문가 키포인트 미리 추출
    expert_keypoints_list = extract_keypoints_from_video(expert_video_path)
    
    # 전문가 키포인트 정규화
    normalized_expert_keypoints_list = [normalize_keypoints(kp) for kp in expert_keypoints_list]
    
    # 웹캠 설정
    cap_webcam = cv2.VideoCapture(0)
    
    frame_idx = 0
    num_expert_frames = len(normalized_expert_keypoints_list)
    
    with mp_pose.Pose() as pose:
        while cap_webcam.isOpened():
            ret_webcam, frame_webcam = cap_webcam.read()
            if not ret_webcam:
                break
            
            # 웹캠 프레임 처리
            frame_webcam_rgb = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
            result_webcam = pose.process(frame_webcam_rgb)
            
            if result_webcam.pose_landmarks:
                # 사용자 키포인트 추출 및 정규화
                user_keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in result_webcam.pose_landmarks.landmark])
                normalized_user_keypoints = normalize_keypoints(user_keypoints)
                
                # 전문가 프레임과 비교 (순환 재생)
                normalized_expert_keypoints = normalized_expert_keypoints_list[frame_idx % num_expert_frames]
                
                # 1D로 변환 후 DTW 거리 계산
                expert_flat = normalized_expert_keypoints.flatten()
                user_flat = normalized_user_keypoints.flatten()
                distance, _ = fastdtw(expert_flat.reshape(-1, 1), user_flat.reshape(-1, 1), dist=euclidean)
                
                """
                # 평가 점수 및 멘트
                if distance < 10:
                    feedback = "Perfect! Great job!"
                elif distance < 30:
                    feedback = "Good job! You'll be perfect!"
                else:
                    feedback = "It's okay! Keep trying!"
                
                # 화면에 텍스트 출력
                cv2.putText(frame_amateur, f"Score: {distance:.2f}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_amateur, feedback, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                """
                # 점수 계산 (100점 만점 변환)
                max_distance = 40  # 'bad' 기준인 40을 최대 거리로 설정
                score = max(100 - (distance / max_distance) * 100, 0)  # 100에서 거리 비율만큼 감소
                
                # 평가 점수 및 멘트
                if score >= 95:
                    feedback = "Perfect! Great job!"
                elif score >= 85:
                    feedback = "Good! You're almost there!"
                elif score >= 75:
                    feedback = "Normal! Keep going!"
                elif score >= 60:
                    feedback = "Nice try! You're getting there!"
                else:
                    feedback = "Good effort! Keep it up!"
            
                # 화면에 텍스트 출력
                cv2.putText(frame_webcam, f"Score: {distance:.2f}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_webcam, feedback, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            # 웹캠 화면 출력
            cv2.imshow("Webcam Pose Comparison", frame_webcam)
            
            # 프레임 인덱스 업데이트 (전문가 영상 순환)
            frame_idx += 1
            
            # 종료 조건 (q 키를 누르면 종료)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
    cap_webcam.release()
    cv2.destroyAllWindows()

# 전문가 영상 경로 설정 - 이부분 파일명 저장하신대로로 바꿔주세요!!
expert_video_path = "videos1/expert_dance1.mp4"

# 함수 실행
process_webcam_and_compare(expert_video_path)
