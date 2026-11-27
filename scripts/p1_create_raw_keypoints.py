# scripts/download_and_extract.py

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pytubefix import YouTube
import mediapipe as mp
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_VIDEO_DIR = os.path.join(ROOT, "data", "raw_videos")
RAW_KEYPOINT_DIR = os.path.join(ROOT, "data", "raw_keypoints")
RAW_TEST_DIR = os.path.join(ROOT, "data", "test_keypoints")

os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
os.makedirs(RAW_KEYPOINT_DIR, exist_ok=True)

TARGET_FPS = 30     # [ADDED] 추출할 프레임레이트 (p1_2의 FPS와 같게)
VIDEO_SPEED = 1.0   # [ADDED] 영상 재생 배속 (일반 영상 1.0, 슬로모션 0.5배속이면 0.5)

mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def download_youtube(url):
    yt = YouTube(url)
    name = yt.title.replace(" ", "_").replace("/", "_")
    filepath = f"{RAW_VIDEO_DIR}/{name}.mp4"
    yt.streams.filter(file_extension='mp4').first().download(
        output_path=RAW_VIDEO_DIR,
        filename=f"{name}.mp4"
    )
    print(f"🎬 Downloaded → {filepath}")
    return filepath, name


def extract_3d_keypoints(video_path, dir_path =RAW_KEYPOINT_DIR, name ="Data", speed=VIDEO_SPEED):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_rows = []

    # [ADDED] 실제 시간 기준 TARGET_FPS가 되도록 프레임을 건너뜀
    #   - 60fps 영상            → 2프레임마다 1개
    #   - 30fps + 0.5배속 영상  → 실제로는 60fps와 같으므로 2프레임마다 1개
    src_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    real_fps = src_fps / speed                     # 실제 동작 기준 프레임레이트
    step = max(1, round(real_fps / TARGET_FPS))
    out_fps = real_fps / step
    print(f"source fps: {src_fps:.1f}, speed x{speed} → using every {step} frame(s) ({out_fps:.1f} fps, real time)")

    pbar = tqdm(total=total_frames, desc="Extracting BlazePose 3D",
                ascii=True,          # unicode 막대 → ASCII 막대로 변경
                dynamic_ncols=False,  # 윈도우 콘솔 버그 방지,
                disable=True
                )
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # [ADDED] 건너뛸 프레임
        if frame_idx % step != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_model.process(rgb)

        if res.pose_world_landmarks:
            for i, lm in enumerate(res.pose_world_landmarks.landmark):
                pose_rows.append({
                    "frame": frame_idx // step,   # [CHANGED] TARGET_FPS 기준 프레임 번호
                    "landmark": i,
                    "x": lm.x,
                    "y": -lm.y,      # flip for unity-like coords
                    "z": lm.z,
                    "visibility": lm.visibility,
                })

        frame_idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()

    df = pd.DataFrame(pose_rows)

    # extract local positions from pelvis


    out_path = f"{dir_path}/{name}.npz"
    np.savez(out_path, data=df.to_numpy(), fps=out_fps)   # [CHANGED] 실제 시간 기준 fps도 같이 저장
    print(f"📌 Saved 3D keypoints → {out_path}")

def main():
    # url_list=["https://youtube.com/shorts/fNvA0dU5fZM?si=FFBbj_IprAQ4S_fs",
    # "https://youtube.com/shorts/rCECyHdKGnA?si=CQ_qbTmh2p0k3oJ3",
    # "https://youtube.com/shorts/GX6Tl9KtQgE?si=hNApdx7ZNOY_U0su",
    # 'https://youtube.com/shorts/Xuzazm0VAik?si=2ugZmNk6XtKHCkxe',
    # "https://youtube.com/shorts/Z5IKezdRTKU?si=n7hRZHvkP2Z809hy",
    # "https://youtube.com/shorts/-XgnHvuScP8?si=yxnOz7JfpsjHnv7q"]
    # url_list = ["https://youtube.com/shorts/vl8sBA0oIYU?si=4qZn3gt-5u2hs2YY",
    #             "https://youtube.com/shorts/xXNFSn_GU7Y?si=xh9T-XFXJxbsSA5I",
    #             "https://youtube.com/shorts/cakv_6_PTBA?si=gqxtP8QKsfpJGsoY",
    #             "https://youtube.com/shorts/N62KmR7zN0A?si=vstkCnOmTsC60D1x",
    #             "https://youtube.com/shorts/3tvVRaFbKKE?si=afZR4ymGQJWGhvRQ"
    #             ]
    #url_list=["https://youtube.com/shorts/ogufY7Ak7dU?si=QnjmZOwAR0nAhRn-"]
    # url_list=["https://youtube.com/shorts/-5LhNSMBrEs?si=2QPPzJ2rRr5jDduz",
    #           "https://youtube.com/shorts/bT2iY8IjEU0?si=oeAC9nITvjEihN4N",
    #           "https://youtube.com/shorts/pILmKQe5V94?si=NxizuiMq1_wAk0oW"]
    # url_list=["https://youtube.com/shorts/636IhyuvN-4?si=mbA9cwv1UcDXlG-_",
    #           "https://youtube.com/shorts/4Ntle88WdKU?si=CaS2ZVKJzPsHAEHr",
    #           "https://youtube.com/shorts/p8eMhRdeX4g?si=Rn84gSdaR0aLualY",]
    url_list=["https://youtube.com/shorts/xxPOJGZ9pKY?si=-3q2udr5aSSqb_AE"]
    for url in url_list:
        video_path, name = download_youtube(url)
        extract_3d_keypoints(video_path, RAW_KEYPOINT_DIR, name)
        # Optional : Diverse Types of Data

if __name__ == "__main__":
    main()