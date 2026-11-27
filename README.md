# 3D Motion Data Post-Processing
중앙대학교 예술공학부 20244817 김지원

유튜브 춤 영상에서 MediaPipe BlazePose로 3D 관절 좌표를 추출하고,
인공지능과 머신러닝 수업에서 제공된 후처리에 **7단계 개선 후처리**를 더해 모션 품질을 높인 과제입니다.

- 영상: REDRED 챌린지 (30fps, 535프레임, 약 17.8초)
- 결과 요약: 떨림 **43.1 → 28.9 mm/frame² (33% 감소)**, 뼈 길이 변동 **17.5% → 0%**, 지면 오차 **781 → 0 mm**

---

## 📁 폴더 구조

```
3D-Motion-Data-Post-Processing/
├── README.md                     ← 지금 보고 있는 설명서
├── requirements.txt              ← 필요한 라이브러리 목록
├── scripts/
│   ├── p1_create_raw_keypoints.py   ① 영상 다운로드 + 3D 관절 추출
│   ├── p1_2_post_processing.py      ② 후처리 + 전후 비교 그래프  ★ 이번 과제 핵심
│   ├── p2_create_dataset.py         (기말 프로젝트용: 데이터셋 생성)
│   ├── p3_train_mlp.py              (기말 프로젝트용: 분류 모델 학습)
│   └── p4_predict.py                (기말 프로젝트용: 동작 예측)
├── utils/
│   └── viser_test.py             ← 3D 모션 재생 화면 (수업 제공 코드)
└── data/
    └── raw_keypoints/
        └── redred.npz            ← 미리 추출해 둔 관절 데이터 (바로 실행용)
```

---

## ▶ 실행 방법

### 0. 준비물
- Python **3.9 ~ 3.11** (개발 환경: Windows, Python 3.9)
- 인터넷 (라이브러리 설치, 영상 다운로드 시)

### 1. 코드 내려받기
```bash
git clone https://github.com/kjw06171/3D-Motion-Data-Post-Processing.git
cd 3D-Motion-Data-Post-Processing
```
또는 GitHub 화면의 **Code → Download ZIP**으로 받아서 압축을 풀어도 됩니다.

### 2. 라이브러리 설치
```bash
pip install -r requirements.txt
```
> conda를 쓴다면 먼저 환경을 만들어 주세요.
> `conda create -n motion python=3.9` → `conda activate motion` → 위 명령 실행

### 3-A. 빠른 실행 (추천, 약 1분)
미리 추출해 둔 `data/raw_keypoints/redred.npz`로 **후처리만** 바로 실행합니다. 영상 다운로드가 필요 없습니다.
```bash
python scripts/p1_2_post_processing.py
```

### 3-B. 처음부터 실행 (영상 다운로드부터, 수 분)
1. `scripts/p1_create_raw_keypoints.py`의 `url_list`에 유튜브 주소를 넣습니다.
2. 순서대로 실행합니다.
```bash
python scripts/p1_create_raw_keypoints.py    # 영상 다운로드 + 관절 추출 → data/raw_keypoints/*.npz
python scripts/p1_2_post_processing.py       # 후처리 + 그래프
```

---

## 📊 실행하면 생기는 결과

| 위치 | 파일 | 내용 |
|---|---|---|
| 터미널 | — | 선택된 스무딩 필터, 단계별 처리 기록, 지표 표 |
| `data/processed/` | `*_target.csv` | 수업 후처리 결과 (Unity용) |
| | `*_refined.csv` | **개선 후처리 결과** (Unity용) |
| `data/figures/` | `*_problems.png` | 원본 데이터의 문제 4가지 |
| | `*_compare.png` | 후처리 전후 비교 그래프 + 지표 막대 |
| | `*_skeleton.png` | 같은 프레임을 원본·수업·개선 방법으로 그린 그림 |
| | `*_filters.png` | 스무딩 필터 4종 비교 |
| | `*_metrics.csv`, `*_filters.csv` | 지표 숫자 |

### 3D 모션 재생 화면 (viser)
실행이 끝나면 터미널에 아래와 같은 주소가 나옵니다.
```
viser (listening *:8080)
HTTP | http://localhost:8080
```
1. 웹 브라우저 주소창에 **`http://localhost:8080`**을 입력합니다. (숫자는 터미널에 나온 **HTTP 줄**을 따르세요)
2. 개선 후처리된 모션이 반복 재생됩니다. 마우스로 돌려 볼 수 있습니다.
3. 다 봤으면 터미널에서 **`Ctrl + C`**를 눌러 종료합니다.

---

## ⚙ 설정 (`scripts/p1_2_post_processing.py` 맨 위)

| 설정 | 기본값 | 설명 |
|---|---|---|
| `VISUALIZE` | `True` | viser 재생 화면 띄우기. `False`면 그래프만 만들고 바로 종료 |
| `COMPARE` | `True` | `data/figures/`에 비교 그래프·지표 저장 |
| `AUTO_SMOOTH` | `True` | 교차검증으로 스무딩 필터 자동 선택 |
| `HAND_STABILIZE` | `True` | 손 안정화 적용 |
| `PALM_FLIP` | `False` | 손바닥 방향을 손마다 180° 뒤집기 |

---

## 🎮 Unity에서 보기

수업에서 제공된 Unity 프로젝트(`BlazeDataExtraction`)를 사용합니다.
1. Mixamo 캐릭터를 `Assets/Character`에 넣고, **Rig → Humanoid → Apply**, **Materials → Extract Textures**
2. 캐릭터를 Scene에 놓고 **Add Component → Actor**, `Skeleton` 버튼으로 뼈대 확인
3. `Blaze Data Extraction`의 **Source Actor**에 캐릭터를 연결
4. **CSV Loader**로 `data/processed/` 폴더를 불러오고 `*_refined.csv` 선택 → Play

> 파일 이름에 한글·이모지가 있으면 Unity가 못 읽을 수 있으니 영어로 바꿔서 사용하세요.

---

## 🔧 수업 후처리 (Provided)

`create_height_corrected_target` (원본 코드 그대로 유지)
매 프레임 발끝(31·32번) 중 낮은 점을 y=0으로 맞추고, 0보다 낮은 값은 0으로 자름(clamp).
→ 지면 문제는 해결하지만, 떨림이 오히려 증가(43.1 → 46.8)하고 뒤꿈치가 잘려 발이 납작해짐(전체 프레임의 6.4%).

## 🛠 개선 후처리 7단계 (Refined)

| 단계 | 사용한 기법 | 하는 일 |
|---|---|---|
| 1. 빠진 프레임 채우기 | 선형 보간 | 사람이 안 잡힌 프레임을 앞뒤 값으로 메움 |
| 2. 좌우 바뀜 되돌리기 | 직전 프레임과 거리 비교 | 좌우를 바꿨을 때 2배 더 가까우면 교체 |
| 3. 튀는 값 지우기 | MAD 이상치 탐지 + 신뢰도 | 속도 급변·신뢰도 < 0.3인 점을 보간 |
| 4. 떨림 줄이기 | Savitzky-Golay 필터 | 필터 4종 × 세기 6단계를 교차검증해 자동 선택 |
| 5. 뼈 길이 고정 | 중앙값 길이 + 좌우 평균 | 뼈 방향은 유지, 길이만 고정 |
| 6. 손 안정화 | 팔뚝 방향 벡터 + 스무딩 | 손을 팔뚝 따라 곧게, 손바닥 방향은 약 1초 단위로 |
| 7. 바닥 맞추기 | 발 관절 6개의 최저점 | 몸 전체를 올림 (자르지 않음) |

원본 함수는 수정하지 않았고, 추가한 코드에는 모두 `[ADDED]` 표시가 있습니다.

### 스무딩 필터 선택 (교차검증)
실제 영상에는 정답이 없으므로, 한 프레임씩 숨기고 필터가 그 프레임을 얼마나 잘 맞히는지로 채점했습니다.
정답을 아는 합성 데이터에서 실제 오차와의 순위 상관 0.98로 검증했습니다.

| 필터 | 최적 세기 | 교차검증 오차 (mm) |
|---|---|---|
| **Savitzky-Golay** | 5 | **31.71** (선택) |
| Butterworth | 10 Hz | 31.77 |
| Moving average | 3 | 32.77 |
| One-Euro | 4 Hz | 50.93 |

## 📈 결과

| 지표 | 원본 | 수업 후처리 | 개선 후처리 |
|---|---|---|---|
| 떨림 (mm/frame²) ↓ | 43.10 | 46.83 | **28.89** |
| 뼈 길이 변동 (%) ↓ | 17.47 | 17.69 | **0.00** |
| 지면 오차 (mm) ↓ | 781.41 | 0.00 | **0.00** |
| clamp된 프레임 (%) | 0 | 6.36 | 0 |
| 원본 대비 변화 (mm) | 0.00 | 0.05 | 40.22 |

## ⚠ 한계

- 골반 중심 좌표라 점프·이동 경로는 표현하지 못함
- 손 점이 3개뿐이라 손바닥이 실제와 같은 방향인지는 보장하지 못함
- 깊이(z) 오차로 팔이 몸을 통과하는 장면이 남음

---

## ❓ 문제 해결

| 증상 | 해결 |
|---|---|
| `Found 0 raw keypoint files` | `data/raw_keypoints/`에 `.npz`가 있는지 확인 (3-B로 추출하거나 `redred.npz` 사용) |
| `ModuleNotFoundError: utils` | `utils/viser_test.py`가 있는지 확인. 없으면 `VISUALIZE = False`로 실행 |
| `module 'mediapipe' has no attribute 'solutions'` | `pip install mediapipe==0.10.14` |
| 실행이 끝나지 않음 | viser 재생 화면이 켜져 있는 상태입니다. `Ctrl + C`로 종료 |
| 그래프 제목의 한글이 □로 보임 | 한글 폰트(맑은 고딕 등)가 없는 환경입니다. 결과에는 영향 없음 |
