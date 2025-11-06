# Visualization: 모델 해석 가능성 분석

본 디렉토리는 파인튜닝된 모델의 예측 과정을 시각화하고 해석하기 위한 Class Activation Map(CAM) 분석 도구를 제공합니다.

---

## 📁 디렉토리 구조

```
visualization/
├── cam.ipynb              # CAM 시각화 메인 노트북
├── configs.yaml           # 시각화 설정 파일
├── vis.md                 # 본 문서
└── cam_results/           # 시각화 결과 저장 디렉토리
    └── integrated_comparison.png
```

---

## 🎯 목적

파인튜닝된 모델이 **어떤 영역**에 집중하여 예측을 수행하는지 시각적으로 분석합니다:

1. **모델 비교**: Adapter 기반 파인튜닝 vs 전체 모델 파인튜닝
2. **예측 신뢰도 검증**: 모델이 올바른 병변 영역을 보고 있는지 확인
3. **오류 분석**: 잘못된 예측 시 모델의 주목 영역 분석

---

## 🔍 시각화 방법

### 1️⃣ XGradCAM (Adapter 기반 모델)

#### 특징
- Gradient 기반의 CAM 방법
- 채널별 중요도를 가중 평균하여 활성화 맵 생성
- **Adapter 레이어**의 출력에 적용

#### 적용 대상
- **Adapter 기반 파인튜닝 모델**
- 마지막 Adapter 레이어를 타겟 레이어로 설정
- 경량 파인튜닝에서 추가된 레이어가 어디를 보는지 분석

#### 장점
- 특정 레이어의 영향력을 직접적으로 확인
- 높은 공간 해상도 유지
- 예측 클래스에 대한 국소적 중요 영역 강조

---

### 2️⃣ Attention Rollout (전체 모델 파인튜닝)

#### 특징
- Transformer의 Self-Attention을 활용
- 모든 레이어의 Attention을 곱하여 누적(Rollout)
- **CLS 토큰**이 이미지 패치에 얼마나 집중하는지 측정

#### 적용 대상
- **전체 모델 파인튜닝 (Full Fine-Tuning)**
- MAE Encoder의 모든 Transformer 블록에서 Attention 추출

#### 장점
- 모델의 전역적 주목 패턴 파악
- ViT 기반 모델의 고유한 특성 활용
- 레이어 간 정보 흐름 추적 가능

---

## 🛠️ 주요 구성 요소

### `cam.ipynb`

전체 시각화 파이프라인을 포함한 Jupyter 노트북

#### 📌 주요 섹션

**1. 데이터 준비**
- APTOS 당뇨망막병증 데이터셋 로드
- Ground Truth 라벨 매핑

**2. 모델 로드**
```python
# 사전 학습된 MAE 인코더
mae_encoder = MaskedAutoencoderViT(...)

# Adapter 기반 파인튜닝 모델
adapter_model = torch.load(adapter_model_path)

# 전체 모델 파인튜닝 모델
direct_model = torch.load(direct_model_path)
```

**3. 래퍼 클래스**
- `EncoderAdapterWrapper`: XGradCAM을 위한 모델 래핑
- `MAEEncoder`: Attention Rollout을 위한 인코더 래핑

**4. 시각화 함수**
- `get_xgradcam_heatmap()`: XGradCAM 히트맵 생성
- `get_attention_rollout_heatmap()`: Attention Rollout 히트맵 생성
- `apply_heatmap_overlay()`: 원본 이미지에 히트맵 오버레이

**5. 통합 비교**
- 각 샘플에 대해 3가지 뷰 생성:
  1. **원본 이미지** (Ground Truth 포함)
  2. **Attention Rollout** (전체 파인튜닝 예측)
  3. **XGradCAM** (Adapter 기반 예측)

---

### `configs.yaml`

시각화에 필요한 모든 설정 정보

#### 주요 설정

```yaml
# 데이터셋 정보
DATASET: 'APTOSDataset'
IMG_DIR: "/path/"
CSV_DIR: "/path/label.csv"

# 모델 설정
IMG_SIZE: 224
PATCH_SIZE: 16
NUM_CLASSES: 2

# 정규화 파라미터
MEAN: [0.4818, 0.2620, 0.0985]
STD: [0.2379, 0.1371, 0.0576]

# 모델 경로
MODEL_PATH: '/path/to/global_model.pt'
```

---

## 📊 시각화 결과 해석

### 결과 이미지 구조

```
┌─────────────┬──────────────────────┬─────────────────────┐
│   Original  │  Attention Rollout   │      XGradCAM       │
│   GT: 0/1   │    Pred: 0/1 (Full)  │  Pred: 0/1 (Adapter)│
├─────────────┼──────────────────────┼─────────────────────┤
│  샘플 1     │   전체 FT 시각화     │  Adapter FT 시각화  │
│  샘플 2     │         ...          │        ...          │
│    ...      │         ...          │        ...          │
└─────────────┴──────────────────────┴─────────────────────┘
```

---

## 🚀 사용 방법

### 1. 환경 설정

```bash
pip install torch torchvision
pip install pytorch-grad-cam
pip install opencv-python matplotlib pyyaml
```

### 2. 설정 파일 준비

`configs.yaml`에서 다음 경로들을 설정:
- 데이터셋 이미지 디렉토리
- CSV 라벨 파일
- 사전 학습 모델 경로
- Adapter 모델 경로
- Direct 파인튜닝 모델 경로

### 3. 노트북 실행

```bash
jupyter notebook cam.ipynb
```

또는 Jupyter Lab:

```bash
jupyter lab cam.ipynb
```

### 4. 결과 확인

`cam_results/integrated_comparison.png`에서 통합 비교 결과 확인

---

## 📚 참고 문헌

### CAM 관련 논문

1. **CAM**: Zhou et al., "Learning Deep Features for Discriminative Localization" (CVPR 2016)
2. **GradCAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (ICCV 2017)
3. **GradCAM++**: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-Based Visual Explanations" (WACV 2018)
4. **Attention Rollout**: Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (ACL 2020)

---

## 🔗 관련 리소스

- **PyTorch GradCAM Library**: https://github.com/jacobgil/pytorch-grad-cam
- **본 프로젝트 프레임워크**: [../src/framework.md](../src/framework.md)

---