# 실험 결과 (Experimental Results)

본 문서는 연합 학습 기반 의료 이미지 분류 프레임워크의 실험 결과를 정리합니다.

---

## 📦 결과 파일 접근

실험 결과 파일들은 용량 문제로 Google Drive에 업로드되어 있습니다.

### 🔗 Google Drive 링크
**[실험 결과 다운로드](https://drive.google.com/drive/folders/1pu7GUgkpS_d2gmrSLnrFh35qxjpr8AfH?hl=en)**

---

## 📁 결과 디렉토리 구조

```
results/
├── pretraining/
│   └── global_model.pt              # 연합 학습으로 사전 학습된 MAE 모델
│
├── fine-tuning/
│   ├── APTOS/
│   │   ├── Adapter_FT/
│   │   │   ├── best_model.pt        # Adapter 기반 파인튜닝 모델
│   │   │   ├── metrics.json         # 학습 메트릭 (Loss, Accuracy, F1)
│   │   │   ├── confusion_matrix.json
│   │   │   └── config.yaml          # 실험 설정
│   │   └── FFT/
│   │       ├── best_model.pt        # 전체 모델 파인튜닝 (Full Fine-Tuning)
│   │       ├── metrics.json
│   │       ├── confusion_matrix.json
│   │       └── config.yaml
│   │
│   ├── ODIR/
│   │   ├── Adapter_FT/
│   │   │   └── ...
│   │   └── FFT/
│   │       └── ...
│   │
│   ├── IDRiD/
│   │   ├── Adapter_FT/
│   │   │   └── ...
│   │   └── FFT/
│   │       └── ...
│   │
│   ├── MESSIDOR/
│   │   ├── Adapter_FT/
│   │   │   └── ...
│   │   └── FFT/
│   │       └── ...
│   │
│   └── NMC/
│       ├── Adapter_FT/
│       │   └── ...
│       └── FFT/
│           └── ...
│
└── sample_dataset/
    └── images/                      # 시각화용 샘플 이미지
        ├── sample_001.png
        ├── sample_002.png
        └── ...
```

---

## 🎯 실험 개요

### 1️⃣ Pretraining (사전 학습)

**목적**: 여러 의료 기관의 분산된 데이터를 활용한 연합 학습 기반 MAE 사전 학습

**결과물**: `pretraining/global_model.pt`

**주요 특징**:
- **알고리즘**: FedAvg (Federated Averaging)
- **참여 클라이언트**: 5개 (NMC, APTOS, ODIR, IDRiD, MESSIDOR)
- **학습 방식**: Self-supervised Learning (Masked Autoencoder)
- **프라이버시 보호**: 원본 데이터 공유 없이 모델 파라미터만 집계

**모델 상세**:
- **Architecture**: Vision Transformer (ViT) 기반 MAE
- **Image Size**: 224×224
- **Patch Size**: 16×16
- **Embedding Dimension**: 768
- **Number of Layers**: 12 (Encoder), 8 (Decoder)
- **Attention Heads**: 12 (Encoder), 16 (Decoder)

---

### 2️⃣ Fine-tuning (파인튜닝)

**목적**: 사전 학습된 모델을 각 데이터셋의 질병 분류 작업에 특화

#### 📊 데이터셋별 결과

각 데이터셋에 대해 두 가지 파인튜닝 전략을 비교:

| 데이터셋 | 질병 종류 | 클래스 수 | Train/Test Split |
|---------|----------|----------|------------------|
| **NMC** | 당뇨망막병증 | 2 (정상/병변) | 75% / 25% |
| **APTOS** | 당뇨망막병증 | 2 (정상/병변) | 75% / 25% |
| **ODIR** | 당뇨망막병증 | 2 (정상/병변) | 75% / 25% |
| **IDRiD** | 당뇨망막병증 | 2 (정상/병변) | 75% / 25% |
| **MESSIDOR** | 당뇨망막병증 | 2 (정상/병변) | 75% / 25% |

---

## 🔬 파인튜닝 전략 비교

### Strategy 1: Adapter-based Fine-Tuning (Adapter_FT)

**특징**:
- 인코더 동결 (Frozen Encoder)
- 경량 Adapter 레이어 추가
- 경량 Adapter 레이어와 Classification Head 학습

**장점**:
- ✅ 파라미터 효율적 (학습 파라미터 수 최소화)
- ✅ 빠른 학습 속도
- ✅ 과적합 방지
- ✅ 저자원 환경에 적합

**구조**:
```
Pretrained MAE Encoder (Frozen)
    ↓
Adapter Layer (Trainable)
    ↓
Global Average Pooling
    ↓
Classification Head (Trainable)
```

---

### Strategy 2: Full Fine-Tuning (FFT)

**특징**:
- 전체 모델 학습 (End-to-End)
- 인코더 가중치 업데이트
- 모든 레이어 미세 조정

**장점**:
- ✅ 데이터셋에 최적화된 표현 학습
- ✅ 높은 표현력
- ✅ 복잡한 패턴 학습 가능

**구조**:
```
Pretrained MAE Encoder (Trainable)
    ↓
Global Average Pooling
    ↓
Classification Head (Trainable)
```

---

## 📈 성능 메트릭

각 모델의 성능은 다음 메트릭으로 평가됩니다:

### 주요 지표

1. **Accuracy (정확도)**
   - 전체 샘플 중 올바르게 예측한 비율
   - `(TP + TN) / (TP + TN + FP + FN)`

2. **F1-Score**
   - Precision과 Recall의 조화 평균
   - 클래스 불균형에 강건한 지표
   - `2 × (Precision × Recall) / (Precision + Recall)`

3. **Confusion Matrix**
   - 실제 라벨 vs 예측 라벨 분포
   - True Positive, False Positive, True Negative, False Negative

4. **Training Loss**
   - 에포크별 학습 손실 변화
   - 수렴 양상 분석

---

## 🖼️ Sample Dataset

### 목적
시각화 및 정성적 분석을 위한 대표 샘플 이미지 모음

### 경로
`sample_dataset/images/`

### 특징
- 다양한 질병 중증도를 포함한 샘플
- 예측 불일치 케이스 포함 (Adapter_FT vs FFT 예측 차이)
- XGradCAM 및 Attention Rollout 시각화에 사용

### 샘플 선정 기준
1. **대표성**: 각 클래스의 전형적인 케이스
2. **다양성**: 경증, 중등도, 중증 병변 포함
3. **흥미로운 케이스**: 
   - 모델 간 예측 불일치
   - 어려운 경계 케이스
   - 모델이 올바른/틀린 주목 영역

### 활용
- [Visualization](../visualization/vis.md) 문서 참조
- `visualization/cam.ipynb`에서 사용

---

## 💾 다운로드 및 사용

### Google Drive에서 다운로드

1. **전체 다운로드**
   - [링크](https://drive.google.com/drive/folders/1pu7GUgkpS_d2gmrSLnrFh35qxjpr8AfH?hl=en) 접속
   - 최상단 폴더 우클릭 → "다운로드"

2. **선택적 다운로드**
   - 필요한 모델만 개별 다운로드 가능
   - 예: `fine-tuning/APTOS/Adapter_FT/` 폴더만 다운로드

### 로컬 경로 설정

다운로드 후 프로젝트에서 사용 시 경로 설정:

```yaml
# configs/aptos.yaml
MODEL_PATH: '/path/to/downloaded/pretraining/global_model.pt'
```

또는 코드에서 직접 지정:

```python
model_path = '/Users/username/Downloads/results/fine-tuning/APTOS/Adapter_FT/best_model.pt'
model = torch.load(model_path, map_location='cuda:0')
```

---

## 🔗 관련 문서

- **프레임워크 구조**: [../src/framework.md](../src/framework.md)
- **시각화 가이드**: [../visualization/vis.md](../visualization/vis.md)
- **프로젝트 README**: [../README.md](../README.md)

