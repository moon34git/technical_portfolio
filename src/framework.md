# Framework 구조

본 프로젝트는 의료 이미지 분류를 위한 연합 학습(Federated Learning) 기반 프레임워크입니다. 세 가지 주요 단계로 구성되어 있습니다.

---

## 📁 디렉토리 구조

```
src/
├── pretraining/     # 연합 학습 기반 사전 학습
├── finetuning/      # 개별 데이터셋 파인튜닝
└── clt/             # 협업 테스팅/추론
```

---

## 1️⃣ Pretraining (사전 학습)

### 📌 목적
여러 클라이언트(의료 기관)가 보유한 분산된 데이터셋을 활용하여 Masked Autoencoder (MAE) 모델을 연합 학습 방식으로 사전 학습합니다.

### 🔑 주요 구성 요소

#### `main.py`
- 연합 학습 프로세스의 진입점
- YAML 설정 파일을 로드하고 Server 객체를 초기화
- 전체 학습 프로세스를 관리

#### `server.py`
- **FedAvg** 알고리즘을 구현한 중앙 서버
- 클라이언트 선택 및 글로벌 모델 집계(Aggregation) 수행
- 선택적으로 **FedBN**(Federated Batch Normalization) 지원
- 주요 기능:
  - `set_clients()`: 여러 데이터셋(NMC, APTOS, ODIR, IDRiD, MESSIDOR)에 대한 클라이언트 생성
  - `select_clients()`: 라운드마다 참여할 클라이언트 샘플링
  - `aggregate_parameters()`: 클라이언트 모델의 가중치를 평균화하여 글로벌 모델 업데이트
  - `save_model()`: 글로벌 모델 및 BN 파라미터 저장

#### `client.py`
- 각 클라이언트의 로컬 학습 로직 구현
- MAE 모델을 사용한 자기 지도 학습(Self-supervised Learning)
- 주요 기능:
  - `load_local_data()`: 클라이언트별 데이터 로드
  - `train()`: 로컬 에포크 동안 MAE 손실 최소화
  - `adjust_learning_rate()`: Warmup 및 Cosine Annealing 스케줄러
  - Mixed Precision Training (AMP) 지원

#### `configs.yaml`
- 전체 학습 설정 (모델 구조, 하이퍼파라미터, 데이터셋 경로 등)

---

## 2️⃣ Finetuning (파인튜닝)

### 📌 목적
사전 학습된 MAE 인코더를 특정 데이터셋의 분류 작업에 맞게 파인튜닝합니다.

### 🔑 주요 구성 요소

#### `finetuning.py`
- 사전 학습된 MAE 모델 로드
- 분류 헤드를 추가하여 지도 학습(Supervised Learning) 수행
- 주요 기능:
  - **프리트레인 모델 로드**: 글로벌 모델 또는 클라이언트별 BN 파라미터 로드
  - **Freeze 옵션**: 인코더를 고정하거나 파인튜닝 가능
  - **Adapter 지원**: `FineTunedMAE_Shallow` 사용 시 경량 어댑터 레이어 추가
  - **평가**: Accuracy, F1-score, Confusion Matrix 계산
  - **Best Model 저장**: 테스트 정확도가 가장 높은 모델 저장

#### `configs/`
- 각 데이터셋별 설정 파일 (aptos.yaml, idrid.yaml, messidor.yaml, nmc.yaml, odir.yaml)
- 데이터셋 경로, 학습률, 배치 크기, 에포크 수 등 정의

#### `scripts.sh`
- 여러 데이터셋에 대한 파인튜닝을 자동화하는 쉘 스크립트

---

## 3️⃣ CLT (Collaborative Testing/Inference)

### 📌 목적
여러 클라이언트 모델을 활용하여 협업 추론을 수행하고, 더 정확한 예측을 도출합니다.

### 🔍 두 가지 협업 추론 방법

---

### 3-1. Classifier-based Collaborative Inference

#### `classifier_based.py`
엔트로피 기반 필터링과 투표(Voting)를 통한 협업 추론

**주요 과정:**

1. **Feature & Entropy 추출**
   - 각 클라이언트 모델이 테스트 샘플의 특징과 예측 엔트로피 계산
   
2. **Entropy Threshold 계산**
   - 각 클라이언트별로 엔트로피 백분위수(percentile) 기반 임계값 설정
   
3. **샘플 필터링**
   - 정책(policy)에 따라 샘플 선택:
     - `strict`: 모든 클라이언트가 낮은 엔트로피를 가진 샘플만 선택
     - `majority`: 과반수 클라이언트가 통과한 샘플 선택
     - `relaxed`: 최소 개수(`min_count`) 이상 클라이언트가 통과한 샘플 선택
   
4. **Voting**
   - 필터링된 샘플에 대해 클라이언트 모델들의 다수결 투표로 최종 예측

**지원 시나리오:**
- `unlabeled`: 새로운 데이터셋(테스트 데이터만)
- `late-joining`: 기존 데이터 + 테스트 데이터 (전체 데이터)

**사용 알고리즘:**
- FedAvg, FedRep, FedProto 등

---

### 3-2. Prototype Distance-based Collaborative Inference

#### `prototype_distance_based.py`
글로벌 프로토타입(Global Prototype)과의 거리 기반 협업 추론

**주요 과정:**

1. **Feature 추출 & Distance 계산**
   - 각 클라이언트 모델이 특징 벡터 추출
   - 글로벌 프로토타입(각 클래스의 대표 특징)과의 L2 거리 계산
   
2. **Z-score 정규화**
   - 각 클라이언트별로 거리 분포를 Z-score로 정규화
   - 클라이언트 간 스케일 차이 보정
   
3. **Z-score Threshold 필터링**
   - 임계값(`ZETA`) 이하의 샘플만 선택
   - 프로토타입에 가까운 확신 있는 샘플만 유지
   
4. **Consensus 찾기**
   - 여러 클라이언트가 동일한 클래스로 예측한 샘플 추출
   - 최소 클라이언트 수(`KAPPA`)를 만족하는 샘플만 선택
   
5. **Z-score Margin Refinement**
   - 두 클래스 간 Z-score 차이(margin)가 큰 샘플만 최종 선택
   - 상위 백분위(`BETA`) 이상의 마진을 가진 샘플만 유지

**지원 시나리오:**
- `unlabeled`: 새로운 데이터셋(테스트 데이터만)
- `late-joining`: 기존 데이터 + 테스트 데이터

---

### 3-3. Federated Learning Algorithms

#### `federated/`
여러 연합 학습 알고리즘을 구현한 서브디렉토리

**알고리즘:**

1. **FedAvg** (`serveravg.py`, `clientavg.py`)
   - 가장 기본적인 연합 학습 알고리즘
   - 모든 클라이언트의 모델 파라미터를 가중 평균
   
2. **FedRep** (`serverrep.py`, `clientrep.py`)
   - Representation과 Head를 분리
   - 글로벌 Representation 공유, 클라이언트별 개인화된 Head 유지
   
3. **FedProto** (`serverproto.py`, `clientproto.py`)
   - 프로토타입 기반 연합 학습
   - 각 클래스의 프로토타입(대표 특징)을 공유하여 학습

#### `federated/main.py`
- 연합 학습 알고리즘 선택 및 실행
- `--cfg` 인자로 알고리즘별 설정 파일 지정

#### `federated/configs/`
- 각 알고리즘별 설정 파일 (fedavg.yaml, fedrep.yaml, fedproto.yaml)

---

## 📊 지원 데이터셋

- **NMC**: 당뇨망막병증 데이터셋
- **APTOS**: Kaggle APTOS 2019 당뇨망막병증 데이터
- **ODIR**: 안저 질환 데이터
- **IDRiD**: 인도 당뇨망막병증 데이터
- **MESSIDOR**: 프랑스 당뇨망막병증 데이터

---

## 🛠️ 사용 방법

### 1. 사전 학습
```bash
cd src/pretraining
python main.py --cfg configs.yaml
```

### 2. 파인튜닝
```bash
cd src/finetuning
python finetuning.py --cfg configs/nmc.yaml
# 또는 scripts.sh 실행
bash scripts.sh
```

### 3. 협업 추론

#### Classifier-based
```bash
cd src/clt
python classifier_based.py --config configs/classifier_based_nmc.yaml --device cuda:0
```

#### Prototype Distance-based
```bash
cd src/clt
python prototype_distance_based.py --config configs/prototype_distance_nmc.yaml --device cuda:0
```

#### Federated Learning
```bash
cd src/clt/federated
python main.py --cfg configs/fedavg.yaml
```

---

## 📝 주요 하이퍼파라미터

### Pretraining
- `GLOBAL_ROUNDS`: 연합 학습 라운드 수
- `JOIN_RATIO`: 각 라운드에 참여하는 클라이언트 비율
- `MASK_RATIO`: MAE 마스킹 비율
- `USE_FEDBN`: FedBN 사용 여부

### Finetuning
- `FREEZE`: 인코더 고정 여부
- `ADAPFT`: Adapter 사용 여부
- `ENCODER_LR`: 인코더 학습률
- `EPOCHS`: 파인튜닝 에포크 수

### CLT (Classifier-based)
- `ETA`: 엔트로피 필터링 백분위수
- `KAPPA`: 최소 동의 클라이언트 수
- `policy`: 필터링 정책 (strict/majority/relaxed)

### CLT (Prototype Distance-based)
- `ZETA`: Z-score 임계값
- `BETA`: Z-score margin 상위 백분위
- `KAPPA`: 최소 합의 클라이언트 수

---

## 📚 참고 문헌

이 프레임워크는 다음 연구들을 기반으로 구현되었습니다:
- **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- **FedBN**: Li et al., "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization"
- **MAE**: He et al., "Masked Autoencoders Are Scalable Vision Learners"
- **FedProto**: Tan et al., "FedProto: Federated Prototype Learning across Heterogeneous Clients"
- **FedRep**: Collins et al., "Exploiting Shared Representations for Personalized Federated Learning"

