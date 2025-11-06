# Technical Portfolio
### Python, SQL, R, Java, Scala, Go, C/C++, Javascript 등 데이터 처리 언어 활용 능력
**[framework.md](./src/framework.md)** : Python 언어를 활용하여 다중 도메인을 갖는 안저 이미지에 대한 3단계 연합학습 및 협업 추론 프레임워크를 구성했습니다.
### 머신러닝 라이브러리를 이용한 재현 가능한 개발 결과물 공개 여부
**[result.md](./results/result.md)** : 다양한 머신러닝 라이브러리를 활용하여 제안하는 아이디어를 구현하여 결과를 보고하였습니다.
### R, Python, Tableau, Power BI 등을 활용한 데이터 시각화 능력
**[vis.md](./visualization/vis.md)** : 학습된 모델 결과를 통해 안저 이미지의 강조되는 부분을 파악하기 위해 시각화를 수행하였습니다.

</br>
</br>

# Abstract
당뇨망막병증을 진단하기 위한 확장가능한 다중 도메인 연합학습 및 협업 추론 파이프라인의 소스 코드입니다. 총 3단계로 구성되어 있으며, 각 단계는 다음과 같습니다:
- **Federated Self-Supervised Pretraining (FSSL)** — 병원 전체에 걸친 MAE 기반 사전 훈련
- **Adapter-based Fine-Tuning** — 훈련 가능한 매개변수가 12%에 불과한 매개변수 효율적인 개인화
- **Collaborative Label Transfer (CLT)** — 라벨이 지정되지 않았거나 늦게 가입한 기관을 위한 품질 인식 FedProto(QA-FedProto)

이 프레임워크는 다양한 임상 기관에서 **Parameter-efficient, Label-efficient, and Domain-generalized**된 당뇨병성 망막증 진단을 위해 설계되었습니다.


</br>

# Directories
- data : 안저 이미지 전처리 및 학습/테스트 로더 설계
- models : 학습, 어댑터 모델
- src: 프레임워크를 위한 3단계 구현 파일
- visualization : 안저 이미지 시각화 파일
- results : 결과 파일
 
