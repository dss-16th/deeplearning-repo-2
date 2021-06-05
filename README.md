# Predict Alzheimer MRI Image Classification 

deeplearning project

***
## 1. INTRO

보건복지부 결과에 따르면 알츠하이머 치매 유병률과 환자가 급증하고 있어 알츠하이머 치매 조기발견이 중요해지고 있습니다.

현재 의료진은 MRI 이미지 판독 시 의료진마다 판단 기준이 달라 더욱 정확한 mri 이미지 판독이 필요합니다.

</br>

### 1-1 Purpose
- Kaggle에서 제공한 알츠하이머 치매 MRI image 데이터를 활용한 딥러닝 프로젝트입니다.
- 본 프로젝트에서는 Alzheimer와 Non-Alzheimer를 구분하였습니다.
- 본 프로젝트의 목적은 DEEP:PHI를 사용하여 두가지 전처리 방법과 하이퍼 파라미터 튜닝을 통해 MRI 이미지 분류 CNN 모델 성능 향상입니다. 
- 본 프로젝트는 Alzheimer 환자 조기 발견 보조 지표로 사용하는 것을 기대합니다.


</br>

### 1-2. Result

### 1-3. DEEP:PHI


### 1-4 데이터

#### 1. 출처
  - kaggle : Alzheimer_binaryclassification
  - 링크1 : https://www.kaggle.com/smiti14/alzheimer-binaryclassification
  - 링크2 : https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images
  
#### 2. 데이터 구성
- Alzheimer : 6,160 (6,160)
- Non_Alzheimer : 6,393 (6,393)

#### 3. 데이터 정보

- 뇌 MRI (뇌를 가로횡당면으로 자른 MRI 데이터)
- IMAGE
  - shape	: [200,176}
  - Color Mode	: Gray(channel 1)
  - Dimension	: 2D
  - Pixel Value Range	: [0,252]
  - Number of Color Channel	: 1
  - Data Type	: unit8 (0 or positive)
- label
  - class Index  : 0/1
  - class Name : Alzheimer /Non_Alzheimer 
- Train/Validation/Test
  - 80% / 10% / 10%

<br/>

### 1-5 팀원 / 역할

- [류승환]
  -  https://github.com/
  - 전처리 기법 중 Merge를 Overlap 방식과 Multiply 방식 비교
  - Multiply방식의 Merge를 이용한 모델들간의 성능 비교
  - 모델관련 하이퍼 파라미터 및 전체적인 파이프라인 논의
- [이주영]
  - https://github.com/leekj3133
  - **팀장**
  - 전처리 기법 Invert와 VGG19모델을 이용하여 알츠하이머 MRI Image Data 예측 모델 구축
  - 모델관련 하이퍼 파라미터 및 전체적인 파이프라인 논의


*****

<br/>




## 3. PROCESS

<br/>

### 3-1 PreProcessing

#### 1. Invert

#### 2. Merge

### 3-2 모델 

### 3-3
****************************
## 4. CONCLUSION
- VGG19모델에서 전처리 기법인 Invert를 활용한 연구방안을 제언
- 사용했던 모델 뿐 아니라 Multiply Merge의 특징을 더 부각시킬 수 있는 모델을 찾아 성능을 개선시키는 연구를 제언
*********

## 5. comment & limitations
- 시간과 GPU가 부족해 원하는 많은 모델을 학습시키는데 한정적
- Train Result와 초기 입력한 Parameter가 저장되지 않아(파이프라인을 다시 실행할 경우) 추가 저장하지 않으면 유실될 가능성 있음
- Test Result를 DEEP:PHI 내에서만 결과를 볼 수 있어 불편
- 데이터 출처에 대한 정확한 정보가 없기 때문에 이를 알아보기 위해 discussion을 찾아 보았으며, 이전에 각종 질문이 있었으나 정작 원작자 본인으로부터 답변을 받지 못함. (신뢰성 부족)
- 출처가 정확한 데이터가 필요하였고, ADNI 사이트에서 신청하였지만 계속해서 승인대기중인 상태. 

