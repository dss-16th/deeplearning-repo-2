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
  -
- [이주영]
  - https://github.com/leekj3133
  - 팀장, 전처리 기법 Invert와 VGG19모델을 이용하여 알츠하이머 MRI Image Data 예측 모델 구축


*****

<br/>




## 3. PROCESS

<br/>

### 3-1 PreProcessing

#### 1. Invert

#### 2. Merge

### 3-2 모델 

### 3-3

## 4. CONCLUSION

## 5. comment & limitations


