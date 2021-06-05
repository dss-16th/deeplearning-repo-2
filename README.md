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

#### 4. 데이터 판단 기준
- 알츠하이머 치매 환자의 대표적인 증상:  뇌실의 크기 변화와 대뇌피질의 수축 
- 대표적인 증상을 기준으로 판단

#### 5. 성능 지표
- sensitivity :  의료 데이터의 중요 지표 
- Accuracy:  얼마나 정확하게 분류를 하는지 판단하는 지표

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
  - 전처리 기법 Invert를 이용한 모델들간의 성능 비교
  - 모델관련 하이퍼 파라미터 및 전체적인 파이프라인 논의


*****

<br/>




## 2. PROCESS

<br/>

### 2-1 PreProcessing

#### 1. Invert

</br>

#### Conclusion

- Invert 전처리 기법을 사용한 경우 기존 모델보다 성능 좋음
- Optimizer에서 SGD를 사용한 모델이 Adam을 사용한 모델보다 성능 좋음
- Drop Out를 사용한 모델은 사용하지 않은 모델보다 성능 좋음
- Batch Size 크기가 작은 모델은 크기가 큰 모델보다 성능 좋음

	=> 알츠하이머 예측을 주제로 한 의료 데이터(MRI)에 전처리 Invert 관련 논문 부족한 실정

	=> 전처리 Invert를 활용한 관련 연구 방안을 제언함

#### 1-1. 특징
- 중요한 영역이 주변보다 어두운 경우, 반전된 이미지를 이용하여 feature 추출이 용이하도록 함
- 이미지 상에서 검은색 부위를 하얀색부분으로 하얀색부분을 검은색 부분으로 반전시킴

</br>

#### 1-2. 사용 이유

- 다수의 논문에서 조영술이나 x-ray Image에 Invert를 적용한 사례가 있음
- but,  Mri 적용사례는 적어 알츠하이머 데이터에 적용시킴.

- 뇌실의 크기와 뇌실의 주름색이 배경색과 같으므로 detection할때 방해가 있을거라 판단
- 대뇌피질의 수축정도를 이용해서 학습시키면 성능이 올라갈거라 판단하여 사용

#### 2. Merge

</br>

### 2-2 모델 

</br>

#### 1. VGG19

</br>

#### 1-1. 구조

</br>

layer층 
- 활성화 함수로 ReLU를 사용하는 3by3 convolutional filter와 2x2 max pooling filter로  구성
- 입력된 이미지가 이 layer층을 지나면 이미지 사이즈는 1/2 줄어들고, depth는 깊어짐
 이 때, 그림과 같이 depth가 64, 128, 256, 512로 점점 깊어져 가는 형태
flatten 층
- 1차원으로 변환되어 Dense층 거침
Dense 층
- 활성화함수로 Softmax 사용 ,dense size는 4096인 두개의 dense 층
- 이 모델이 만들어질 때 사용했던 데이터인 ImageNet의 class의 개수인 1000이 Dense size로 설정

</br>

#### 1-2. 특징 및 파라미터



<br/>

#### 1-2-1. 특징
 - 네트워크의 깊이가 성능에 미치는 영향을 보기 위해 만들어진 모델
 - 3by3 의 작은 필터를 사용하여 발생하는 파라미터 개수를 줄임.
 ex) 5x5변수 =25 , 3x3 2개 사용 3x3x2 = 18 연산량 감소
 -  깊이를 깊게 함으로써 활성화함수를 많이 사용 가능
 
#### 1-2-2. 사용 이유
- 영상 의료데이터를 사용한 다수의 논문에서 VGG모델을 사용
- VGG16보다 layer추가로 인한 성능이 올라가기 때문

#### 1-2-3. 파라미터
- Epoch :  50
- Batch size : 16, 25 
- Dropout 사용 유무
- Invert사용 유무
- Optimizer: SGD
  - 선택한 이유:  최근 모델들은 대부분 Adam 을 사용
  - 여러가지 방법으로 Adam을 사용하였으나 모든 모델의 성능이 60-70%로 매우 낮음
  - SGD를 사용함으로써 최소 80%로 성능이 향상
  - So, SGD 사용
      - Adam은 SGD보다 탄력적인 일반화가 되지않고 학습이 포화 되어 성능이 낮다고 판단 

#### 2. Efficient Net

</br>


#### 3. Inception resnet V2

</br>


****************************

</br>

## 3. CONCLUSION
- VGG19모델에서 전처리 기법인 Invert를 활용한 연구방안을 제언
- 사용했던 모델 뿐 아니라 Multiply Merge의 특징을 더 부각시킬 수 있는 모델을 찾아 성능을 개선시키는 연구를 제언

</br>

*********
</br>

## 4. comment & limitations
- 시간과 GPU가 부족해 원하는 많은 모델을 학습시키는데 한정적
- Train Result와 초기 입력한 Parameter가 저장되지 않아(파이프라인을 다시 실행할 경우) 추가 저장하지 않으면 유실될 가능성 있음
- Test Result를 DEEP:PHI 내에서만 결과를 볼 수 있어 불편
- 데이터 출처에 대한 정확한 정보가 없기 때문에 이를 알아보기 위해 discussion을 찾아 보았으며, 이전에 각종 질문이 있었으나 정작 원작자 본인으로부터 답변을 받지 못함. (신뢰성 부족)
- 출처가 정확한 데이터가 필요하였고, ADNI 사이트에서 신청하였지만 계속해서 승인대기중인 상태. 

*********

</br>

## 5. 추후 연구 과제
- Alzheimer MRI image 3 Class 다중분류
(AD: 치매, MCI: 경도인지장애, NC: 정상)
- ADNI 데이터(3D) 이용하여 모델 학습
  - ADNI 
    - AD 진행을 늦추거나  중지시키는 치료법 조사를 개발하기 위해 만든 표준화된 프로토콜 세트 공유 사이트
    - 과학 간행물의 수 : 1500개 이상 작성, 다수의 논문 출간


