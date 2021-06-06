

# Predict Alzheimer MRI Image Classification 

<img width="737" alt="스크린샷 2021-06-05 오후 6 28 10" src="https://user-images.githubusercontent.com/75352728/120887051-c96a1c80-c62b-11eb-8da5-01ecf7335534.png">

deeplearning project

***
## 1. INTRO

<img width="1209" alt="스크린샷 2021-06-05 오후 6 29 10" src="https://user-images.githubusercontent.com/75352728/120887072-ec94cc00-c62b-11eb-93a6-97ab3a013225.png">

보건복지부 결과에 따르면 알츠하이머 치매 유병률과 환자가 급증하고 있어 알츠하이머 치매 조기발견 중요

현재 의료진은 MRI 이미지 판독 시 의료진마다 판단 기준이 달라 더욱 정확한 mri 이미지 판독이 필요

</br>

### 1-1 Purpose

- Kaggle에서 제공한 알츠하이머 치매 MRI image 데이터를 활용한 딥러닝 프로젝트
- 본 프로젝트에서는 Alzheimer와 Non-Alzheimer를 구분
- 목적 :  DEEP:PHI를 사용하여 두가지 전처리 방법과 하이퍼 파라미터 튜닝을 통한 MRI 이미지 분류 CNN 모델 성능 향상 
- 기대 방향 : Alzheimer 환자 조기 발견 보조 지표


</br>

### 1-2. CONCLUSION

<img width="1372" alt="스크린샷 2021-06-05 오후 6 45 06" src="https://user-images.githubusercontent.com/75352728/120887489-2797ff00-c62e-11eb-8c29-6806a8e544e0.png">
- VGG19모델에서 전처리 기법인 Invert를 활용한 연구방안 제언
- 사용했던 모델 뿐 아니라 Multiply Merge의 특징을 더 부각시킬 수 있는 모델을 찾아 성능을 개선시키는 연구 제언

<br>

### 1-3. DEEP:PHI

![1](https://user-images.githubusercontent.com/78459269/120887158-3e3d5680-c62c-11eb-87c6-24137c401715.png)


- GUI 기반의 파이프라인 기술을 통해 코딩없이 직관적인 AI 모델 설계가 가능한 플랫폼 
- 딥러닝 파이프라인을 블럭단위로 설정하여 학습

<br>

### 1-4 데이터

#### 1. 출처
  - kaggle : Alzheimer_binaryclassification
  - 링크1 : https://www.kaggle.com/smiti14/alzheimer-binaryclassification
  - 링크2 : https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images
  
#### 2. 데이터 구성
- Alzheimer : 6,160 (6,160)
- Non_Alzheimer : 6,393 (6,393)

#### 3. 데이터 정보


<img width="400" alt="스크린샷 2021-06-06 오후 7 26 29" src="https://user-images.githubusercontent.com/75352728/120921116-19fd7a80-c6fd-11eb-8ef0-b1d7d6ae65d4.png">


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


<img width="500" alt="스크린샷 2021-06-06 오후 7 28 10" src="https://user-images.githubusercontent.com/75352728/120921154-55984480-c6fd-11eb-869e-e7ed6c441b0c.png">

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

<img width="600" alt="스크린샷 2021-06-06 오후 7 29 56" src="https://user-images.githubusercontent.com/75352728/120921222-95f7c280-c6fd-11eb-8ba9-6fa1b390ed33.png">



#### Result

- Invert 전처리 기법을 사용한 경우 기존 모델보다 성능 좋음
- Optimizer에서 SGD를 사용한 모델이 Adam을 사용한 모델보다 성능 좋음
- Drop Out를 사용한 모델은 사용하지 않은 모델보다 성능 좋음
- Batch Size 크기가 작은 모델은 크기가 큰 모델보다 성능 좋음

	=> 알츠하이머 예측을 주제로 한 의료 데이터(MRI)에 전처리 Invert 관련 논문 부족한 실정

	=> 전처리 Invert를 활용한 관련 연구 방안 제언

#### 1-1. 특징



<img width="450" alt="스크린샷 2021-06-06 오후 7 33 20" src="https://user-images.githubusercontent.com/75352728/120921306-0e5e8380-c6fe-11eb-9082-6ea41cba647b.png">

- 중요한 영역이 주변보다 어두운 경우, 반전된 이미지를 이용하여 feature 추출 용이
- 이미지 상에서 검은색 부위를 하얀색부분으로 하얀색부분을 검은색 부분으로 반전

</br>

#### 1-2. 사용 이유


- 다수의 논문에서 조영술이나 x-ray Image에 Invert를 적용한 사례가 있음
- but,  Mri 적용사례는 적어 알츠하이머 데이터에 적용시킴

<img width="450" alt="스크린샷 2021-06-06 오후 7 33 43" src="https://user-images.githubusercontent.com/75352728/120921318-1ae2dc00-c6fe-11eb-8c57-13c00e57aeb2.png">

- 뇌실의 크기와 뇌실의 주름색이 배경색과 같으므로 detection할때 방해가 있을거라 판단
- 대뇌피질의 수축정도를 이용해서 학습시키면 성능이 올라갈거라 판단하여 사용

#### 2. Merge

</br>

<img width="600" alt="스크린샷 2021-06-06 오후 7 32 00" src="https://user-images.githubusercontent.com/75352728/120921269-df481200-c6fd-11eb-903c-97c85f7d3a14.png">

#### Result
- 첫 번째 이미지와 두 번째 이미지에 각각 가중치를 곱하여 합치는 Overlap 방식의 Merge보다 같은 위치의 원소끼리 곱하여 더하는 Multiply 방식의 Merge가 더 성능이 좋음

  => 이미지를 Multiply 방식으로 merge하는 기법에 대한 연구방안을 제언함
  <br>  
#### 2-1. 특징

<img width="700" alt="스크린샷 2021-06-06 오후 7 36 07" src="https://user-images.githubusercontent.com/75352728/120921401-70b78400-c6fe-11eb-9f64-abeda9aa0e3d.png">


- Overlap Merge : 가중치에 따라 이미지의 밝기 정도를 개선
- Multiply Merge : 대뇌 크기의 수축 정도를 개선시키는 기대감에 사용

<br>

#### 2-2. 사용 이유


<img width="700" alt="스크린샷 2021-06-06 오후 7 34 57" src="https://user-images.githubusercontent.com/75352728/120921363-482f8a00-c6fe-11eb-8d9b-2db1d9ce50c6.png">


- Overlap Merge : 의료데이터를 이용한 kaggle 대회에서 이미지의 밝기를 개선시키는 전처리 기법으로 사용하여 참고
- Multiply Merge : Overlap Merge와 비교하기 위함

### 2-2 모델 

</br>

#### 1. VGG19

</br>

#### 1-1. 구조

</br>

<img width="500" alt="스크린샷 2021-06-06 오후 7 39 17" src="https://user-images.githubusercontent.com/75352728/120921487-e3286400-c6fe-11eb-8ace-6f3703aa4097.png">


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


<img width="450" alt="스크린샷 2021-06-06 오후 7 36 27" src="https://user-images.githubusercontent.com/75352728/120921411-7f05a000-c6fe-11eb-93a4-b7c7128005c8.png">

<br/>

#### 1-2-1. 특징
 - 네트워크의 깊이가 성능에 미치는 영향을 보기 위해 만들어진 모델
 - 3by3 의 작은 필터를 사용하여 발생하는 파라미터 개수 줄임
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

#### 2-1. 구조

<br>

layer층 
- 이미지가 input되면 width(채널의 갯수), Dapth(신경망 깊이), resolution(해상도) 이 3가지의 scaling을 적절한 가중치에 맞춰 학습
- 입력된 이미지에 convolutional 필터를 적용할수로 원본 이미지의 해상도는 줄어들고, 채널은 점점 올라감

</br>

#### 2-2. 특징 및 파라미터

<br/>

#### 2-2-1. 특징

- 해당 이미지의 최적의 가중치를 찾아서 학습 (grid search방식)
- 다양한 Layer를 구성 (MBConv 사용)
- MBConv : 이미지의 정보를 효율적으로 전달하기 위해 다양한 Layer로 구성하게 만드는 다른 이미지 모델에서 쓰이는 Conv

#### 2-2-2. 사용 이유
- 최근 ImageNet 학습에서 성능이 좋게나온 논문을 근거로 함

#### 2-2-3. 파라미터 (논문을 기반으로 동일하게 적용)
- epoch : 20
- batch size : 32 
- Augmentation 유/무 
  - drop out(0.25)
  - Zoom
  - Rotate
- Optimizer
  - ADAM

</br>

*********
</br>

## 4. comment & limitations
- 시간과 GPU가 부족해 원하는 많은 모델을 학습시키는데 한정적
- Train Result와 초기 입력한 Parameter가 저장되지 않아(파이프라인을 다시 실행할 경우) 추가 저장하지 않으면 유실될 가능성 있음
- Test Result를 DEEP:PHI 내에서만 결과를 볼 수 있어 불편
- 데이터 출처에 대한 정확한 정보가 없기 때문에 이를 알아보기 위해 discussion을 찾아 보았으며, 이전에 각종 질문이 있었으나 정작 원작자 본인으로부터 답변을 받지 못함 (신뢰성 부족)
- 출처가 정확한 데이터가 필요하였고, ADNI 사이트에서 신청하였지만 계속해서 승인대기중인 상태

*********

</br>

## 5. 추후 연구 과제
- Alzheimer MRI image 3 Class 다중분류
(AD: 치매, MCI: 경도인지장애, NC: 정상)
- ADNI 데이터(3D) 이용하여 모델 학습
  - ADNI 
    - AD 진행을 늦추거나  중지시키는 치료법 조사를 개발하기 위해 만든 표준화된 프로토콜 세트 공유 사이트
    - 과학 간행물의 수 : 1500개 이상 작성, 다수의 논문 출간


