dnn_ga
=============

Hansung Science High School 2020 Club Project<br/>


### How to use
* Please use Python version >= 3.7
* Please check [requirements](requirements.txt)
```
$ pip install -r requirements.txt 
```
### How it works
1. 유전알고리즘을 통해 DNN 알고리즘의 최적의 neuron 개수와 hidden_layer 개수, batch_size를 구하는 것이 목표
2. 따라서 유전알고리즘의 유전자에는 neuron 개수, hidden_layer 개수, batch_size가 해당됨
3. DNN 알고리즘은 유동적인 neuron 개수, hidden_layer 개수에 따라 모델을 스스로 생성
4. MNIST Dataset을 DNN에서 학습하는 것처럼 CIFAR-10 Dataset을 흑백으로 변환한 뒤, 이를 input 값으로 사용
5. 모델의 평가 정확도, 학습 속도, 평가 속도에 따라 서로 다른 Stack을 할당함.
 Stack이 일정 값 이상일 경우 해당 모델은 부적합한 모델이라고 판단하고 제거함.
 Stack이 작을 수록 우리의 목표에 맞는 모델이므로 해당 모델의 유전자를 다음 세대로 전달.
6. 일정 세대 만큼 지난 후에 나오는 모델의 유전자 정보가 가장 적합한 모델의 정보

### Club Fossil
한성과학고등학교 2020년 1학기 인공지능개발반
* 동아리장 및 레포지토리 관리: 오혁재
* 테스트용 이미지 흑백화 및 처리
  * 곽재우
  * 오혁재
* 테스트용 DNN
  * 허진석
  * 이현수
  * 김정환
* 테스트 DNN 유전
  * 김상민
  * 정노아
  * 지유근

프로젝트의 코드들은 담당 부분만 수정하지 않고 유동적으로 개발하였습니다.


