# 로지스틱 회귀
## 이론 문제 1
**다음중 로지스틱 회귀에 대하여 잘못 설명한 것을 고르시오.**
**답 : 2번**

	1.  로지스틱함수는 확률밀도 함수이다.
	2.  출력값의 범위가 항상 0~1의 사이이므로 경사하강법을 수행할 때 기울기가 급격하게 변해서 발산하는 기울기 폭주가 발생할 수 있다.
	3.  시그모이드 함수는 출력값의 범위가 매우 좁기 때문에, 역전파 과정에서 기울기 소실 문제가 발생할 수 있다.
	4.  로지스틱 회귀는 일반적으로 선형회귀와는 다르게 종속 변수가 범주형 데이터를 대상으로 한다.
	5.  이진분류에는 시그모이드 함수를, 다중분류에는 소프트맥스 함수를 사용하는 것이 일반적이다.

2. 기울기 폭주는 일어나지 않는다.

## 이론 문제 2
**다음은 ‘확률적 경사 하강법’에 대한 설명이다. 교재와 아래 첨부한 논문을 참고하여 다음 중 올바르게 설명한 것을 모두 선택하시오. (2개)**
**답 : 1, 4**

논문: https://arxiv.org/pdf/1609.04747.pdf 의 p.2

	1.  Parameter를 수정하는 정도를 나타내는 학습률에 따라 ‘확률적 경사 하강법’은 ‘배치 경사 하강법’ 보다 더 최적인 parameter를 구할 수 있다 
	2.  ‘확률적 경사 하강법’은 한번에 여러개의 train sample을 사용하여 최적의 모델을 찾는 알고리즘이다
	3.  ‘배치 경사 하강법’은 한번에 전체 훈련샘플을 사용하여 경사 하강법을 수행한다. 따라서 모델 최적화 과정에서 훈련에 사용할 데이터가 많은 상황에서 ‘배치 경사 하강법’은 '확률적 경사 하강법'보다 불필요한 연산을 덜 한다.
	4.  에포크(epoch)의 횟수가 매우 많을 때 과대적합의 가능성이 발생하고, 횟수가 매우 적을 때 과소적합의 가능성이 발생한다.
	5.  손실함수의 값이 클수록 ‘확률적 경사 하강법’으로 훈련되는 모델은 최적화가 잘 된다.


## 실습 문제 3. 토마토 분류

철수는 토마토 파스타를 먹는 중, 문득 토마토가 과일인지 채소인지 궁금해졌다. 철수는 과일과 채소의 영양소를 분석해보면 토마토가 과일인지 채소인지 알 수 있다고 생각했다. 철수는 수업 시간에 배웠던 로지스틱 회귀를 이용하여 토마토가 채소일 확률을 측정하려고 한다. 주어진 코드의 뒷부분을 완성하여 과일과 채소를 분류하는 모델을 만들어 토마토를 분류해보자.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

food = pd.read_csv('<https://www.dropbox.com/s/ry9t14zj8fwdwtq/food_samples.csv?dl=1>')
tomato = pd.read_csv('<https://www.dropbox.com/s/kk1qscljxe5kb7x/tomato.csv?dl=1>')

food_input = (food[['단백질(g)', '지질(g)', '탄수화물(g)', '총당류(g)', '나트륨(mg)', '콜레스테롤(mg)', '총 포화 지방산(g)', '트랜스 지방산(g)']]).to_numpy()
food_target = food['식품상세분류'].to_numpy()
tomato_input = (tomato[['단백질(g)', '지질(g)', '탄수화물(g)', '총당류(g)', '나트륨(mg)', '콜레스테롤(mg)', '총 포화 지방산(g)', '트랜스 지방산(g)']]).to_numpy()

train_input, test_input, train_target, test_target = train_test_split(food_input, food_target, random_state=42)

scaler = StandardScaler()
scaler.fit(train_input)
train_scaled_input = scaler.transform(train_input)
test_scaled_input = scaler.transform(test_input)
scaled_tomato = scaler.transform(tomato_input)

LogiReg = LogisticRegression(C=100)
LogiReg.fit(train_scaled_input, train_target)
```
**참고**
주어진 데이터는 음식 100g에 들어간 영양소의 무게이다.

**제한조건**
로지스틱 회귀를 사용하되, 로지스틱 회귀 fit() 이외에 다른 메서드를 사용하지 말 것 (힌트 : 로지스틱 회귀의 coef_ 속성과 intercept_ 속성을 이용하세요)

**출력**
주어진 토마토가 채소인지 과일인지 출력하고 그 확률을 소수 여섯째 자리에서 반올림하여 출력하여라

**기타**
데이터 출처 : 식품 영양성분 데이터 베이스 https://www.foodsafetykorea.go.kr/fcdb/

### 답안
``` python
decision = LogiReg.coef_.dot(np.transpose(scaled_tomato)) + LogiReg.intercept_
sigmoid = 1/(1+np.exp(-decision))

for i,j in zip(tomato['식품명'],sigmoid.reshape(-1)):
	k = "채소"
	if j <= 0.5:
		k = "과일"
		j = 1 - j

print(i, " =", k, " : ", "%0.6f"%j)
```

## 이론 문제 4
**틀린 것을 모두 골라라 ( 2개 )**
**답 :  2, 4**

	1번. 정확도대신 손실 함수를 모델 성능의 지표로 삼는 이유는 매개변수의 미소한 변화에 손실 함수는 연속적으로 변화하는 반면 정확도는 반응이 없거나 불연속적으로 변화하기 때문이다.
	2번. 로지스틱 회귀 분석은 비 선형적인 시그모이드 함수를 활성화 함수로 사용해 비 선형 값을 출력하기 때문에 MAE을 사용할 수 없다.
		MAE의 식을 제곱하게 되면 한 개의 최저점을 갖는 볼록 함수가 아닌 여러 개의 최저점을 갖는 함수가 나오기 때문이다.
	3번. 크로스엔트로피 손실 함수는 분류 모델의 발견될 확률 분포와 예측 분포사이의 차이를 측정하고 예측에 대해 가능한 모든 값을 저장한다.
	4번. 분류해야 할 클래스가 3개 이상인 문제에서 데이터의 라벨을 반드시 one – hot encoding의 형태로 표현되어야 한다.
	5번. optimization 과정은 손실 함수의 결괏값을 최소화하는 가중치와 편향을 찾는 것으로 optimizer의 back propagation과정을 통해 이루어진다.

## 실습 문제 5
**<타이타닉 사망 및 생존자 예측 모델 구축>**
첨부된 titanic_train데이터와 titanic_test 데이터를 다운받고 4장에서 배웠던 LogisticRegression과 SGD를 이용하여 타이타닉 사고 때 생존한 사람들을 예측해보는 모델을 만들어라.

**제한조건**

LogisticRegression과 SGD를 사용할 때 fit, transform, score을 제외하고 다른 메서드는 사용하면 안된다. 하지만 각 함수안에 있는 파라미터들을 바꾸는 것이라면 허용한다. 또한, titanic_train 데이터와 titanic_test 데이터를 한 데이터셋으로 결합한 후 분석을 진행하라.

**출력**

score 함수를 이용하여 train셋의 정확도와 test셋의 정확도를 출력한다. 자세한 방식은 아래 사진을 참고하라.

  

LogisticRegression

![](https://lh4.googleusercontent.com/Zm2NoW8g2BTqekQzbppH69IgpGKLKf9vXblfVNBNU0JPyjjuoUYZQ-Rk_uqedAdVsYJu906Ee5jeFlIRxwxy-QCNSCU47rTwW9YFXsSHVzmFW2mp9x43A234o_uzJqpp75KQcqD4)

SGD

![](https://lh6.googleusercontent.com/fz_PFyn41n9ogije1zelZ5AaHFZ5VWNMttb4BrXEXHD3YaYrK9FDuw929NWF5S-jsuRozCPPlmVHLd1zR8X0BJVO6S6G8sxVN7GzmSLQ6uhHs4EEDzvJbrFt-C049B8GpsQFfmle)

→ 이러한 방식으로 제출하라. 정확도는 분석하는 사람마다 다르므로 위 사진과 같지 않아도 된다.

  

**유의사항**

전처리 도중에 SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame 라는 에러가 뜰 수도 있다. 이 문제의 경우에 한해서는 무시해도 괜찮은 에러이므로 그대로 진행하면 된다. 또한, 예측하는 target 데이터의 이름은 survival이라는 것을 유의하자.

**힌트**

1.  pandas 모듈에는 전체적인 데이터가 어떤 자료형인지, 몇 개가 있는지 알려주는 info라는 함수가 있다. 이를 이용해서 전체적인 데이터를 파악한 뒤 전처리를 진행하면 전처리에 대한 방향성을 알기 쉽다.
2.  데이터를 병합한 후 index번호가 어떻게 되는지 살펴보고 문제가 있다면, 이를 해결하기 위해 어떻게 해야하는지 생각해 보라.
3.  데이터에 결측치가 있다면 이에 대한 전처리를 진행한 뒤 분석을 진행해야 한다.
4.  데이터에 포함된 모든 열을 굳이 input, target 데이터에 넣을 필요는 없다. 하지만, 특정한 열은 정확도를 개선시키는 데 큰 도움이 된다. 이러한 열을 찾아야 한다. 