python의 numpy 패키지와 pytorch는 vector, matrix, tensor를 쉽게 다룰 수 있는 많은 함수와 자료구조를 제공

[Pytorch로 시작하는 딥러닝] https://wikidocs.net/52460 

1. **텐서의 차원과 형태 관련**

```
X, Y        # Tensors
X.dim()  # 차원을 출력
X.shape, X.size() # 텐서의 모양을 출력
```

import torch

X =torch.rand(3,4, 5)



print('X.dim()', X.dim())

print('X.shape', X.shape)

print('X.size()', X.size())



X.dim() 3 

X.shape torch.Size([3, 4, 5]) 

X.size() torch.Size([3, 4, 5])



2. **Mul 과 Matmul의 차이**

> ```
> X.matmul(Y) # 행렬 곱 (내적)
> X.mul(Y)    # X*Y 와 동일. 각 원소 별로 곱하는 연산.
> ```

X = torch.rand(3,2,5) # (2,5)행렬 3개

Y = torch.rand(3,5,3) # (5,3)행렬 3개

X내적Y = (2,3)행렬 3개

> 배치 multiplication 2dim 행렬을 반복해서 내적을 하지 않고
>
> 동시에 병렬로 연산을 할 수 있게끔 해서 
>
> 학습을 더 빠르게 진행시킬 수 있게 한다.

D = X.matmul(Y) # 행렬 곱 (내적)

print(D)

print(D.shape)

\# X.mul(Y)  # X*Y 와 동일. 각 원소 별로 곱하는 연산.



tensor([[[1.4064, 1.1559, 0.8575],         [1.1159, 1.0953, 1.2287]],         [[2.1129, 1.7717, 1.4594],         [0.8134, 0.5126, 0.5119]],         [[1.8193, 2.0829, 1.6199],         [1.2398, 1.6927, 1.6878]]]) torch.Size([3, 2, 3])



3. **Broadcasting**

> X의 shape이 [100,1,20] 이고, Y가 [100,40,20] 일때, 크기가 다르기 때문에 원소별 연산 (예; +, * 등) 적용이 불가한 것처럼 보이지만, numpy와 pytorch는 broadcasting을 지원하여, X의 두번째 차원 1인 경우, Y의 두번째 차원인 40에 맞춰 X의 값들을 반복 복제하는 식으로 자동으로 크기를 조절하여 연산이 가능하도록 만들어줍니다. 매번 복제하는 코드를 명시적으로 구현할 필요 없이, 매우 편리하게 응용될 수 있지만, 의도치 않게 broadcasting이 되면 디버깅을 할때 오류를 잡는 것이 어려워 질 수 있으므로 꼭 주의하여 사용 것이 좋습니다.

>따로 복사를 하거나 강제로 for문을 돌려서 연산을 하는거보다
>
>Broadcasting을 사용하게 되면 더 빠르게 연산처리를 할 수 있다.
>
>유의할 점은 의도치 않게 연산을 할 가능성이 있기 때문에 
>
>comment 처리를 잘 해서 디버깅시에 유의해야한다.
>
>



4. **view() (numpy의 경우는 reshape())**

> 텐서의 shape를 바꿔야할 경우 사용합니다. 매우 자주 쓰이는 함수이므로 꼭 알아두세요!

X.shape

X.view(3,10).shape

torch.Size([3, 10])

> (3,2,5) 의 2dim 행렬이 10개의 값을 가지고 있기 때문에 1dim으로 낮추고
>
> 10개를 가진다 의 (3,10)으로 바꿔준다



5. **axis개념**

> 많은 함수에서 매개변수로 요구되는 개념입니다. 다차원 텐서에 해당 함수 연산을 어떤 축으로 적용 할지를 결정하는데 사용합니다.
> 예; np.concaternate((A1,B1),axis = 0) 이런식으로 함수의 매개변수 중 하나로 자주 등장합니다.

> 어떤 오퍼레이션이 한 축으로만 일어나야할때 그걸 고정한다



6. **squeeze & unsqueeze**

> 특정 차원이 1인 경우 축소시키거나 차원을 확장시킬 때 사용합니다. 자주 등장하므로 알아두면 좋습니다.
>
> ```
> torch.squeeze(X) # X: [100,1,20] => [100,20]
> ```

> broadcasting을 할때 많이 사용하고
>
> 그냥 1을 없애줄때 차원을 변경해 줄때 많이 쓴다



7. **type casting**

> 컴퓨터는 여러가지 자료형(int,float,uint8 등)이 있지만 type이 같지 않으면 수치적인 문제가 발생할 수 있기 때문에 항상 유의하는 것이 좋습니다. 따라서 자료형이 무엇인지 그리고 어떻게 바꾸는지 알고 있어야 합니다.

> floating 32bit 짜리를 가장 많이 이용한다.
>
> 인공지능 모델들이 32bit에 가장 많이 최적화가 되어있다.
>
> 타입은 항상 유의하는 것이 좋다.



8. **concatenate**

> 두 개 이상의 텐서들을 지정된 축으로 쌓아서 더 큰 텐서를 만드는 함수. 하나의 텐서로 합치기 위한 입력 텐서들은 concatenate하는 축을 제외하고는 모두 같은 크기를 가져야 합니다.

> 쌓으려는 축을 제외하고는 모두 같은 크기를 가져야한다..



FloatTensor = float 32bit 짜리 tensor

torch.FloatTensor( 행 [ 열 [ ]  ] )



 import pathlib

path = pathlib.Path('/content/gdrive/MyDrive/Colab Notebooks/사전학습/health_data.csv') 



data_file = pd.read_csv(path)

data_file.head() # 위에 몇개만 읽어봅시다.

정제가 되지 않은 NaN이나 null 이 포함된 데이터인 경우



data_file.isnull() 로 확인이 가능하고

data_file = data_file.dropna(axis=0).reset_index(drop=True)

data_file.head()

의 코드로 제거할 수 있다.



> csv 파일의 경우 pytorch에서 바로 사용할 수 없다.
>
> 따라서 tensor의 형태로 바꿔주어야한다.
>
> pytorch의 가장 기본적인 형식은 tensor 이다.

```
height = torch.tensor(data_file.height)
weight = torch.tensor(data_file.weight)
```



shape, dim, float

```python
x_train = height.view([height.shape[0],1]).float()
y_train = weight.view([weight.shape[0],1]).float()
```



랜덤으로 노이즈 데이터를 만들어서 실제로 돌려보자

```python
# x_train dataset의 경우 145부터 190사이의 랜덤한 숫자 50개, 
# y_train dataset의 경우 45부터 85 사이의 랜덤한 숫자 50개를 생성하여 concatenate를 시킵니다.
x_train = torch.cat((torch.rand(50,1)*45+145,x_train), axis=0)
# rand(50,1)*45 => 0~45의 값을 가지는 랜덤 값 50개
y_train = torch.cat((torch.rand(50,1)*40+45,y_train), axis=0)
```





