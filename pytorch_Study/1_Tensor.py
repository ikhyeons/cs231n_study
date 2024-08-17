import torch
import numpy as np


# 배열을 직접 넣어서 텐서를 생성
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)
#Numpy 에서 ndarray를 텐서로 변환할 수 있음
np_array = np.array(data)
x_tensor = torch.from_numpy(np_array)
print(x_tensor)

#다른 tensor 기반으로 생성하기
##명시적으로 재정의(override)하지 않는다면, 인자로 주어진 텐서의 속성(모양, 자료형)을 유지함
x_ones = torch.ones_like(x_data) # x_data 의 속성을 유지
print(f"Ones Tensor : \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data 의 속성을 덮어씀
print(f"Random Tensor : \n {x_rand} \n")

#무작위(random) 또는 상수(constant) 값을 사용하기
shape = (2, 3)
rand_tensor = torch.rand(shape, dtype=torch.float)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor : \n {rand_tensor} \n")
print(f"Ones Tensor : \n {ones_tensor} \n")
print(f"Zeros Tensor : \n {zeros_tensor} \n")

#tensor의 속성(attribute)
tensor = torch.rand(3, 4)
print(f"Shape of tensor : {tensor.shape}")
print(f"datatype of tensor : {tensor.dtype}")
print(f"Device tensor is stored on : {tensor.device}")
print(f"tensor is : \n{tensor}")

# GPU에 tensor 밀어넣기
print(tensor.device)
if torch.cuda.is_available():
    print("gpu available")
    tensor_g = tensor.to("cuda")
print(tensor_g.device)

# tensor 결합하기
t1 = torch.cat([tensor, tensor, tensor] , dim=1)
print(t1)
t2 = torch.cat([tensor, tensor, tensor] , dim=0)
print(t2)

# 산술연산
## 행렬 곱
tensor = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
y1 = tensor @ tensor.T
print("y1: ", y1)
y2 = tensor.matmul(tensor.T)
print("y2: ", y2)
y3 = torch.rand_like(y1)
print("y3: ", y3)

##요소별 곱을 계산함
z1 = tensor * tensor
print("z1 : ", z1)
z2 = tensor.mul(tensor)
print("z2 : ", z2)
z3 = torch.rand_like(tensor)
print("first z3: ", z3)
torch.mul(tensor, tensor, out=z3)
print("z3 : ", z3)

#단일 요소 tensor의 경우 item()을 사용하여 숫자로 변환 가능
tensor_sum = tensor.sum()
sum = tensor_sum.item()
print("item : ", sum, "type : ", type(sum))


