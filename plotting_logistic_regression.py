import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel
from statsmodels.genmod.families.links import logit
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
with open('logit_model.pkl','rb') as file:
    logistic_model = pickle.load(file)


coeffi = logistic_model.params.values

# 격자 생성을 위한 distance와 speed_diff의 범위 정의
distance_range = np.linspace(0, 100, 100)
speed_diff_range = np.linspace(-10, 100, 100)

# 격자 생성
D, S = np.meshgrid(distance_range, speed_diff_range)
# TTC = D/S  # TTC 계산
X_grid = np.column_stack((np.ones(D.size), D.ravel(), S.ravel()))  # 상수 항 포함

# X_grid = np.column_stack((np.ones(D.size), TTC.ravel()))
# 모델을 사용하여 격자 위의 지점들에 대한 p 예측
p_predicted = logistic_model.predict(X_grid)
p_predicted = p_predicted.reshape(D.shape)

print(logistic_model.summary())
# 3D 그래프 준비
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

test_set = [(20,10),(20,40),(30,40),(30,20),(30,30),(40,30),(-5,5)]
# for i in test_set:
#     print(logistic_model.predict(np.column_stack((np.ones(1), i[0], i[1]/3.6)))[0])
#     print(1/(1+math.exp(-(coeffi[0]+coeffi[1]*i[0]+coeffi[2]*i[1]/3.6))))
#     print()
#     # print('(X,Y)=(%dm, %dkm/h)=%2.5f'%(*i, results.predict(1, i[0], i[1]/3.6)))

# 예측된 p 값을 사용하여 3D 플롯 그리기
ax.plot_surface(D, S, p_predicted,  cmap='summer', alpha=0.5)

ax.set_xlabel('$X_{1}$: Distance [m]',fontsize = 15)
ax.set_ylabel('$X_{2}$: Speed Difference [km/h]',fontsize = 15)
ax.set_zlabel('$Y$',fontsize = 15)
ax.set_zlim([0.0, 1.0])
plt.title('Interaction probability $Y$',fontsize = 15)

plt.show()