import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


train_data=np.array(pd.read_csv('linear_data_train.csv'))
test_data=np.array(pd.read_csv('linear_data_test.csv'))

X_train =train_data[:, 0:2]
Y_train = train_data[:, 2]
X_test =test_data[:,0:2]
Y_test= test_data[:,2]


N=X_train.shape[0]
lr=0.01
epochs=3

m = np.random.rand(2,1)
b = np.random.rand(1,1)

fig=plt.figure()
ax=fig.add_subplot(projection='3d')

x_0_range=np.arange(X_train[:,0].min(),X_train[:,0].max(),0.1)
x_1_range=np.arange(X_train[:,1].min(),X_train[:,1].max(),0.1)

errors=[]

for i in range(epochs):
    for n in range(N):

        y_pred = np.matmul(X_train[n:n+1],m)+b
        e= np.subtract(Y_train[n], y_pred)

        #update
        m = m+ lr*X_train[n:n+1,:].T*e
        b +=lr*e

        # Error
        y_pred=np.matmul(X_train,m)+b
        error=np.mean(Y_train - y_pred)
        errors.append(error)
        print('error',error)

        #plot data
        ax.clear()
        x_0, x_1 = np.meshgrid(x_0_range, x_1_range)
        z = x_0 * m[0] + x_1 * m[1] + b
        ax.plot_surface(x_0, x_1, z, rstride=1, cstride=1, alpha = 0.4)
        ax.scatter(X_train[Y_train == 1,0], X_train[Y_train == 1,1], Y_train[Y_train == 1], c='r', marker='o')
        ax.scatter(X_train[Y_train == -1,0], X_train[Y_train == -1,1], Y_train[Y_train == -1], c='g', marker='o')
        ax.set_xlabel('X0')
        ax.set_ylabel('X1')
        ax.set_zlabel('Y')
        plt.pause(0.001)

