import numpy as np
from scipy.special import softmax

class LogisticRegression:

    def __init__(self, lr:float=0.1, w_init:np.ndarray=None, b_init:float=0):
        """
        ## 配置初始参数
        - input:
            - **`lr`**: 学习率
            - **`w_init`**: 初始权重
            - **`b_init`**: 初始偏置
        - no output
        """
        self.lr = lr
        if w_init is None:
            self.w = None
        else:
            self.w = w_init.copy()
        self.b = b_init


    def predict(self, x:np.ndarray):
        """
        ## 单次推理
        - **`y = softmax(w * x + b)`**
        - input:
            - **`x`**: 输入特征向量
        - output:
            - **`y`**: 输出推理结果\in(0,1)
        """
        return softmax(np.dot(self.w, x) + self.b)
    

    def train_sample_wise(self, x:np.ndarray, y:int):
        """
        ## 逐样本训练
        - 对于每个样本:
            - **`w = w - lr * (f(x_i) - y_i) * x_i`**
            - **`b = b - lr * (f(x_i) - y_i)`**
        - input:
            - **`x`**: 单个训练样本的特征向量
            - **`y`**: 单个训练样本的标记值
        - no output
        """
        y_predict = self.predict(x)

        self.w -= self.lr * (y_predict - y) * x
        self.b -= self.lr * (y_predict - y)


    def train_batch_wise(self, x_batch:np.ndarray, y_batch:np.ndarray):
        """
        ## 逐批次训练
        - 对于整个批次内的样本:
            - **`w = w - lr * \\average {(f(x_i) - y_i) * x_i}`**
            - **`b = b - lr * \\average {(f(x_i) - y_i)}`**
                - **`\\sum`** 和 **`\\average`** 的版本同样均存在
        - input:
            - **`x_batch`**: 训练样本的特征向量的集合, shape=(number,dimension)
            - **`y_batch`**: 训练样本的标记值的集合
        - no output
        """
        loss_w = 0
        loss_b = 0

        for idx in range(x_batch.shape[0]):
            x = x_batch[idx]
            y = y_batch[idx]
            y_predict = self.predict(x)

            loss_w += (y_predict - y) * x
            loss_b += y_predict - y
        
        self.w -= self.lr * loss_w / x_batch.shape[0]
        self.b -= self.lr * loss_b / x_batch.shape[0]

    def train(self, X:np.ndarray, y:np.ndarray, config:str="sample-wise", batch_size:int=None):
        """
        ## 训练
        - input:
            - **`X`**: 整个训练集的特征向量部分, shape=(number,dimension)
            - **`y`**: 整个训练集的标记值部分
            - **`config`**: 选择逐样本/逐批次训练
            - **`batch_size`**: 选择逐批次训练时需提供 batch 大小
        - no output
        """
        n_samples, n_features = X.shape
        if self.w is None:
            self.w = np.zeros(n_features)

        if config == "sample-wise":
            indices = np.random.permutation(n_samples)
            for idx in indices:
                self.train_sample_wise(X[idx], y[idx])

        elif config == "batch-wise":
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for idx in range(0, n_samples, batch_size):
                end_idx = min(idx + batch_size, n_samples)
                X_batch = X_shuffled[idx:end_idx]
                y_batch = y_shuffled[idx:end_idx]
                self.train_batch_wise(X_batch, y_batch)
