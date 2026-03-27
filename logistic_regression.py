import numpy as np
from scipy.special import expit

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
        - **`y = sigmoid(w * x + b)`**
        - input:
            - **`x`**: 输入特征向量
        - output:
            - **`y`**: 输出推理结果\in(0,1)
        """
        return expit(np.dot(self.w, x) + self.b)
    

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

# 示例（基本沿用Perceptron）
if __name__ == "__main__":
    # 加载一组线性可分的数据集
    data_path = "examples/linearly_separable_data.npz"
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    # 修改原Perceptron标记，以符合逻辑回归的格式
    y = np.where(y == -1, 0, 1)

    print(f"Data loaded from: {data_path}")
    print(f"Total samples: {X.shape[0]}")
    
    # 划分训练集和测试集（前50正+50负训练，后10正+10负测试）
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    train_pos_idx = pos_idx[:50]
    test_pos_idx = pos_idx[50:60]
    train_neg_idx = neg_idx[:50]
    test_neg_idx = neg_idx[50:60]

    train_idx = np.concatenate([train_pos_idx, train_neg_idx])
    test_idx = np.concatenate([test_pos_idx, test_neg_idx])
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Training set: {len(X_train)} samples (50 positive, 50 negative)")
    print(f"Test set: {len(X_test)} samples (10 positive, 10 negative)")
    print()
    
    # 训练sample-wise模型
    lr = 0.1
    logistic_sample = LogisticRegression(lr=lr)
    print("Training sample-wise...")
    logistic_sample.train(X_train, y_train, config="sample-wise")
    
    # 训练batch-wise模型
    logistic_batch = LogisticRegression(lr=lr)
    batch_size = 10
    print(f"Training batch-wise (batch_size={batch_size})...")
    logistic_batch.train(X_train, y_train, config="batch-wise", batch_size=batch_size)
    
    # 测试
    pred_sample = np.array([logistic_sample.predict(x) for x in X_test])
    pred_batch = np.array([logistic_batch.predict(x) for x in X_test])
    
    acc_sample = np.mean((pred_sample >= 0.5).astype(int) == y_test) * 100
    acc_batch = np.mean((pred_batch >= 0.5).astype(int) == y_test) * 100
    
    # 输出结果
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Sample-wise model:")
    print(f"  w = [{logistic_sample.w[0]:.3f}, {logistic_sample.w[1]:.3f}], b = {logistic_sample.b:.3f}")
    print(f"  Test accuracy: {acc_sample:.2f}%")
    print(f"\nBatch-wise model (batch_size={batch_size}):")
    print(f"  w = [{logistic_batch.w[0]:.3f}, {logistic_batch.w[1]:.3f}], b = {logistic_batch.b:.3f}")
    print(f"  Test accuracy: {acc_batch:.2f}%")


# 预期输出：

# Data loaded from: examples/linearly_separable_data.npz
# Total samples: 120
# Training set: 100 samples (50 positive, 50 negative)
# Test set: 20 samples (10 positive, 10 negative)

# Training sample-wise...
# Training batch-wise (batch_size=10)...

# ==================================================
# TEST RESULTS
# ==================================================
# Sample-wise model:
#   w = [1.803, -0.397], b = 0.097
#   Test accuracy: 100.00%

# Batch-wise model (batch_size=10):
#   w = [0.653, -0.155], b = 0.023
#   Test accuracy: 100.00%
