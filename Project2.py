import numpy as np

def relu(x):
    return np.maximum(0, x*0.1)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class MLPRegressor:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, momentum):
        # 設定 seed 固定隨機結果
        np.random.seed(150)

        # 初始化權重及偏差
        self.weights_input_hidden1 = np.random.randn(input_size, hidden_size1)
        self.bias_hidden1 = np.zeros((1, hidden_size1))
        self.weights_hidden1_hidden2 = np.random.randn(hidden_size1, hidden_size2)
        self.bias_hidden2 = np.zeros((1, hidden_size2))
        self.weights_hidden2_output = np.random.randn(hidden_size2, output_size)
        self.bias_output = np.zeros((1, output_size))

        # momentum 參數
        self.momentum = momentum

        # momentum 動量
        self.velocity_hidden2_output = np.zeros_like(self.weights_hidden2_output)
        self.velocity_hidden1_hidden2 = np.zeros_like(self.weights_hidden1_hidden2)
        self.velocity_input_hidden1 = np.zeros_like(self.weights_input_hidden1)

    def forward(self, X):
        # 前饋階段
        self.hidden_output1 = relu(np.dot(X, self.weights_input_hidden1) + self.bias_hidden1)
        self.hidden_output2 = relu(np.dot(self.hidden_output1, self.weights_hidden1_hidden2) + self.bias_hidden2)
        self.predictions = np.dot(self.hidden_output2, self.weights_hidden2_output) + self.bias_output
        return self.predictions

    def backward(self, X, y, learning_rate):
        # 倒傳遞階段
        error = self.predictions - y
        output_delta = error
        hidden2_error = np.dot(output_delta, self.weights_hidden2_output.T)
        hidden2_delta = hidden2_error * relu_derivative(self.hidden_output2)
        hidden1_error = np.dot(hidden2_delta, self.weights_hidden1_hidden2.T)
        hidden1_delta = hidden1_error * relu_derivative(self.hidden_output1)

        # 更新權重和偏差，使用 momentum
        self.velocity_hidden2_output = self.momentum * self.velocity_hidden2_output - \
                                       learning_rate * np.dot(self.hidden_output2.T, output_delta)
        self.weights_hidden2_output += self.velocity_hidden2_output
        self.bias_output -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        self.velocity_hidden1_hidden2 = self.momentum * self.velocity_hidden1_hidden2 - \
                                        learning_rate * np.dot(self.hidden_output1.T, hidden2_delta)
        self.weights_hidden1_hidden2 += self.velocity_hidden1_hidden2
        self.bias_hidden2 -= learning_rate * np.sum(hidden2_delta, axis=0, keepdims=True)

        self.velocity_input_hidden1 = self.momentum * self.velocity_input_hidden1 - \
                                      learning_rate * np.dot(X.T, hidden1_delta)
        self.weights_input_hidden1 += self.velocity_input_hidden1
        self.bias_hidden1 -= learning_rate * np.sum(hidden1_delta, axis=0, keepdims=True)

        ''' # 更新權重和偏差，不使用 momentum
        self.weights_hidden2_output -= learning_rate * np.dot(self.hidden_output2.T, output_delta)
        self.bias_output -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.weights_hidden1_hidden2 -= learning_rate * np.dot(self.hidden_output1.T, hidden2_delta)
        self.bias_hidden2 -= learning_rate * np.sum(hidden2_delta, axis=0, keepdims=True)
        self.weights_input_hidden1 -= learning_rate * np.dot(X.T, hidden1_delta)
        self.bias_hidden1 -= learning_rate * np.sum(hidden1_delta, axis=0, keepdims=True)
        '''

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            predictions = self.forward(X)
            loss = mean_squared_error(y, predictions)
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

def MLP_prediction(state, name):
    if name == "train4dAll":
        state = (state - X_mean1) / X_std1
        prediction = mlp4d.forward(state) * y_std1 + y_mean1 + 40
    elif name == "train6dAll":
        state = (state - X_mean2) / X_std2
        prediction = mlp6d.forward(state) * y_std2 + y_mean2 + 40
    return prediction

# train 4D
data = np.loadtxt('train4dAll.txt')
# data proprocessing
X = data[:, :-1]
y = data[:, -1:]

X_mean1 = np.mean(X, axis=0)
X_std1 = np.std(X, axis=0)

y_mean1 = np.mean(y, axis=0)
y_std1 = np.std(y, axis=0)

X = (X - X_mean1) / X_std1
y = (y - y_mean1) / y_std1

mlp4d = MLPRegressor(X.shape[1], 5, 3, 1, 0.9)
mlp4d.train(X, y, epochs=600, learning_rate=0.0001)

# train 6D
data = np.loadtxt('train6dAll.txt')
# data proprocessing
X = data[:, :-1]
y = data[:, -1:]

X_mean2 = np.mean(X, axis=0)
X_std2 = np.std(X, axis=0)

y_mean2 = np.mean(y, axis=0)
y_std2 = np.std(y, axis=0)

X = (X - X_mean2) / X_std2
y = (y - y_mean2) / y_std2

mlp6d = MLPRegressor(X.shape[1], 5, 2, 1, 0.9)
mlp6d.train(X, y, epochs=900, learning_rate=0.0001)
