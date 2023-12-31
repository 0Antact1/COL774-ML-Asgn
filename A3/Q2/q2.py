#!/usr/bin/env python
# coding: utf-8

# # Q2. Neural Networks

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# (a) Custom Neural Network

# In[2]:


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_grad(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - s.T@s

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_grad(x):
    return sigmoid(x)*(1-sigmoid(x))

def cross_entropy_loss(p, q, eps=1e-12):
    m = p.shape[0]
    loss = -np.sum(p * np.log(q + eps)) / m
    return loss

def cross_entropy_grad_netk(p, q):
    # because p[k] = 1 iff k is true label
    return q - p


# In[3]:


# TODO: check error derivative
# DOne: checked init (not fine)


# In[72]:


class ANN:
    def __init__(self, n: int, hidden_size: list, r: int):
        self.input_size = n
        self.hidden_size = hidden_size
        self.output_size = r

        # TODO: try Xavier init

        # list of inter-layer weights : 
        std_init = np.sqrt(2/(n + hidden_size[0]))
        self.weights = [np.random.normal(0, std_init, size=(n, hidden_size[0]))]
        # self.weights = [np.random.randn(n, hidden_size[0])]   # init=randn

        # list of inter-layer biases : init=0
        self.biases = [np.zeros((1,hidden_size[0]))]

        for i in range(1,len(hidden_size)):
            std_init = np.sqrt(2/(hidden_size[i-1] + hidden_size[i]))
            self.weights.append(np.random.normal(0, std_init, size=(hidden_size[i-1], hidden_size[i])))
            self.biases.append(np.zeros((1, hidden_size[i])))

        std_init = np.sqrt(2/(hidden_size[-1] + r))
        self.weights.append(np.random.normal(0, std_init, size=(hidden_size[-1], r)))

        self.biases.append(np.zeros((1, r)))

        self.num_layers = len(self.biases)


    def fwd_prop(self, X):
        # list of layer outputs
        self.z = []
        self.actv = []

        A = X
        
        self.actv.append(A)
        self.z.append(np.eye(A.shape[0], A.shape[1]))        # to maintain same length

        for i in range(self.num_layers):
            Z = A@self.weights[i] + self.biases[i]
            self.z.append(Z)

            if i == self.num_layers-1:
                A = softmax(Z)
            else:
                A = sigmoid(Z)

            self.actv.append(A)

        # len(self.actv) == self.num_layers + 1 == len(self.weights) + 1
        return A


    def bwd_prop(self, y_true, lr):
        m = y_true.shape[0]
        y_pred = self.actv[-1]

        self.w_grad = []
        self.b_grad = []

        dZ = cross_entropy_grad_netk(y_true, y_pred)
        # dActv = softmax_grad(self.z[-1]).mean(axis=0)

        # print(dActv.shape)
        # print(dZ.shape)

        dB = np.mean(dZ, axis=0, keepdims=True)
        self.biases[-1] -= lr*dB
        self.b_grad.append(dB)

        dW = self.actv[-2].T@dZ/m

        # print(dZ.shape)
        # print(dW.shape)

        self.weights[-1] -= lr*dW
        self.w_grad.append(dW)

        for l in range(self.num_layers-2,-1,-1):
            dJd = dZ@self.weights[l+1].T
            # dActv = sigmoid_grad(self.z[l+1]).mean(axis=0)        
            dActv = (self.actv[l+1] * (1 - self.actv[l+1])).mean(axis=0)        # same thing
    
            dZ = dJd*dActv
            dW = self.actv[l].T@dZ/m
            self.w_grad.append(dW)
            
            dB = np.mean(dZ, axis=0, keepdims=True)
            self.b_grad.append(dB)

            self.weights[l] -= lr*dW
            self.biases[l] -= lr*dB

        self.w_grad.reverse()
        self.b_grad.reverse()

        return self.w_grad, self.b_grad

    # SUS: performs better when not updating weights[1] ?
    # def bwd_prop_old(self, y_true, lr):
    #     m = y_true.shape[0]
    #     y_pred = self.actv[-1]

    #     dZ = y_pred - y_true
    #     dW = np.dot(self.actv[-2].T, dZ) / m
    #     db = np.sum(dZ, axis=0, keepdims=True) / m
    #     dA = np.dot(dZ, self.weights[-1].T)

    #     self.weights[-1] -= lr * dW
    #     self.biases[-1] -= lr * db

    #     for i in range(len(self.weights)-2, -1, -1):
    #         dZ = dA * self.actv[i+1] * (1 - self.actv[i+1])
    #         dW = np.dot(self.actv[i].T, dZ) / m
    #         db = np.sum(dZ, axis=0, keepdims=True) / m
    #         dA = np.dot(dZ, self.weights[i].T)

    #         # Update weights and biases
    #         self.weights[i] -= lr * dW
    #         self.biases[i] -= lr * db


    def train(self, X, y_true, epochs=100, tol=1e-4, batch_size=32, lr=0.01, adaptive=False):
        m = X.shape[0]
        num_batches = m//batch_size

        loss = cross_entropy_loss(y_true, self.fwd_prop(X))
        print(f"Epoch 0/{epochs} - Loss: {loss:.4f}")
        
        # for i in range(self.num_layers):
        #     print('Initial w')
        #     print(self.weights[i])
        
        for epoch in range(epochs):
            #TODO: shuffle
            # permute_idx = np.random.permutation(len(y_true))
            # X = X[permute_idx]
            # y_true = y_true[permute_idx]
            
            loss_old = loss

            for i in range(num_batches):
                start = i*batch_size
                end = start+batch_size

                X_batch = X[start:end]
                y_batch = y_true[start:end]

                # Forward propagation
                y_pred = self.fwd_prop(X_batch)
                # print(self.actv)

                # Backward propagation
                if adaptive:
                    lr_adapt = lr/np.sqrt(epoch+1)
                    # if new:
                    #     self.bwd_prop_new(y_batch, lr_adapt)
                    # else:
                    self.bwd_prop(y_batch, lr_adapt)
                else:
                    # if new:
                    #     self.bwd_prop_new(y_batch, lr)
                    # else:
                    self.bwd_prop(y_batch, lr)

            # Compute loss after each epoch
            y_pred = self.fwd_prop(X)
            loss = cross_entropy_loss(y_true, y_pred)            

            if (epoch+1)%10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

            if loss_old-loss < tol:
                print(f'Within tolerance = {tol}')
                return


    def predict(self, X, out_matrix=False):
        y_pred = self.fwd_prop(X)
        if out_matrix:
            return y_pred

        return np.argmax(y_pred, axis=1)+1


# In[4]:


def get_data(x_path, y_path):
    '''
    Args:
        x_path: path to x file
        y_path: path to y file
    Returns:
        x: np array of [NUM_OF_SAMPLES x n]
        y: np array of [NUM_OF_SAMPLES]
    '''
    x = np.load(x_path)
    y = np.load(y_path)

    y = y.astype('float')
    x = x.astype('float')

    #normalize x:
    x = 2*(0.5 - x/255)
    return x, y


# In[5]:


from sklearn.metrics import classification_report

def get_metric(y_true, y_pred):
    results = classification_report(y_pred, y_true)
    print(results)


# In[6]:


X_train, y_train = get_data('../part_b/x_train.npy', '../part_b/y_train.npy')
X_test, y_test = get_data('../part_b/data_test/x_test.npy', '../part_b/data_test/y_test.npy')


# In[7]:


from sklearn.preprocessing import OneHotEncoder

# enc = OneHotEncoder()
label_encoder = OneHotEncoder(sparse=False)
label_encoder.fit(np.expand_dims(y_train, axis = -1))


# In[13]:


# y_train_oh = enc.fit_transform(y_train.reshape(-1,1)).toarray()
# y_test_oh = enc.transform(y_test.reshape(-1,1)).toarray()

# y_test_oh.shape


# In[8]:


y_train_oh = label_encoder.transform(np.expand_dims(y_train, axis = -1))
y_test_oh = label_encoder.transform(np.expand_dims(y_test, axis = -1))


# In[9]:


y_test_oh


# In[10]:


def get_f1score(y_true, y_pred):
    f1_scores_dict = [v for v in classification_report(y_true, y_pred, output_dict=True).values()]
    f1_score = sum([label['f1-score'] for label in f1_scores_dict[:5]])/5

    return f1_score


# - Testing ANN

# In[23]:


nn = ANN(1024, [100], 5)
nn.train(X_train, y_train_oh, batch_size=32, lr=0.01, new=True)


# In[35]:


y_pred = nn.predict(X_train)
get_metric(y_train, y_pred)


# In[28]:


y_pred = nn.predict(X_test)
get_metric(y_test, y_pred)


# In[30]:


print(np.unique(y_pred, return_counts=1), np.unique(y_test, return_counts=1))


# In[26]:


nnn = ANN(1024, [100], 5)
nnn.train(X_train, y_train_oh, epochs=500, batch_size=32, lr=0.01)


# In[31]:


y_pred_500 = nnn.predict(X_test)
get_metric(y_test, y_pred_500)


# Test accuracy on hidden layer [100]:
# - Epochs=100 : 67% accuracy, 0.7278 ce-loss
# - Epochs=500 : 72% accuracy, 0.6146 ce-loss

# In[27]:


p = np.array([0, 1, 0])
q = np.array([0, 0, 1])
cross_entropy_loss(p, q)


# (b) Single Layer ANN

# In[32]:


hidden_size_list = [[1],[5],[10],[50],[100]]

train_f1_score_list_uni = []
test_f1_score_list_uni = []

for hidden_size in hidden_size_list:
    nn_uni = ANN(1024, hidden_size, 5)

    nn_uni.train(X_train, y_train_oh, epochs=200)

    print(f"For hidden_size: {hidden_size}:")

    print("Train scores:-")
    y_pred_uni = nn_uni.predict(X_train)
    get_metric(y_train, y_pred_uni)

    f1_score = get_f1score(y_train, y_pred_uni)
    train_f1_score_list_uni.append(f1_score)

    print("Test scores:-")
    y_pred_uni = nn_uni.predict(X_test)
    get_metric(y_test, y_pred_uni)

    f1_score = get_f1score(y_test, y_pred_uni)
    test_f1_score_list_uni.append(f1_score)

    print(f"{hidden_size} done")


# In[33]:


fig, ax = plt.subplots()

ax.plot(hidden_size_list, train_f1_score_list_uni, label="Train")
ax.plot(hidden_size_list, test_f1_score_list_uni, label="Test")

plt.legend()
plt.title('F1-scores with hidden units')
plt.xlabel('Number of hidden units')
plt.ylabel('Accuracy')
plt.xscale('log')


# (c) Multilayer ANN

# In[34]:


hidden_size_list = [[512],[512,256],[512,256,128],[512,256,128,64]]

train_f1_score_list_mul = []
test_f1_score_list_mul = []
for hidden_size in hidden_size_list:
    nn_mul = ANN(1024, hidden_size, 5)

    # try more epochs for larger net?
    nn_mul.train(X_train, y_train_oh, epochs=500)

    print(f"For hidden_size: {hidden_size}:")

    print("Train scores:-")
    y_pred_mul = nn_mul.predict(X_train)
    get_metric(y_train, y_pred_mul)

    f1_score = get_f1score(y_train, y_pred_mul)
    train_f1_score_list_mul.append(f1_score)

    print("Test scores:-")
    y_pred_mul = nn_mul.predict(X_test)
    get_metric(y_test, y_pred_mul)

    f1_score = get_f1score(y_test, y_pred_mul)
    test_f1_score_list_mul.append(f1_score)


# In[43]:


hidden_size = [512, 256, 128, 64]

nn_mul = ANN(1024, hidden_size, 5)

# try more epochs for larger net?
nn_mul.train(X_train, y_train_oh, epochs=500, tol=1e-6)

print(f"For hidden_size: {hidden_size}:")

print("Train scores:-")
y_pred_mul = nn_mul.predict(X_train)
get_metric(y_train, y_pred_mul)

f1_score = get_f1score(y_train, y_pred_mul)
train_f1_score_list_mul[-1] = (f1_score)

print("Test scores:-")
y_pred_mul = nn_mul.predict(X_test)
get_metric(y_test, y_pred_mul)

f1_score = get_f1score(y_test, y_pred_mul)
test_f1_score_list_mul[-1] = (f1_score)


# In[44]:


train_f1_score_list_mul


# In[59]:


plt.bar(np.arange(4) - 0.2, train_f1_score_list_mul, 0.4, label="Train")
plt.bar(np.arange(4) + 0.2, test_f1_score_list_mul, 0.4, label="Test")

plt.xticks(np.arange(4), hidden_size_list)
plt.rc('xtick', labelsize=16)
plt.title('F1-scores with network depth')
plt.xlabel('Hidden Layers')
plt.ylabel('Accuracy')
plt.legend()


# (d) Adaptive Learning Rate

# In[61]:


hidden_size_list = [[512],[512,256],[512,256,128],[512,256,128,64]]

train_f1_score_list_adpt = []
test_f1_score_list_adpt = []
for hidden_size in hidden_size_list:
    nn_adpt = ANN(1024, hidden_size, 5)

    # try more epochs for larger net?
    nn_adpt.train(X_train, y_train_oh, epochs=1000, tol=1e-6, adaptive=True)

    print(f"For hidden_size: {hidden_size}:")

    print("Train scores:-")
    y_pred_adpt = nn_adpt.predict(X_train)
    get_metric(y_train, y_pred_adpt)

    f1_score = get_f1score(y_train, y_pred_adpt)
    train_f1_score_list_adpt.append(f1_score)

    print("Test scores:-")
    y_pred_adpt = nn_adpt.predict(X_test)
    get_metric(y_test, y_pred_adpt)

    f1_score = get_f1score(y_test, y_pred_adpt)
    test_f1_score_list_adpt.append(f1_score)


# In[66]:


plt.bar(np.arange(2) - 0.2, train_f1_score_list_adpt, 0.4, label="Train")
plt.bar(np.arange(2) + 0.2, test_f1_score_list_adpt, 0.4, label="Test")

plt.xticks(np.arange(2), hidden_size_list[:2])
plt.rc('xtick', labelsize=4)
plt.title('F1-scores with network depth')
plt.xlabel('Hidden Layers')
plt.ylabel('Accuracy')
plt.legend()


# Only ran 2 of 4 models above, doesn't seem to run well with a reduced learning rate.

# In[67]:


hidden_size_list = [[512],[512,256],[512,256,128],[512,256,128,64]]

train_f1_score_list_adpt = []
test_f1_score_list_adpt = []
for hidden_size in hidden_size_list:
    nn_adpt = ANN(1024, hidden_size, 5)

    # try more epochs for larger net?
    nn_adpt.train(X_train, y_train_oh, epochs=1000, tol=1e-6, lr=0.5, adaptive=True)

    print(f"For hidden_size: {hidden_size}:")

    print("Train scores:-")
    y_pred_adpt = nn_adpt.predict(X_train)
    get_metric(y_train, y_pred_adpt)

    f1_score = get_f1score(y_train, y_pred_adpt)
    train_f1_score_list_adpt.append(f1_score)

    print("Test scores:-")
    y_pred_adpt = nn_adpt.predict(X_test)
    get_metric(y_test, y_pred_adpt)

    f1_score = get_f1score(y_test, y_pred_adpt)
    test_f1_score_list_adpt.append(f1_score)


# In[70]:


plt.bar(np.arange(4) - 0.2, train_f1_score_list_adpt, 0.4, label="Train")
plt.bar(np.arange(4) + 0.2, test_f1_score_list_adpt, 0.4, label="Test")

plt.xticks(np.arange(4), hidden_size_list)
plt.rc('xtick', labelsize=8)
plt.title('F1-scores with network depth (adaptive lr=0.5/sqrt(n))')
plt.xlabel('Hidden Layers')
plt.ylabel('Accuracy')
plt.ylim(0.4,1)
plt.legend()


# The learning rate used seems to work much better than with seed=0.01. All the models gain much better scores than before. We can see an increasing trend with network depth, as the f1 scores go from below 80 to 90+. But it falls off at [512,256,128,64], most likely because the learning rate we chose for this model was too large(can be inferred from the losses increasing near the epoch ends), that can be controlled by stopping our model when loss function increases. (Infact, the 3rd model also showed min loss = 0.1964)
# Nevertheless, running for more than 500 epochs was useful since test results improve, suggesting an underfit earlier. The adaptive lr also helps deal with overfitting here.

# (e) RelU Activation

# In[125]:


def relu(x, downscale=1):
    return np.where(x>0, x*downscale, 0)

def relu_grad(x, downscale=1):
    # subderivative at x=0 is set to 1(right limit)
    return np.where(x>=0, downscale, 0)


# In[99]:


class ANN_Relu(ANN):
    def fwd_prop(self, X):
        # list of layer outputs
        self.z = []
        self.actv = []

        A = X
        
        self.actv.append(A)
        self.z.append(np.eye(A.shape[0], A.shape[1]))        # to maintain same length

        for i in range(self.num_layers):
            Z = A@self.weights[i] + self.biases[i]
            self.z.append(Z)

            if i == self.num_layers-1:
                A = softmax(Z)
            else:
                A = relu(Z)

            self.actv.append(A)

        # len(self.actv) == self.num_layers + 1 == len(self.weights) + 1
        return A
    
    def bwd_prop(self, y_true, lr):
        m = y_true.shape[0]
        y_pred = self.actv[-1]

        self.w_grad = []
        self.b_grad = []

        dZ = cross_entropy_grad_netk(y_true, y_pred)
        # dActv = softmax_grad(self.z[-1]).mean(axis=0)

        # print(dActv.shape)
        # print(dZ.shape)

        dB = np.mean(dZ, axis=0, keepdims=True)
        self.biases[-1] -= lr*dB
        self.b_grad.append(dB)

        dW = self.actv[-2].T@dZ/m


        # print(dW[0])
        # print(dW.shape)

        self.weights[-1] -= lr*dW
        self.w_grad.append(dW)

        for l in range(self.num_layers-2,-1,-1):
            dJd = dZ@self.weights[l+1].T
            dActv = relu_grad(self.z[l+1]).mean(axis=0)
    
            dZ = dJd*dActv
            dW = self.actv[l].T@dZ/m
            self.w_grad.append(dW)
            
            dB = np.mean(dZ, axis=0, keepdims=True)
            self.b_grad.append(dB)

            self.weights[l] -= lr*dW
            self.biases[l] -= lr*dB

        self.w_grad.reverse()
        self.b_grad.reverse()

        return self.w_grad, self.b_grad


# In[75]:


import time


# In[104]:


hidden_size_list = [[512],[512,256],[512,256,128],[512,256,128,64]]

train_f1_score_list_relu = []
test_f1_score_list_relu = []
for hidden_size in hidden_size_list:
    t = time.time()
    nn_relu = ANN_Relu(1024, hidden_size, 5)

    # try more epochs for larger net?
    nn_relu.train(X_train, y_train_oh, epochs=500, tol=1e-6, lr=0.01, adaptive=True)

    print(f"For hidden_size: {hidden_size}:")

    print("Train scores:-")
    y_pred_relu = nn_relu.predict(X_train)
    get_metric(y_train, y_pred_relu)

    f1_score = get_f1score(y_train, y_pred_relu)
    train_f1_score_list_relu.append(f1_score)

    print("Test scores:-")
    y_pred_relu = nn_relu.predict(X_test)
    get_metric(y_test, y_pred_relu)

    f1_score = get_f1score(y_test, y_pred_relu)
    test_f1_score_list_relu.append(f1_score)

    print(f"Time taken for {hidden_size}= {time.time()-t}")


# Reduced the learning rate back to 0.01 because it was exploding the weights(presumably because ReLU isn't bound like sigmoid). Also, reduced the epoch limit back to 500 since ReLU should ideally take lesser time to converge. 

# In[107]:


plt.bar(np.arange(4) - 0.2, train_f1_score_list_relu, 0.4, label="Train")
plt.bar(np.arange(4) + 0.2, test_f1_score_list_relu, 0.4, label="Test")

plt.xticks(np.arange(4), hidden_size_list)
plt.rc('xtick', labelsize=8)
plt.title('F1-scores with network depth (adaptive lr=0.5/sqrt(n))')
plt.xlabel('Hidden Layers')
plt.ylabel('Accuracy')
plt.legend()


# In[ ]:


# fig, ax = plt.subplots()

# ax.plot(hidden_size_list, train_f1_score_list_relu, label="Train")
# ax.plot(hidden_size_list, test_f1_score_list_relu, label="Test")

# plt.title('F1-scores with network depth (adaptive lr=0.5/sqrt(n))')
# plt.xlabel('Network depth')
# plt.ylabel('F1-score')


# We see f1-scores around the 70% mark, improving faintly with network depth. The last model converges before 500 epochs as the error falls into the tolerance limit, making it run for almost half the time of the previous one, while giving almost the same results.

# Compared to the sigmoid-based models, the relu-based models do infact seem to perform better If we compare results with same learning seed = 0.01, there is an increase of around 4% over their sigmoid counterparts. It would be interesting to see the performance of relu model with higher lr(need to figure out a solution to blowing-up gradients, probably tweaking units to have falling weights https://stackoverflow.com/questions/37232782/nan-loss-when-training-regression-network)

# In[116]:


asa = np.random.randn(2,5)
wasa = asa - asa.max(axis=1, keepdims=True)
print(asa)
print(wasa)


# In[119]:


print(softmax(asa))
print(softmax(wasa))


# One solution here seems to subtract the max value so that softmax layer doesn't overflow. (Edit: Other layers do, still might be a potential solution)

# In[135]:


class ANN_Relu_Scale(ANN_Relu):
    def fwd_prop(self, X):
        # list of layer outputs
        self.z = []
        self.actv = []

        A = X
        
        self.actv.append(A)
        self.z.append(np.eye(A.shape[0], A.shape[1]))        # to maintain same length

        for i in range(self.num_layers):
            Z = A@self.weights[i] + self.biases[i]
            self.z.append(Z)

            if i == self.num_layers-1:
                # Z_modifive = Z - Z.max(axis=1, keepdims=True)
                A = softmax(Z)
                # A = softmax(Z_modifive)
            else:
                # also in previous layers?
                # Z_modifive = Z - Z.max(axis=1, keepdims=True)
                A = relu(Z, downscale=1e-2)

            self.actv.append(A)

        # len(self.actv) == self.num_layers + 1 == len(self.weights) + 1
        return A
    
    def bwd_prop(self, y_true, lr):
        m = y_true.shape[0]
        y_pred = self.actv[-1]

        self.w_grad = []
        self.b_grad = []

        dZ = cross_entropy_grad_netk(y_true, y_pred)
        # dActv = softmax_grad(self.z[-1]).mean(axis=0)

        # print(dActv.shape)
        # print(dZ.shape)

        dB = np.mean(dZ, axis=0, keepdims=True)
        self.biases[-1] -= lr*dB
        self.b_grad.append(dB)

        dW = self.actv[-2].T@dZ/m


        # print(dW[0])
        # print(dW.shape)

        self.weights[-1] -= lr*dW
        self.w_grad.append(dW)

        for l in range(self.num_layers-2,-1,-1):
            dJd = dZ@self.weights[l+1].T
            dActv = relu_grad(self.z[l+1], downscale=1e-2).mean(axis=0)
    
            dZ = dJd*dActv
            dW = self.actv[l].T@dZ/m
            self.w_grad.append(dW)
            
            dB = np.mean(dZ, axis=0, keepdims=True)
            self.b_grad.append(dB)

            self.weights[l] -= lr*dW
            self.biases[l] -= lr*dB

        self.w_grad.reverse()
        self.b_grad.reverse()

        return self.w_grad, self.b_grad


# Current relu uses a downscale factor of 0.01 (tried others, 0.1 blows up and 0.001 vanishes).

# In[136]:


hidden_size_list = [[512],[512,256],[512,256,128],[512,256,128,64]]

train_f1_score_list_relu_s = []
test_f1_score_list_relu_s = []
for hidden_size in hidden_size_list:
    t = time.time()
    nn_relu = ANN_Relu_Scale(1024, hidden_size, 5)

    # try more epochs for larger net?
    nn_relu.train(X_train, y_train_oh, epochs=500, tol=1e-6, lr=0.5, adaptive=True)

    print(f"For hidden_size: {hidden_size}:")

    print("Train scores:-")
    y_pred_relu = nn_relu.predict(X_train)
    get_metric(y_train, y_pred_relu)

    f1_score = get_f1score(y_train, y_pred_relu)
    train_f1_score_list_relu_s.append(f1_score)

    print("Test scores:-")
    y_pred_relu = nn_relu.predict(X_test)
    get_metric(y_test, y_pred_relu)

    f1_score = get_f1score(y_test, y_pred_relu)
    test_f1_score_list_relu_s.append(f1_score)

    print(f"Time taken for {hidden_size}= {time.time()-t}")


# In[137]:


print(train_f1_score_list_relu_s)
print(test_f1_score_list_relu_s)


# Only the single layered model runs properly on this downscaled relu function.

# (f) MLP Classifier

# In[108]:


from sklearn.neural_network import MLPClassifier


# In[109]:


hidden_size_list = [[512],[512,256],[512,256,128],[512,256,128,64]]

train_f1_score_list = []
test_f1_score_list = []

for hidden_size in hidden_size_list:
    mlp_clf = MLPClassifier(hidden_layer_sizes=hidden_size, activation='relu', solver='sgd', batch_size=32, alpha=0, learning_rate='invscaling')

    mlp_clf.fit(X_train, y_train)

    print(f"For hidden_size: {hidden_size}:")

    print("Train scores:-")
    y_pred_mlp = mlp_clf.predict(X_train)
    get_metric(y_train, y_pred_mlp)

    f1_score = get_f1score(y_train, y_pred_mlp)
    train_f1_score_list.append(f1_score)

    print("Test scores:-")
    y_pred_mlp = mlp_clf.predict(X_test)
    get_metric(y_test, y_pred_mlp)

    f1_score = get_f1score(y_test, y_pred_mlp)
    test_f1_score_list.append(f1_score)


# In[113]:


fig, ax = plt.subplots()

ax.plot(np.arange(1,5), train_f1_score_list, label="Train")
ax.plot(np.arange(1,5), test_f1_score_list, label="Test")

plt.title('F1-scores with network depth')
plt.xlabel('Network depth')
plt.ylabel('Accuracy')

