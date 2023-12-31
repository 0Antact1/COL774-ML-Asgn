#!/usr/bin/env python
# coding: utf-8

# # Q2. Image Classification

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import time


# In[2]:


from cvxopt import matrix, solvers
from sklearn import svm
from PIL import Image


# In[180]:


def generate_dataset(n,dir="train",img_dim=(16,16),en_digit=6):
    img_data = []
    img_labels = []

    for i in range(n):
        x = (en_digit+i)%6
        img_dir = f"../ml-svm/{dir}/{x}"

        for file in os.listdir(img_dir):
            img_path = os.path.join(img_dir,file)
            img = Image.open(img_path)
            
            img_res = img.resize(img_dim)
            img_arr = np.array(img_res).flatten()/255

            img_data.append(img_arr)

        len_diff = (len(img_data)-len(img_labels))
        img_labels += [i]*len_diff
        
        print(x,"done")

    X_ = np.vstack(img_data)
    y_ = np.array(img_labels).reshape(-1,1)
    print(X_.shape, y_.shape)

    return X_, y_


# (a) cvxopt solver with linear kernel

# In[6]:


# kernelsx
def linearKernel(x,z):
    # return m*m matrix
    return x@z.T

def polynomialKernel(x,z,c=3,p=2):
    return (c+x@z.T)**p


# In[7]:


# A = np.array([[2,5],[5,-2]])

# vec_ker = np.vectorize(linearKernel, signature="()")
# aks = vec_ker(A,A)
# aks


# In[102]:


X, y = generate_dataset(2)


# In[8]:


X_val, y_val = generate_dataset(2,dir="val")


# In[10]:


class SVM:
    def __init__(self, kerfx = linearKernel, c=1):
        self.ker = kerfx
        self.C = c


    def fit(self, X: np.ndarray, y:np.ndarray, autoparam=True):
        m, n = X.shape
        
        # Gram matrix??
        # K = np.zeros((m, m))
        # for i in range(m):
        #     for j in range(m):
        #         K[i,j] = self.ker(X[i], X[j])

        K = self.ker(X,X)

        P = matrix((y@y.T)*K, tc='d')
        q = matrix(-np.ones(m), tc='d')
        A = matrix(y.T, tc='d')     
        b = matrix([0], tc='d')

        if self.C is None:
            G = matrix(-np.eye(m), tc='d')
            h = matrix(np.zeros(m), tc='d')
        else:
            G = matrix(np.vstack([-np.eye(m), np.eye(m)]), tc='d')
            h = matrix(np.hstack([np.zeros(m), self.C*np.ones(m)]), tc='d')
            # h = matrix(np.hstack([np.zeros(m), self.C*np.ones(m)]).T, tc='d')

        solve = solvers.qp(P,q,G,h,A,b)
        alpha = np.ravel(solve['x'])

        # finding support vectors
        sv = (alpha>1e-6)
        self.sv_idx = np.arange(len(alpha))[sv]

        self.sv_alpha = alpha[sv]
        self.sv_x = X[sv]
        self.sv_y = y[sv]
        self.sv_count = len(self.sv_alpha)

        sv_perc = round(self.sv_count*100/m, 2)
        print(f"{self.sv_count} support vectors out of {m} samples ({sv_perc}% sv)")
        print()

        if autoparam:
            self.wandb()


    def wandb(self, elaborate=True):
        self.sv_alpha = self.sv_alpha.reshape(-1,1)

        # finding bias
        ker_sv = self.ker(self.sv_x,self.sv_x)
        b_sum = self.sv_y - np.sum(ker_sv * self.sv_alpha * self.sv_y, axis=0)
        self.b = np.mean(b_sum)

        # finding weights
        if self.ker == linearKernel:
            self.w = self.sv_x.T @ (self.sv_alpha*self.sv_y)
        else:
            # self.w = np.array(None)
            self.w = None

        if elaborate:
            print("Weights:\n", self.w)
            print("Bias: ", self.b)
            print()


    def predict(self, X_val: np.ndarray):
        if self.w is not None:
            project = X_val@self.w + self.b
        
        else:
            ker_sv_val = self.ker(self.sv_x,X_val)
            vec_prod = self.sv_alpha*self.sv_y
            # print(ker_sv_val.shape)
            # print(vec_prod.shape)
            
            y_p = np.sum(ker_sv_val*vec_prod,axis=0).reshape(-1,1)
            # print(y_p.shape)
            
            # y_p = np.zeros(len(X_val))
            # for i in range(len(X_val)):
            #     s = 0
            #     for alp, sv_y, sv_x in zip(self.alpha, self.sv_y, self.sv_x):
            #         s += alp*sv_y*self.ker(X_val[i],sv_x)
                
            #     y_p[i] = s
                
            project = y_p + self.b

        return np.sign(project)


# In[11]:


# compare compute time
t0 = time.time()

svm_linear = SVM()
svm_linear.fit(X,y)

print("Time taken using linear kernel = ", time.time()-t0)


# In[ ]:


svm_linear.wandb()


# In[17]:


# yd = np.sign(X@svm_linear.w + svm_linear.b)

# y_pred not work(for some reason)
# but sgn(w.Tx+b) works
y_predt = svm_linear.predict(X)

accu_train = np.mean(y_predt == y)
print("Train accuracy =", accu_train)


# In[18]:


# yss = np.sign(X_val@svm_linear.w + svm_linear.b)
y_predv = svm_linear.predict(X_val)

accu_val = np.mean(y_predv == y_val)
print("Test accuracy =", accu_val)


# In[19]:


# what is top 6 coeff?
top_idx = np.argpartition(svm_linear.sv_alpha.flatten(), -6)[-6:]
top_idx


# In[20]:


sv6_lin = svm_linear.sv_x[top_idx]

for i in range(6):
    sv6 = (sv6_lin[i]*255).reshape(16,16,3)
    # print(sv6.shape)
    sv6 = sv6.astype(np.uint8)
    img_sv = Image.fromarray(sv6)
    img_sv.save(f"./q2a/q2a_sv{i}.png")


# In[21]:


ww = (svm_linear.w*255).reshape(16,16,3)
ww = ww.astype(np.uint8)
img_w = Image.fromarray(ww)
img_w.save("./q2a/q2a_w.png")


# (b) cvxopt solver with gaussian kernel

# In[22]:


def gaussianKernel(x,z,gamma=0.001):
    m1 = x.shape[0]
    m2 = z.shape[0]
    
    x_norm_vec = np.sum(x**2,axis=1).reshape(-1,1)
    z_norm_vec = np.sum(z**2,axis=1).reshape(-1,1)
    # print(x_norm_vec.shape)
    
    x_prod = x_norm_vec@np.ones(m2).reshape(1,-1)
    # print(x_prod.shape)
    z_prod = z_norm_vec@np.ones(m1).reshape(1,-1)
    
    return np.exp(-gamma*(x_prod + z_prod.T - 2*x@z.T))


# In[21]:


# np.ones(3).reshape(1,-1)

# x = np.array([[1,2,3],[2,4,0],[34,11,11]])
# z = np.array([[5,2,1],[1,31,1]])
# xz = gaussianKernel(x,z)
# xz


# In[23]:


t0 = time.time()

svm_gaussian = SVM(gaussianKernel)
svm_gaussian.fit(X,y)

print("Time taken using Gaussian kernel = ", time.time()-t0)


# Finding common SVs

# In[30]:


# len(np.intersect1d(svm_linear.sv_x,svm_gaussian.sv_x))
num_sv = np.sum(np.logical_and(svm_linear.sv_idx,svm_gaussian.sv_idx))
# num_sv = len(np.intersect1d(svm_linear.sv_idx,svm_gaussian.sv_idx))
print("Number of SVs common between Linear and Gaussian = ", num_sv)


# In[31]:


y_predt_gaus = svm_gaussian.predict(X)

accu_train = np.mean(y_predt_gaus == y)
print("Train accuracy =", accu_train)


# In[32]:


y_predv_gaus = svm_gaussian.predict(X_val)

accu_val = np.mean(y_predv_gaus == y_val)
print("Test accuracy =", accu_val)


# Validation accuracy of Gaussian and Linear are very similar, around 85% for both. We found 1269 common SVs between kernels.

# In[71]:


# asa = SVM(gaussianKernel)
# asa.fit(X_val,y_val)

# y_predv = asa.predict(X_val)
# accu_val = np.mean(y_predv == y_val)
# accu_val


# In[33]:


top_idx = np.argpartition(svm_gaussian.sv_alpha.flatten(), -6)[-6:]
top_idx


# In[34]:


sv6_gsn = svm_gaussian.sv_x[top_idx]

for i in range(6):
    sv6 = (sv6_gsn[i]*255).reshape(16,16,3)
    # print(sv6.shape)
    sv6 = sv6.astype(np.uint8)
    img_sv = Image.fromarray(sv6)
    img_sv.save(f"./q2b/q2b_sv{i}.png")


# (c) scikit-learn SVM model

# In[35]:


t0 = time.time()

svm_linear_skl = svm.SVC(kernel='linear')
svm_linear_skl.fit(X,y.flatten())

print("Time taken by linear Scikit-learn SVM = ", time.time()-t0)


# In[36]:


y_predt_skl = svm_linear_skl.predict(X)

# transpose to get row vec
accu_train_skl = np.mean(y_predt_skl == y.T)
print("Train(lin) accuracy = ", accu_train_skl)

y_predv_skl = svm_linear_skl.predict(X_val)

accu_val_skl = np.mean(y_predv_skl == y_val.T)
print("Test(lin) accuracy = ", accu_val_skl)


# In[44]:


print("No. of SVs in Scikit-learn SVM(lin) = ", len(svm_linear_skl.support_))


# In[37]:


t0 = time.time()

# default kernel is rbf = gaussian
svm_gaussian_skl = svm.SVC(gamma=0.001)
svm_gaussian_skl.fit(X,y.flatten())

print("Time taken by gaussian Scikit-learn SVM = ", time.time()-t0)


# In[38]:


y_predt_skl = svm_gaussian_skl.predict(X)

# transpose to get row vec
accu_train_skl = np.mean(y_predt_skl == y.T)
print("Train(rbf) accuracy = ", accu_train_skl)

y_predv_skl = svm_gaussian_skl.predict(X_val)

accu_val_skl = np.mean(y_predv_skl == y_val.T)
print("Test(rbf) accuracy = ", accu_val_skl)


# In[45]:


print("No. of SVs in Scikit-learn SVM(rbf) = ", len(svm_gaussian_skl.support_))


# In the scikit-learn implementation, we see a decent increase in accuracy of the rbf kernel compared to linear.
# 
# No. of SVs for linear is similar, 1384 v 1379.
# No. of SVs for gaussian is more in cvxopt-based model compared to sklearn(rbf), 2147 v 1462.

# In[39]:


# Comparing w and b
w_skl = svm_linear_skl.coef_.T
b_skl = svm_linear_skl.intercept_

print("Implemented w:", svm_linear.w)
print("Sklearn w:", w_skl)
print("Norm diff = ", np.linalg.norm(svm_linear.w-w_skl))

print("Implemented b: ", svm_linear.b)
print("Sklearn b: ", b_skl)


# (d) fun if time

# In[183]:


X_fun, y_fun = generate_dataset(2,"train",(32,32))


# # Multiclass Image Classification

# (a) One-vs-one classifier

# In[97]:


class MultiSVM:
    def __init__(self, k_cls, kerfx=gaussianKernel, c=1):
        self.C = c
        self.ker = kerfx
        self.num_classes = k_cls

        num_classifiers = int(k_cls*(k_cls-1)/2)
        self.classifiers = [SVM(kerfx,c) for i in range(num_classifiers)]
        self.class_pairs = [(-1,-1)]*num_classifiers

    def fit(self, X, y):
        k = 0
        for i in range(self.num_classes):
            for j in range(i+1,self.num_classes):
                idx_this = np.logical_or(y == i, y == j).flatten()
                X_this = X[idx_this]
                y_this = y[idx_this]
                # print(X_this.shape)
                y_this = np.where(y_this == i,1,-1)
                # print(y)
                # print(y_this)

                self.classifiers[k].fit(X_this,y_this)
                self.class_pairs[k] = (i,j)

                print((i,j), "done")
                print()
                k += 1


    def wandb(self):
        self.w_list = [np.zeros(0)]*len(self.classifiers)
        self.b_list = [None]*len(self.classifiers)

        for i in range(len(self.classifiers)):
            self.w_list[i] = self.classifiers[i].w
            self.b_list[i] = self.classifiers[i].b        


    def predict(self, X_val):
        classes = np.arange(self.num_classes)
        votes = {k: 0 for k in range(self.num_classes)}

        m,n = X_val.shape
        
        list_y_preds = []
        for k in range(len(self.classifiers)):
            pos_class = self.class_pairs[k][0]
            neg_class = self.class_pairs[k][1]
            
            y_pred_k = self.classifiers[k].predict(X_val)
            y_pred_k = np.where(y_pred_k==1, pos_class, neg_class)

            list_y_preds.append(y_pred_k)
        
        stack_y_preds = np.hstack(list_y_preds)
        pred = np.full((m,1),-1)
        for i in range(m):
            vote, count = np.unique(stack_y_preds[i], return_counts=True)

            # reverse lists to get descending order
            # (in order of largest score)
            vote = np.flip(vote)
            count = np.flip(count)

            pred_vote = vote[count.argmax()]
            pred[i][0] = pred_vote
        
        return pred


# Some code for testing below:

# In[126]:


vote, count = np.unique([2,2,3,3,1], return_counts=True)
vote = np.flip(vote)
pred_vote = vote[count.argmax()]
pred_vote


# In[61]:


# l = svm_linear.predict(X_val)
# g = svm_gaussian.predict(X_val)
# lsk = svm_linear_skl.predict(X_val).reshape(-1,1)
# gsk = svm_gaussian_skl.predict(X_val).reshape(-1,1)

# out =  np.hstack([l,g,lsk,gsk])
# pred = np.zeros(400)
# for i in range(400):
#     vote, count = np.unique(out[i], return_counts=True)
#     pred_vote = vote[count.argmax()]
#     pred[i] = pred_vote

# pred = pred.reshape(-1,1)

# for p in [l,g,lsk,gsk,pred]:
#     print(np.mean(p == y_val))


# In[85]:


# modifive to label 0 as 0 (not -1)
y_modifive = np.where(y==1,1,0)
y_val_modifive = np.where(y_val==1,1,0)


# In[83]:


t0 = time.time()

svm_reichtest = MultiSVM(2)
svm_reichtest.fit(X,y_modifive)

print("Time taken by Multiclass SVM = ", time.time()-t0)


# In[92]:


y_pred_modifive = svm_reichtest.predict(X_val)
np.mean(y_pred_modifive == y_val_modifive)


# Generating full-size data:

# In[93]:


X_full, y_full = generate_dataset(6)


# In[103]:


X_val_full, y_val_full = generate_dataset(6,dir="val")


# In[98]:


t0 = time.time()

svm_reich = MultiSVM(6)
svm_reich.fit(X_full,y_full)

print("Time taken by Multiclass SVM = ", time.time()-t0)


# In[107]:


y_predv_full = svm_reich.predict(X_val_full)

accu_val_full = np.mean(y_predv_full == y_val_full)
print("Multiclass SVM overall accuracy = ", accu_val_full)


# (b) scikit-learn Multiclass SVM

# In[111]:


t0 = time.time()

svm_reich_skl = svm.SVC(decision_function_shape='ovo',gamma=0.001)
svm_reich_skl.fit(X_full,y_full.flatten())

print("Time taken by Multiclass Scikit-learn SVM = ", time.time()-t0)


# In[127]:


y_predv_skl_full = svm_reich_skl.predict(X_val_full)

accu_val_skl = np.mean(y_predv_skl_full == y_val_full.T)
print("Test(multiclass) accuracy = ", accu_val_skl)


# In[184]:


svm_reich_skl.score(X_val_full,y_val_full)


# Similar to the case with binary Gaussian kernel classifier, for Multiclass scikit SVM:
# - Accuracy is significantly higher, 65.58% v 55.42%
# - And training time is lesser, 69.45s v 1060.90s

# (c) Confusion Matrix for Multiclassifier

# In[129]:


from sklearn.metrics import confusion_matrix as cm
import seaborn as sns


# In[130]:


cm_cvxreich = cm(y_val_full,y_predv_full)
cm_sklreich = cm(y_val_full,y_predv_skl_full)


# In[132]:


sns.heatmap(cm_cvxreich, annot=True, fmt='g', xticklabels=np.arange(6), yticklabels=np.arange(6), cmap='viridis')
plt.ylabel('MultiSVM Predicted Labels')
plt.xlabel('Actual Labels')


# In[133]:


sns.heatmap(cm_sklreich, annot=True, fmt='g', xticklabels=np.arange(6), yticklabels=np.arange(6), cmap='viridis')
plt.ylabel('ScikitSVM Predicted Labels')
plt.xlabel('Actual Labels')


# which into which missclassify? 12 ex of wrongs. comment
# 
# For ex:
# - We see a lot of 4 is classified into: 2,3 by SKL and 0,2 by MultiSVM (4 and 2 might be similar labels)
# - A lot of 5 is predicted as 0 by both models

# In[176]:


idx_miscls = np.arange(len(y_val_full))[(y_val_full.flatten() != y_predv_skl_full)]
idx_miscls = np.random.choice(idx_miscls,12)

x_miscls = X_val_full[idx_miscls]
x_miscls.shape


# In[177]:


for i in range(12):
    miss = (x_miscls[i]*255).reshape(16,16,3)
    # print(sv6.shape)
    miss = miss.astype(np.uint8)
    img_sv = Image.fromarray(miss)
    img_sv.save(f"./q2nc/mis/q2nc_mislabel{i}.png")


# (d) K-fold CV accuracy

# In[186]:


from sklearn.model_selection import KFold


# In[ ]:


# Accuracies on X, y (2 labels only) =>

# [0.48613445378151265,
#  0.49495798319327733,
#  0.8804621848739496,
#  0.8888655462184873,
#  0.8932773109243698,
#  0.9039915966386554]


# In[193]:


C_list = [1e-5,1e-3,1,5,10,100]
kf = KFold(n_splits=5, shuffle=True)

X_kf = X_full
y_kf = y_full.flatten()

t0 = time.time()

accu_kfold = []
for C in C_list:
    svm_schies = svm.SVC(C=C,gamma=0.001)
    
    scores = []
    for i, (train_idx, test_idx) in enumerate(kf.split(X_kf)):
        svm_schies.fit(X_kf[train_idx], y_kf[train_idx])
        score = svm_schies.score(X_kf[test_idx],y_kf[test_idx])
        scores.append(score)

        print(f"Fold {i+1} done for C={C}, at time {time.time()-t0}")
    
    scores = np.array(scores)
    accu_kfold.append(np.mean(scores))

print("Total time to generate 5-fold cv scores for C values on complete dataset = ", time.time()-t0)
accu_kfold


# In[208]:


fig, ax = plt.subplots()
ax.set_xscale('log')
ax.plot(C_list,accu_kfold,label='Kfold CV')

ax.set_title("Accuracy v/s C (kfold CV and validation)")
ax.set_xlabel("C (hyperparameter)")
ax.set_ylabel("Accuracy")


# In[199]:


t0 = time.time()

accu_valk = []
for C in C_list:
    svm_schies = svm.SVC(C=C,gamma=0.001)
    
    svm_schies.fit(X_full,y_full.flatten())
    score = svm_schies.score(X_val_full, y_val_full.flatten())

    print(f"{C} done")
    accu_valk.append(score)

print("Total time to generate val-scores for C values on complete dataset = ", time.time()-t0)


# In[209]:


ax.plot(C_list,accu_valk,label='Validation')
ax.legend()
ax.figure


# In[215]:


print("Accuracy for C=1:",accu_kfold[2],accu_valk[2])
print("Best accuracy (for C=10):",accu_kfold[-2],accu_valk[-2])


# We see that out of C = [1e-5,1e-2,1,5,10], we obtain the best accuracy for C=10 in case of both Kfold cross-validation and train-test validation.
# 
# PS: Upon going higher, we see it still increases for C=100 in both metrics.
