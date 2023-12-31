#!/usr/bin/env python
# coding: utf-8

# # Q4. Gaussian Discriminant Analysis

# In[15]:


import numpy as np
import matplotlib.pyplot as plt


# (a) GDA with same Covariance

# In[16]:


def std_normalize(X: np.ndarray):
    X_mu = X.mean(axis=0)
    X_std = X.std(axis=0)
    return (X-X_mu)/X_std

def sigmoid(X: np.ndarray):
    return 1/(1+np.exp(-X))


# In[17]:


def sigmaParticiple(X: np.ndarray, Y: np.ndarray, type=-1):
    X0 = X[Y==0]
    X1 = X[Y==1]
    m0 = X0.mean(axis=0)
    m1 = X1.mean(axis=0)

    if type == 0:
        return ((X0-m0).T@(X0-m0))/len(X0)
    elif type == 1:
        return ((X1-m1).T@(X1-m1))/len(X1)
    else:
        return ((X0-m0).T@(X0-m0) + (X1-m1).T@(X1-m1))/len(Y)


# In[18]:


X_train = np.loadtxt("../data/q4/q4x.dat")
Y_train = np.loadtxt("../data/q4/q4y.dat",dtype=object)     # couldn't convert Alaska to float?


# In[19]:


X = std_normalize(X_train)
Y = np.where(Y_train == "Canada",1,0)


# In[20]:


X0 = X[Y==0]
X1 = X[Y==1]

n = len(X)
n0 = len(X0)
n1 = len(X1)

mu0 = X0.mean(axis=0)
mu1 = X1.mean(axis=0)
phi = n0/n

print("Mean(mu0) of 'Alaska' labels = ", mu0)
print("Mean(mu1) of 'Canada' labels = ", mu1)
print("phi = ", phi)


# In[21]:


sigma_common = sigmaParticiple(X,Y)
print("Common Covariance:")
print(sigma_common)


# (b) Plot of Training Data

# In[22]:


x0 = X.T[0]
x1 = X.T[1]

fig, ax = plt.subplots()

s = plt.scatter(x0,x1,c=Y,cmap="bwr")
plt.title('Salmon Distribution according to Ring Diameters')
plt.xlabel('x0 (normalized)')
plt.ylabel('x1 (normalized)')

plt.legend(handles=s.legend_elements()[0], labels=['Alaska','Canada'])


# (c) Plot of GDA Linear Boundary

# In[25]:


# We find the separation boundary at points where P1 = P2

def plotty(X: np.ndarray, x0_params: list, x1_params: list, phi):
    p0_term = (X-x0_params[0]).T@x0_params[2]@(X-x0_params[0])
    p1_term = (X-x1_params[0]).T@x1_params[2]@(X-x1_params[0])
    log_term = np.log(x0_params[1]) - np.log(x1_params[1]) + 2*np.log(phi/(1-phi))
    return p0_term - p1_term + log_term

plotty_vec = np.vectorize(plotty, signature="(2),(3),(3),()->()")

def GDA_plot_decision(x0_params, x1_params, phi, ax, color='k'):
    x0_space = np.linspace(-3,2,400)
    x1_space = np.linspace(-2,3,400)
    plot_mesh = np.stack(np.meshgrid(x0_space, x1_space),axis=2)

    prob_mesh = plotty_vec(plot_mesh,x0_params,x1_params,phi)
    quad_curve = plot_mesh[(prob_mesh<0.01)&(prob_mesh>-0.01)].T

    ax.plot(quad_curve[0], quad_curve[1], color=color)


# In[26]:


det_sc = np.linalg.det(sigma_common)
inv_sc = np.linalg.inv(sigma_common)

x0_params = [mu0, det_sc, inv_sc]
x1_params = [mu1, det_sc, inv_sc]

GDA_plot_decision(x0_params, x1_params, phi, ax, color='g')
ax.figure


# (d) Parameter Estimates for GDA

# In[27]:


print("Mean(mu0) of 'Alaska' labels = ", mu0)
print("Mean(mu1) of 'Canada' labels = ", mu1)


# In[28]:


sigma0 = sigmaParticiple(X,Y,0)
sigma1 = sigmaParticiple(X,Y,1)

print("Covariance matrix(Sigma0) of 'Alaska' labelled datapoints: ")
print(sigma0)

print()

print("Covariance matrix(Sigma1) of 'Canada' labelled datapoints: ")
print(sigma1)


# (e) Quadratic Boundary

# In[29]:


det_sigma0 = np.linalg.det(sigma0)
det_sigma1 = np.linalg.det(sigma1)

inv_sigma0 = np.linalg.inv(sigma0)
inv_sigma1 = np.linalg.inv(sigma1)


# In[30]:


x0_params = [mu0, det_sigma0, inv_sigma0]
x1_params = [mu1, det_sigma1, inv_sigma1]

GDA_plot_decision(x0_params,x1_params,phi,ax)
ax.figure


# (f) Comparison between Linear and Quadratic Boundary

# The linear boundary generated from a common Covariance matrix performs fairly well, having 7-8 misclassifications.
# 
# Whereas, the quadratic boundary generated from GDA with separate Sigma_i for the labels 'Alaska' and 'Canada', performs marginally better with fewer misclassifications. But, it also requires much more computational time, which may not be justified by performance gains.
#  
# The quadratic boundary generated is hyperbolic, as can be seen from the plot above.
