import numpy as np
import X,y,X_val,y_val,linearKernel from q2

class SVM:
    def __init__(self, kerfx = linearKernel, c=1):
        self.ker = kerfx
        self.C = c


    def fit(self, X: np.ndarray, Y:np.ndarray):
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
        # ind = np.arange(len(alpha))[sv]

        self.sv_alpha = alpha[sv]
        self.sv_x = X[sv]
        self.sv_y = y[sv]
        print(f"{len(self.sv_alpha)} support vectors out of {m} samples")


    def wandb(self):
        self.sv_alpha = self.sv_alpha.reshape(-1,1)

        # finding bias
        ker_sv = self.ker(self.sv_x,self.sv_x)
        b_sum = self.sv_y - np.sum(ker_sv * self.sv_alpha * self.sv_y, axis=0)
        self.b = np.mean(b_sum)

        # finding weights
        if self.ker == linearKernel:
            self.w = self.sv_x.T @ (self.sv_alpha*self.sv_y)
        else:
            self.w = None

        print("Weights:\n", self.w)
        print("Bias = ", self.b)


    def predict(self,X_val):
        if self.w is not None:
            project = X_val@self.w + self.b
        
        else:
            ker_sv_val = self.ker(self.sv_x,X_val)
            print(ker_sv_val.shape)
            y_p = ker_sv_val @ self.sv_alpha*self.sv_y
            # y_p = np.zeros(len(X_val))
            # for i in range(len(X_val)):
            #     s = 0
            #     for alp, sv_y, sv_x in zip(self.alpha, self.sv_y, self.sv_x):
            #         s += alp*sv_y*self.ker(X_val[i],sv_x)
                
            #     y_p[i] = s
                
            project = y_p + self.b

        return np.sign(project)


def gaussianKernel(x,z,gamma=0.001):
    m1 = x.shape[0]
    m2 = z.shape[0]
    
    x_norm_vec = np.sum(x**2,axis=1).reshape(-1,1)
    z_norm_vec = np.sum(z**2,axis=1).reshape(-1,1)
    print(x_norm_vec.shape)
    
    x_prod = x_norm_vec@np.ones((1,m2))
    print(x_prod.shape)
    z_prod = z_norm_vec@np.ones(m1).T
    
    return np.exp(-gamma*(x_prod + z_prod.T - 2*x@z.T))

def create_gaussian_kernel(gamma=0.001):
    def kernel(x, z):
        xx = (x**2).sum(axis=1).reshape(-1,1)@np.ones((1,z.shape[0]))
        zz = (z**2).sum(axis=1).reshape(-1,1)@np.ones((1,x.shape[0]))
        return np.exp(-gamma*(xx + zz.T - 2*x@z.T))
    
    return kernel


svm_gaussian = SVM(create_gaussian_kernel())
svm_gaussian.fit(X,y)

svm_gaussian.wandb()

y_predv = svm_gaussian.predict(X_val)
yss = np.sign(X_val@svm_gaussian.w + svm_gaussian.b)
print(y_predv.shape,yss.shape)

accu_val = np.mean(y_predv == y_val)
print(accu_val)