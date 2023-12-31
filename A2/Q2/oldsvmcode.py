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

        # try eye??
        P = matrix((y@y.T)*K, tc='d')
        # P = matrix((y@y.T)*K + self.C*np.eye(m), tc='d')
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

        sol = solvers.qp(P,q,G,h,A,b)
        # alpha = np.array(sol['x']).flatten()
        
        # # finding support vectors
        # sv = (alpha>1e-6)

        alphas = np.array(sol['x'])
        idx = (alphas > 1e-6).flatten()
        self.sv_idx = idx.nonzero()[0]
        print(f"Got {len(self.sv_idx)} support vectors")
        self.sv_x = X[idx]
        self.sv_y = y[idx]
        self.sv_alpha = alphas[idx]

        if self.ker == linearKernel:
            self.w = self.sv_x.T @ (self.sv_alpha * self.sv_y)
        
        self.b = self.sv_y - np.sum(self.ker(self.sv_x,self.sv_x) * self.sv_alpha * self.sv_y, axis=0)
        self.b = np.mean(self.b)
        
        print(self.sv_alpha.shape)
        print(self.w)
        print(self.b)

        # self.b = self.sv_y
        # self.b -= np.sum(self.ker(self.sv_x,self.sv_x) * self.sv_alpha * self.sv_y, axis=0, dtype=int)
        # self.b = np.mean(self.b)

        # # finding intercept b
        # self.b = 0
        # for k in range(len(self.sv_alpha)):
        #     self.b += self.sv_y[k]
        #     self.b -= np.sum(self.sv_alpha*self.sv_y*K[idx[k],sv])
        # self.b /= len(self.sv_alpha)

        # # finding weights
        # if self.ker == linearKernel:
        #     self.w = np.zeros(n)
        #     for k in range(len(self.sv_alpha)):
        #         self.w += self.sv_alpha[k]*self.sv_y[k]*self.sv_x[k]
        # else:
        #     self.w = None

    def predict(self,X_val):
        project = np.sum(self.ker(X_val,self.sv_x)*self.sv_alpha*self.sv_y, axis=0)
        project += self.b

        return np.sign(project)