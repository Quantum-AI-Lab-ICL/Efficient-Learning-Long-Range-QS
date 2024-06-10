import numpy as np
import random as rnd
import math
import matplotlib.pyplot as plt
import time
import cmath

import scipy
from scipy import sparse as sps
from scipy.optimize import leastsq, curve_fit

from sklearn.linear_model import LassoCV
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.tools.fit import fit_with_sum_of_exp, sum_of_exp
from tenpy.networks.mpo import MPO
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import MPOModel, CouplingMPOModel, CouplingModel
from tenpy.models.lattice import Chain
from tenpy.algorithms import dmrg
from tenpy.tools.params import asConfig, get_parameter


n = 8 # number of qubits

# parameters of the Hamiltonian
h_max = math.e
exponent_alpha = 3
J_max = 2

# hyperparameters of the ML model
alphas = [2**(-8), 2**(-7), 2**(-6), 2**(-5)]
Rs = [5, 10, 20, 40]
gammas = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75]

XGate = [[0,1],[1,0]]
YGate = [[0,0-1j],[0+1j,0]]
ZGate = [[1,0],[0,-1]]


N_test = 40 # the number of testing samples
distance = 2 # the distance delta determining the size of neighbourhood of parameters used to approximate a given p-body observable


def SymmetricExpSum(x,*args):
    #calculates f(x) = sum_i a_i*(b_i^x + b_i^{n-x})
    # args = [a_0, b_0, a_1, b_1, ...]

    if len(args)%2 == 1:
        print("Error: wrong number of args")
    k = int(len(args)/2)
    
    result = 0
    for i in range(k):
        a = args[2*i]
        b = args[2*i+1]
        result += a*(b**x + b**(n-x))
        
    return result
    
    
def SymmetricFit(fit_range,n_exp):
    # fits 1/min{x,n-x}^\alpha with sum_i a_i*(b_i^x + b_i^{n-x}) according to the method in the Appendix of original paper

    def AdjustedFunction(xdata):
        f = np.zeros(fit_range)
        for j in range(fit_range):
            if xdata[j] <= n/2:
                f[j] = (1 / xdata[j])**exponent_alpha - 0.5 /((n-xdata[j])**exponent_alpha)
            else:
                f[j] = 0.5/(xdata[j]**exponent_alpha)
                
        return f
            
    lam, pref = fit_with_sum_of_exp(AdjustedFunction, n_exp, fit_range)

    results = [None]*(2*n_exp)
    results[::2] = pref
    results[1::2] = lam
    
    return results


class RandomIsingChainFromGrid(MPOModel):
    # sets up the Hamiltonian as an MPO

    def __init__(self, model_params):
        n = model_params.get('L',8)
        J = model_params.get('JParams',0.5+np.zeros(n))
        h = model_params.get('hParams',0.5+np.zeros(n))

        site = SpinHalfSite(conserve = None, sort_charge = None)
        lat = Chain(n, site, bc_MPS="finite", bc="open")
        
        Sigmax, Sigmaz, Id = site.Sigmax, site.Sigmaz, site.Id
        
        
        def decay(x):
            d = np.zeros(len(x))
            
            for j in range(len(x)):
                d[j] = (1 / min(x[j],n-x[j]))**exponent_alpha
  
            return d
        
        n_exp = 0
        
        tolerance = 1e-6


        fit_range = n+n%2
        MaxError = 1
        MaxErrorLast = 10
        MaxN = 30
        
        doublecheck = 1
        check = 0
        
        while MaxError > tolerance and n_exp < MaxN and doublecheck:
        
            n_exp += 1
                            
            xdata = np.arange(1,n)
            ydata = decay(xdata)

            popt = SymmetricFit(fit_range,n_exp)
            yfit = SymmetricExpSum(xdata,*popt)
          

            MaxErrorLast = MaxError
            MaxError = max(abs(ydata-yfit))
            
            if check:
                doublecheck = 0
            
            if MaxError>MaxErrorLast and n_exp>10:
                check = 1
                n_exp -= 2
            
            lam = []
            pref = []
                            
            k = int(len(popt)/2)
            
            for i in range(k):
                pref = np.append(pref, popt[2*i])
                pref = np.append(pref, popt[2*i]*(popt[2*i+1]**n))
                lam = np.append(lam,popt[2*i+1])
                lam = np.append(lam,1/(popt[2*i+1]))

                   
        n_exp = len(lam)
                                                                
        grids = []
        
        for i in range(n):
           
            grid = [[None for _ in range(2*n_exp+2)] for _ in range(2*n_exp+2)]
            grid[0][0] = 1*Id
            grid[0][2*n_exp+1] = h[i]*Sigmax
            grid[2*n_exp+1][2*n_exp+1] = 1*Id
           
            for j in range(n_exp):
                grid[0][j+1] = lam[j]*J[i]*Sigmaz
                grid[0][n_exp+j+1] = lam[j]*Sigmaz
                
                grid[j+1][j+1] = lam[j]*Id
                grid[n_exp+j+1][n_exp+j+1] = lam[j]*Id
                
                grid[j+1][2*n_exp+1] = pref[j]*J[i]*Sigmaz
                grid[n_exp+j+1][2*n_exp+1] = pref[j]*Sigmaz
           
            grids.append(grid)
                
        H = MPO.from_grids(lat.mps_sites(), grids, bc='finite', IdL=0, IdR=-1)
        MPOModel.__init__(self, lat, H)


    
def CalculatePropertiesDMRG(J,h):
    # evaluates the observable for the given parameters using DMRG

    C = np.zeros(n)
    
    model_params = dict(L=n, bc="open", bc_MPS="finite", JParams = J, hParams = h)

    Ising = RandomIsingChainFromGrid(model_params)
        
    product_state = []
    for i in range(Ising.lat.N_sites):
        product_state.append(rnd.choice(["up", "down"]))
    psi = MPS.from_product_state(Ising.lat.mps_sites(), product_state, bc = Ising.lat.bc_MPS)

    dmrg_params = {
        'mixer': False,  # setting this to True helps to escape local minima
        'max_E_err': 1.e-6,
        'trunc_params': {
            #'chi_max': 10,
            'svd_min': 1.e-9
        },
        'combine': True,
        'chi_list': {0: 5, 5: 10, 15: 20, 25: 50, 90:100},
        'max_sweeps': 100
    }
    
    Ising.test_sanity()
        
    info = dmrg.run(psi, Ising, dmrg_params)
    
    
    X = psi.expectation_value('Sigmax')
    
    Z0ZiLists = psi.correlation_function('Sigmaz','Sigmaz')
        
    return info['E'], X, Z0ZiLists


def InitiateChain():
    # sets the random parameters in the Hamiltonians

    J = J_max * np.random.rand(n)
    h = h_max * np.random.rand(n)
    
    return J, h
        
    
def CalculateProperties(J,h):
    # evaluates the observable for the given parameters using exact diagonalisation

    H = sps.csr_matrix((2**n,2**n))
    
    def dist(i,j):
        d = min(abs(i-j), n-abs(i-j))
        return d
    
    for i in range(n):
    
        XiGate = sps.kron(sps.kron(sps.eye(2**i),XGate),sps.eye(2**(n-i-1)))
        ZiGate = sps.kron(sps.kron(sps.eye(2**i),ZGate),sps.eye(2**(n-i-1)))

        H += h[i] * XiGate
    
        for j in range(n):
        
            if i < j:
                                    
                ZjGate = sps.kron(sps.kron(sps.eye(2**j),ZGate),sps.eye(2**(n-j-1)))
                
                H += (1+J[i]*J[j]) / (dist(i,j)**exponent_alpha) * ZiGate * ZjGate
                                
    eval, evec = sps.linalg.eigsh(H,k=1,which="SA")
 
                
    X = np.zeros(n)
    
    Z0ZiLists = np.zeros((n,n))

    for i in range(n):

        ZiGate = sps.kron(sps.kron(sps.eye(2**i),ZGate),sps.eye(2**(n-i-1)))
        
        for j in range(n):
        
            ZjGate = sps.kron(sps.kron(sps.eye(2**j),ZGate),sps.eye(2**(n-j-1)))
            
            Gates = ZiGate*ZjGate

            Z0ZiLists[i][j] = np.real((np.conj(np.transpose(evec)).dot(Gates.dot(evec))).item())
        
        XiGate = sps.kron(sps.kron(sps.eye(2**i),XGate),sps.eye(2**(n-i-1)))

        X[i] = np.real((np.conj(np.transpose(evec)).dot(XiGate.dot(evec))).item())

    return eval.item(), X, Z0ZiLists
    
    
def ZOfQubit(Control,J,h,a,distance):
    # separates the parameters in the given neighbourhood
    
    if Control == -1:
    
        MaxZLength = (2*distance+1)+(2*distance+1)*distance
        Z = np.zeros(MaxZLength)

    
        for i in range(distance+1):
            Z[distance+i]=h[(a+i)%n]
            Z[distance-i]=h[(a-i)%n]
                        
        for i in range(2*distance+1):
        
            for j in range(2*distance+1):
            
                if j>i:
                                    
                    VParameter = 1+J[(i+a-distance)%n]*J[(j+a-distance)%n]
                    Z[2*distance+j+i*2*distance-i-int(i*(i-1)/2)] = VParameter
                    
            
    else:
        
        MaxZLength = 2*((2*distance+1)+(2*distance+1)*distance)
        Z = np.zeros(MaxZLength)
        
        for i in range(distance+1):
            Z[distance+i]=h[(a+i)%n]
            Z[distance-i]=h[(a-i)%n]
                        
        for i in range(2*distance+1):
        
            for j in range(2*distance+1):
            
                if j>i:
                                    
                    VParameter = 1+J[(i+a-distance)%n]*J[(j+a-distance)%n]
                    Z[2*distance+j+i*2*distance-i-int(i*(i-1)/2)] = VParameter
                    
                    
        for i in range(distance+1):
            Z[(2*distance+1)+(2*distance+1)*distance+distance+i]=h[(Control+a+i)%n]
            Z[(2*distance+1)+(2*distance+1)*distance+distance-i]=h[(Control+a-i)%n]
                        
        for i in range(2*distance+1):
        
            for j in range(2*distance+1):
            
                if j>i:
                                    
                    VParameter = 1+J[(Control+i+a-distance)%n]*J[(Control+j+a-distance)%n]
                    Z[(2*distance+1)+(2*distance+1)*distance+2*distance+j+i*2*distance-i-int(i*(i-1)/2)] = VParameter
                    
    return Z
    
def PhiOfZ(Control,R,gamma,Z,omegas):
    #maps to randomised Fourier features

    if Control == -1:
    
        MaxZLength = (2*distance+1)+(2*distance+1)*distance

    else:
    
        MaxZLength = 2*((2*distance+1)+(2*distance+1)*distance)


    l = MaxZLength
            
    phi = np.zeros(2*R)
        
    for s in range(R):
        
        prod = np.dot(Z,omegas[s])
                                    
        phi[2*s] = math.cos(gamma/math.sqrt(l) * prod)
        phi[2*s + 1] = math.sin(gamma/math.sqrt(l) * prod)

    return phi


def CapitalPhi(Control,R,gamma,J,h,omegas):

    CPhi = []

    Z = ZOfQubit(Control,J,h,0,distance)
    CPhi = np.append(CPhi,PhiOfZ(Control,R,gamma,Z,omegas))
            
    return CPhi
    
    
class FeatureMappedLasso(BaseEstimator, ClassifierMixin):
    # this applies the feature mapping and then uses cross-validated Lasso model

    def __init__(self, alphas, Omegas, Control, R=20, Gamma=0.5):
        
        self.alphas = alphas
        self.Omegas = Omegas
        self.R = R
        self.Gamma = Gamma
        self.Control = Control

    def fit(self, Js, Cs):
    
        Js, Cs = check_X_y(Js,Cs)
            
        labels = LabelEncoder()
        labels.fit(Cs)
            
        self.classes_ = labels.classes_
    
        self.model = LassoCV(alphas = self.alphas)

        R = self.R
        gamma = self.Gamma
        omegas = self.Omegas
        Control = self.Control
        
        N_train = len(Cs)
        
        results = Cs.copy()
                    
        # need to have at least 5 training samples for CV, LassoCV, etc; so just duplicate them if less than 5
        if N_train == 1:
            results = np.append(results,[results,results,results,results])
        if N_train == 2:
            results = np.append(results,[results,results])
        if N_train == 3 or N_train == 4:
            results = np.append(results,results)
            
            
        PhiMatrix = np.zeros((N_train,2*R))

        for repeats in range(N_train):

            JappH = Js[repeats].copy()
            J = JappH[:len(JappH)//2].copy()
            h = JappH[len(JappH)//2:].copy()

            PhiMatrix[repeats] = CapitalPhi(Control,R,gamma,J,h,omegas)
                    
        # need to have at least 5 training samples for CV, LassoCV, etc; so just duplicate them if less than 5
        if N_train == 1:
            PhiMatrix = np.vstack((PhiMatrix,PhiMatrix,PhiMatrix,PhiMatrix,PhiMatrix))
        if N_train == 2:
            PhiMatrix = np.vstack((PhiMatrix,PhiMatrix,PhiMatrix))
        if N_train == 3 or N_train == 4:
            PhiMatrix = np.vstack((PhiMatrix,PhiMatrix))
        
        self.model.fit(PhiMatrix,results)
        return self

    def predict(self, J):

        # Check if fit has been called
        check_is_fitted(self.model)
        
        J = np.asarray(J)
        
        R = self.R
        gamma = self.Gamma
        omegas = self.Omegas
        Control = self.Control

        
        if np.shape(J) != (2*n,):
        
            a,b = np.shape(J)
                        
            predictions = np.zeros(a)
                    
            for i in range(a):
            
                JSingle = J[i].copy()
                JOne = JSingle[:len(JSingle)//2].copy()
                hOne = JSingle[len(JSingle)//2:].copy()
        
                predictions[i] = self.model.predict(CapitalPhi(Control,R,gamma,JOne,hOne,omegas).reshape(1,-1)).item()
            
        
        else:

            JOne = J[:len(J)//2].copy()
            hOne = J[len(J)//2:].copy()

            predictions = self.model.predict(CapitalPhi(Control,R,gamma,JOne,hOne,omegas).reshape(1,-1))
       
        
        
        return predictions
    
    def set_params(self, R, Gamma):
    
        self.R = R
        self.Gamma = Gamma
        
        return self
        
    def get_params(self, deep=True):
    
        return {'alphas': self.alphas, 'Omegas': self.Omegas, 'R': self.R, 'Gamma': self.Gamma, 'Control': self.Control}


@ignore_warnings(category=ConvergenceWarning) # ignoring convergence warning
def main():

    JTests = []
    CTests = []
    XTests = []
    ZTests = []
    
    # set up testing samples
    for i in range(N_test):

        print("Setting up test",i+1,"/",N_test)

        J, h = InitiateChain()
        JappH = np.append(J,h)
        JTests.append(JappH)
        
        if n <= 16:
            C, X, Z = CalculateProperties(J,h)
        else:
            C, X, Z = CalculatePropertiesDMRG(J,h)
            
        CTests.append(C)
        XTests.append(X)
        ZTests.append(Z)
    
    # shows the range and standard deviation of observables
    print("\nValues of energy in the testing data:")
    rangeTests = np.asarray(CTests).max()-np.asarray(CTests).min()
    print("Range is ",rangeTests)
    print("Standard deviation is ",np.std(CTests))

        

    print("Setting up train 1/1")

    JSingle, hSingle = InitiateChain()
            
    if n <= 16:
        EnergyTrivial, Xs, ZZs = CalculateProperties(JSingle,hSingle)
    else:
        EnergyTrivial, Xs, ZZs = CalculatePropertiesDMRG(JSingle,hSingle)

    TPs = []

    for i in range(n):

        JappH = np.append(np.roll(JSingle,-i),np.roll(hSingle,-i))
        TPs.append(JappH)
            
            
            
    ############################################# Xs predictor

    MaxZLength = (2*distance+1)+(2*distance+1)*distance

    omegas = np.random.normal(0,1,(max(Rs),MaxZLength))

    param_grid = dict(R = Rs, Gamma = gammas)

    model = FeatureMappedLasso(alphas, omegas, -1)

    # need to have at least 5 training samples for CV, LassoCV, etc; so just duplicate them if less than 5
    if n == 1:
        XsElongated = np.append(Xs,[Xs,Xs,Xs,Xs])
        TPsElongated = np.vstack((TPs,TPs,TPs,TPs,TPs))
    elif n == 2:
        XsElongated = np.append(Xs,[Xs,Xs])
        TPsElongated = np.vstack((TPs,TPs,TPs))
    elif n == 3 or n == 4:
        XsElongated = np.append(Xs,Xs)
        TPsElongated = np.vstack((TPs,TPs))
    else:
        TPsElongated = TPs.copy()
        XsElongated = Xs.copy()
                
    XPredictor = GridSearchCV(model,param_grid,cv = 5, scoring = 'neg_root_mean_squared_error', return_train_score = False)
    XPredictor.fit(np.asarray(TPsElongated),np.asarray(XsElongated))

    variance = 0
    mean = np.asarray(XsElongated).mean()
    k = len(XsElongated)
    for j in range(len(XsElongated)):
        variance += (XsElongated[j]-mean)**2
    score = XPredictor.score(np.asarray(TPsElongated),np.asarray(XsElongated))
    print("Coeff of determination for X's is: ",1-k*(score**2)/variance)

    ############################################# ZZs predictor

    NofDistances = int((n-n%2)/2)

    MaxZLength = 2*((2*distance+1)+(2*distance+1)*distance)

    omegas = [np.random.normal(0,1,(max(Rs),MaxZLength)) for _ in range(NofDistances)]

    param_grid = dict(R = Rs, Gamma = gammas)

    ZZModels = [FeatureMappedLasso(alphas, omegas[control], control+1) for control in range(NofDistances)]

                
    ZZPredictors = [GridSearchCV(ZZModels[i],param_grid,cv = 5, scoring = 'neg_root_mean_squared_error', return_train_score = False) for i in range(NofDistances)]

    for d in range(NofDistances):

        ZResults = []
        TPs = []

        for i in range(n):
        
            j = (i+d+1)%n
            result = ZZs[i][j]
            
            JappH = np.append(np.roll(JSingle,-i),np.roll(hSingle,-i))

            ZResults.append(result)
            TPs.append(JappH)

    
        # need to have at least 5 training samples for CV, LassoCV, etc; so just duplicate them if less than 5
        if n == 1:
            ZZsElongated = np.append(ZResults,[ZResults,ZResults,ZResults,ZResults])
            TPsElongated = np.vstack((TPs,TPs,TPs,TPs,TPs))
        elif n == 2:
            ZZsElongated = np.append(ZResults,[ZResults,ZResults])
            TPsElongated = np.vstack((TPs,TPs,TPs))
        elif n == 3 or n == 4:
            ZZsElongated = np.append(ZResults,ZResults)
            TPsElongated = np.vstack((TPs,TPs))
        else:
            ZZsElongated = ZResults.copy()
            TPsElongated = TPs.copy()

        ZZPredictors[d].fit(np.asarray(TPsElongated),np.asarray(ZZsElongated))


    error = 0

    for repeats in range(N_test):

        J = np.asarray(JTests[repeats].copy())
        JOne = J[:len(J)//2].copy()
        hOne = J[len(J)//2:].copy()
        result = CTests[repeats]
            
        sum = 0

        for i in range(n):

            JappH = np.append(np.roll(JOne,-i),np.roll(hOne,-i))
                  
            sum += hOne[i]*XPredictor.predict(JappH).item()
                        
            for j in range(n):
            
                if(i<j):
                
                    d = min(abs(i-j),n-abs(i-j))
                    
                    if(abs(i-j) <= n-abs(i-j)):
                
                        JappH = np.append(np.roll(JOne,-i),np.roll(hOne,-i))
                        
                        sum += (1+JOne[i]*JOne[j]) * ((1 / min(abs(i-j),n-abs(i-j)))**exponent_alpha) * ZZPredictors[d-1].predict(JappH).item()
                        
                    if(abs(i-j) > n-abs(i-j)):
                
                        JappH = np.append(np.roll(JOne,-j),np.roll(hOne,-j))
                        
                        sum += (1+JOne[i]*JOne[j]) * ((1 / min(abs(i-j),n-abs(i-j)))**exponent_alpha) * ZZPredictors[d-1].predict(JappH).item()
                    
                  
                                            
        prediction = sum
        
        #print("Result is",result," and prediction is",prediction)
                        
        error += 1/N_test *((result - prediction)**2)
        
                


    print("\nN_train is ",1," and the error in energy prediction is ",math.sqrt(error))

    EnergyErrorTrivial = 0


    for repeats in range(N_test):

        EnergyErrorTrivial += 1/N_test *((CTests[repeats] - EnergyTrivial)**2)


    print(", while the trivially obtained error is",math.sqrt(EnergyErrorTrivial))

    print("\n")

    return 0


main()
