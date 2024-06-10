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
from tenpy.models.model import MPOModel, CouplingMPOModel, CouplingModel, NearestNeighborModel
from tenpy.models.lattice import Chain
from tenpy.algorithms import dmrg
from tenpy.tools.params import asConfig, get_parameter


n = 8 # number of qubits

# parameters of the Hamiltonian
J_max = 2

# hyperparameters of the ML model
alphas = [2**(-8), 2**(-7), 2**(-6), 2**(-5)]
Rs = [5, 10, 20, 40]
gammas = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75]

XGate = [[0,1],[1,0]]
YGate = [[0,0-1j],[0+1j,0]]
ZGate = [[1,0],[0,-1]]

N_test = 10 # the number of testing samples
N_train = 1 # the number of training samples; this is supposed to be 1 for the paper, but you can use more
distance = 4 # the distance delta determining the size of neighbourhood of parameters used to approximate a given p-body observable

MaxZLength = 2*distance+1


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
    

class HeisenbergChainFromGrid(MPOModel):
    # sets up the Hamiltonian as an MPO

    def __init__(self, model_params):
        n = model_params.get('L',8)
        J = model_params.get('JParams',0.5+np.zeros(n))
        
        site = SpinHalfSite(conserve = None, sort_charge = None)
        lat = Chain(n, site, bc_MPS="finite", bc="open")
        
        Sigmax, Sigmay, Sigmaz, Id = site.Sigmax, site.Sigmay, site.Sigmaz, site.Id
        
        
        size = 8
                            
        grids = []
        
        for i in range(n):
        
            k = 0
            if i ==  0:
                k = J[n-1]
            if i == n-1:
                k = 1
           
            grid = [[None for _ in range(size)] for _ in range(size)]
            
            grid[0][0] = 1*Id
            grid[size-1][size-1] = 1*Id
            grid[0][size-1] = 0*Id
           
            grid[1][1] = 1*Id
            grid[2][2] = 1*Id
            grid[3][3] = 1*Id
            


            grid[0][1] = k*Sigmax
            grid[0][2] = k*Sigmay
            grid[0][3] = k*Sigmaz
            
            grid[1][size-1] = k*Sigmax
            grid[2][size-1] = k*Sigmay
            grid[3][size-1] = k*Sigmaz
            
            grid[0][4] = J[i]*Sigmax
            grid[0][5] = J[i]*Sigmay
            grid[0][6] = J[i]*Sigmaz

            grid[4][size-1] = 1*Sigmax
            grid[5][size-1] = 1*Sigmay
            grid[6][size-1] = 1*Sigmaz
            
            grids.append(grid)
                            
        H = MPO.from_grids(lat.mps_sites(), grids, bc='finite', IdL=0, IdR=-1)
        MPOModel.__init__(self, lat, H)
        
    
def CalculatePropertiesDMRG(J):
    # evaluates the observable for the given parameters using DMRG

    model_params = dict(L=n, JParams = J)

    Heisenberg = HeisenbergChainFromGrid(model_params)
        
    # sets up initial MPS as a random product state
    product_state = []
    for i in range(Heisenberg.lat.N_sites):
        product_state.append(rnd.choice(["up", "down"]))
    psi = MPS.from_product_state(Heisenberg.lat.mps_sites(), product_state, bc = Heisenberg.lat.bc_MPS)

    dmrg_params = {
        'mixer': True,  # setting this to True helps to escape local minima
        'max_E_err': 1.e-6,
        'trunc_params': {
            #'chi_max': 10,
            'svd_min': 1.e-9
        },
        'combine': True,
        'chi_list': {0: 5, 5: 10, 15: 20, 25: 50, 90:100},
        'max_sweeps': 100
    }
    
    Heisenberg.test_sanity()
        
    info = dmrg.run(psi, Heisenberg, dmrg_params)
        
    ZZ = npc.outer(psi.sites[0].Sigmaz.replace_labels(['p', 'p*'], ['p0', 'p0*']), psi.sites[1].Sigmaz.replace_labels(['p', 'p*'], ['p1', 'p1*']))
        
    YY = npc.outer(psi.sites[0].Sigmay.replace_labels(['p', 'p*'], ['p0', 'p0*']), psi.sites[1].Sigmay.replace_labels(['p', 'p*'], ['p1', 'p1*']))
        
    XX = npc.outer(psi.sites[0].Sigmax.replace_labels(['p', 'p*'], ['p0', 'p0*']), psi.sites[1].Sigmax.replace_labels(['p', 'p*'], ['p1', 'p1*']))
        
    Gates = XX + YY + ZZ
    
    C = 1/3 *psi.expectation_value(Gates)
        
    last = 1/3 * (psi.expectation_value_term([('Sigmax',0),('Sigmax',n-1)])+psi.expectation_value_term([('Sigmay',0),('Sigmay',n-1)])+psi.expectation_value_term([('Sigmaz',0),('Sigmaz',n-1)]))
            
    return np.append(C,last), info['E']


def InitiateChain():
    # sets the random parameters in the Hamiltonians
    #J = (J_01, J_12, ... , J_{n-2,n-1}, J_{n-1,0})
    
    J = J_max * np.random.rand(n)
    
    if n == 2:
        J[1] = 0
       
    return J
    
def CalculateProperties(J):
    # evaluates the observable for the given parameters using exact diagonalisation

    H = sps.csr_matrix((2**n,2**n))
    
    for i in range(n):
    
        j = (i+1)%n
    
        XiGate = sps.kron(sps.kron(sps.eye(2**i),XGate),sps.eye(2**(n-i-1)))
        YiGate = sps.kron(sps.kron(sps.eye(2**i),YGate),sps.eye(2**(n-i-1)))
        ZiGate = sps.kron(sps.kron(sps.eye(2**i),ZGate),sps.eye(2**(n-i-1)))
                
        XjGate = sps.kron(sps.kron(sps.eye(2**j),XGate),sps.eye(2**(n-j-1)))
        YjGate = sps.kron(sps.kron(sps.eye(2**j),YGate),sps.eye(2**(n-j-1)))
        ZjGate = sps.kron(sps.kron(sps.eye(2**j),ZGate),sps.eye(2**(n-j-1)))

        Gates = np.real((XiGate*XjGate + YiGate*YjGate + ZiGate*ZjGate))

        H += J[i] * Gates
        
    eval, evec = sps.linalg.eigsh(H,k=1,which="SA")
            
 
                
    C = np.zeros(n)
    #C = (C_01, C_12, ... , C_{n-2,n-1}, C_{n-1,0})

    for i in range(n):
    
        j = (i+1)%n

        XiGate = sps.kron(sps.kron(sps.eye(2**i),XGate),sps.eye(2**(n-i-1)))
        YiGate = sps.kron(sps.kron(sps.eye(2**i),YGate),sps.eye(2**(n-i-1)))
        ZiGate = sps.kron(sps.kron(sps.eye(2**i),ZGate),sps.eye(2**(n-i-1)))
                
        XjGate = sps.kron(sps.kron(sps.eye(2**j),XGate),sps.eye(2**(n-j-1)))
        YjGate = sps.kron(sps.kron(sps.eye(2**j),YGate),sps.eye(2**(n-j-1)))
        ZjGate = sps.kron(sps.kron(sps.eye(2**j),ZGate),sps.eye(2**(n-j-1)))

        Gates = 1/3 * (XiGate*XjGate + YiGate*YjGate + ZiGate*ZjGate)
            
        C[i] = np.real((np.conj(np.transpose(evec)).dot(Gates.dot(evec))).item())

    return C, eval.item()


def ZOfQubit(J,a,distance):
    # separates the parameters in the given neighbourhood
    
    Z = np.zeros(MaxZLength)
    
    
    for i in range(distance+1):
        Z[distance+i]=J[(a+i)%n]
        Z[distance-i]=J[(a-i)%n]

                
    return Z
    
def PhiOfZ(R,gamma,Z,omegas):
    #maps to randomised Fourier features

    l = MaxZLength
            
    phi = np.zeros(2*R)
        
    for s in range(R):
        
        prod = np.dot(Z,omegas[s])
                                    
        phi[2*s] = math.cos(gamma/math.sqrt(l) * prod)
        phi[2*s + 1] = math.sin(gamma/math.sqrt(l) * prod)

    return phi


def CapitalPhi(R,gamma,J,omegas):

    CPhi = []

    Z = ZOfQubit(J,0,distance)
    CPhi = np.append(CPhi,PhiOfZ(R,gamma,Z,omegas))
            
    return CPhi
    
    
class FeatureMappedLasso(BaseEstimator, ClassifierMixin):
    # this applies the feature mapping and then uses cross-validated Lasso model


    def __init__(self, alphas, Omegas, R=20, Gamma=0.5):
        
        self.alphas = alphas
        self.Omegas = Omegas
        self.R = R
        self.Gamma = Gamma

    def fit(self, Js, Cs):
    
        Js, Cs = check_X_y(Js,Cs)
            
        labels = LabelEncoder()
        labels.fit(Cs)
            
        self.classes_ = labels.classes_
    
        self.model = LassoCV(alphas = self.alphas)

        R = self.R
        gamma = self.Gamma
        omegas = self.Omegas
        
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

            J = Js[repeats].copy()

            PhiMatrix[repeats] = CapitalPhi(R,gamma,J,omegas)
            
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
        
        
        if np.shape(J) != (n,):
        
            a,b = np.shape(J)
                        
            predictions = np.zeros(a)
                    
            for i in range(a):
            
                JSingle = J[i].copy()
        
                predictions[i] = self.model.predict(CapitalPhi(R,gamma,JSingle,omegas).reshape(1,-1)).item()
            
        
        else:

            predictions = self.model.predict(CapitalPhi(R,gamma,J,omegas).reshape(1,-1))
       
        return predictions
    
    def set_params(self, R, Gamma):
    
        self.R = R
        self.Gamma = Gamma
        
        return self
        
    def get_params(self, deep=True):
    
        return {'alphas': self.alphas, 'Omegas': self.Omegas, 'R': self.R, 'Gamma': self.Gamma}


@ignore_warnings(category=ConvergenceWarning) # ignoring convergence warning
def main():


    JTests = []
    CTests = []
    EnergyTests = []

    # set up testing samples
    for i in range(N_test):

        print("Setting up test",i+1,"/",N_test)

        J = InitiateChain()
        JTests.append(J)
        
        if n <= 16:
            C, energy = CalculateProperties(J)
        else:
            C, energy = CalculatePropertiesDMRG(J)
  
        CTests.append(C)
        EnergyTests.append(energy)


    # shows the range and standard deviation of observables
    print("\nValues of C_{ij}'s in the testing data:")
    print("min = ",np.asarray(CTests).min(),", max = ",np.asarray(CTests).max())
    rangeTests = np.asarray(CTests).max()-np.asarray(CTests).min()
    print("Range is ",rangeTests)
    print("Standard deviation is ",np.std(CTests),"\n")

    Js = []
    Cs = []
    EnergyTrivial = 0
        
    # set up training samples
    for i in range(N_train):

        print("Setting up train",i+1,"/",N_train)
        J = InitiateChain()
        
        if n <= 16:
            C, energy = CalculateProperties(J)
        else:
            C, energy = CalculatePropertiesDMRG(J)
            
        EnergyTrivial = energy

        for j in range(n):

            Js.append(np.roll(J,-j))
            Cs.append(C[j])
        

    omegas = np.random.normal(0,1,(max(Rs),MaxZLength))

    param_grid = dict(R = Rs, Gamma = gammas)

    model = FeatureMappedLasso(alphas, omegas)

    
    # need to have at least 5 training samples for CV, LassoCV, etc; so just duplicate them if less than 5
    if n == 1:
        CsElongated = np.append(Cs,[Cs,Cs,Cs,Cs])
        JsElongated = np.vstack((Js,Js,Js,Js,Js))
    elif n == 2:
        CsElongated = np.append(Cs,[Cs,Cs])
        JsElongated = np.vstack((Js,Js,Js))
    elif n == 3 or n == 4:
        CsElongated = np.append(Cs,Cs)
        JsElongated = np.vstack((Js,Js))
    else:
        JsElongated = Js.copy()
        CsElongated = Cs.copy()
                
    # train the model using 5-fold cross-validation
    grid = GridSearchCV(model,param_grid,cv = 5, scoring = 'neg_root_mean_squared_error', return_train_score = False)
    grid.fit(np.asarray(JsElongated),np.asarray(CsElongated))


    variance = 0
    mean = np.asarray(CsElongated).mean()
    k = len(CsElongated)
    for j in range(len(CsElongated)):
        variance += (CsElongated[j]-mean)**2
    score = grid.score(np.asarray(JsElongated),np.asarray(CsElongated))
    print("\nCoeff of determination is: ",1-k*(score**2)/variance)


    TotalError = 0
    EnergyPredictions = np.zeros(N_test)

    for i in range(n):
        
        error = 0
        
        for repeats in range(N_test):

            J = np.asarray(JTests[repeats].copy())
            result = CTests[repeats][(i)%n]
                                                
            prediction = grid.predict(np.roll(J,-i).copy()).item()
            
            EnergyPredictions[repeats] += 3*J[i]*prediction
            
            #print("Result is",result," and prediction is",prediction)
                            
            error += 1/N_test *((result - prediction)**2)
            
        TotalError += 1/n * math.sqrt(error)
            

    #print(grid.best_params_,"Alpha = ",grid.best_estimator_.model.alpha_)
                
    del(grid)

    print("N_train was ",N_train,"and the average RMS error is ",TotalError,"\n")

    EnergyErrorPredict = 0
    EnergyErrorTrivial = 0

    print("Energy range is ",np.asarray(EnergyTests).max()-np.asarray(EnergyTests).min())
    print("Energy standard deviation is ",np.std(EnergyTests))


    for repeats in range(N_test):

        EnergyErrorPredict += 1/N_test *((EnergyTests[repeats] - EnergyPredictions[repeats])**2)
        EnergyErrorTrivial += 1/N_test *((EnergyTests[repeats] - EnergyTrivial)**2)


    print("Energy prediction RMS error is ",math.sqrt(EnergyErrorPredict))
    
    if N_train  == 1:
        
        print(", while the trivially obtained error is",math.sqrt(EnergyErrorTrivial))
        
    print("\n")

    return 0


main()
