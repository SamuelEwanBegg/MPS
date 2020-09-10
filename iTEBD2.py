#iTEBD2 
#Author: Samuel Begg, contractions based on code of Frank Pollman and original paper of Vidal
#Date: 08/09/2020
#2nd order trotter [O(dt^3) accuracy] + adaptive bond dimension size + schmidt cut-off

import MPS_methods as mps
import numpy as np
import copy
from joblib import Parallel, delayed
import multiprocessing as mp
import numpy.random as rand
import scipy.linalg as lin
import scipy.io as spio
import matplotlib.pyplot as plt

##################################################
#Control Panel
cycle_times = 5000 #cycle_times*delta = physical simulation time
times = 3*cycle_times #number of steps
chiM = 15 #max bond dimension
d = 2; #physical dimension
delta =0.01; #time-step 

#Hamiltonian Parameters
J = 1.0
Jx = 0.5
Jy = 1.3
hx = 0.0

plotting = 1 #generate plots = 1, no plots = 0
###########################################################################
#Initialize MPS

G  =[]
G =G +  [np.asarray(np.zeros((d,1,1),dtype = complex))]
G =G +  [np.asarray(np.zeros((d,1,1),dtype = complex))]

#Default all-up state 
G[0][0,0,0]=1.0
G[0][1,0,0]=0.0
G[1][0,0,0]=1.0
G[1][1,0,0]=0.0
l = []
#Product State
l = l + [[1.0]]
l = l + [[1.0]]
#Bond Dimension
chi = 1

#Can also initialize random tensor
#G = [np.random.rand(d,chi,chi)] + [np.random.rand(d,chi,chi)] 
#l = [np.random.rand(chi)] + [np.random.rand(chi)] 

#######################################################
#Create 2-gate unitary

#Initialise Pauli Matrices
Sx = 0.5*np.asarray([[0,1.0],[1.0,                                              0]])
Sy = 0.5*np.asarray([[0,-1.0j],[1.0j,                                           0]])
Sz = 0.5*np.asarray([[1.0,0],[0,-1.0]])

# Create 2-site Hamiltonian
H = -J*np.kron(Sz,Sz) - Jx*np.kron(Sx,Sx) - Jy*np.kron(Sy,Sy)- 0.5*hx*np.kron(np.identity(2),Sx) - 0.5*hx*np.kron(Sx,np.identity(2))

#Uf-even gates, Uh-odd gates (f for full time-step, h for half time-step in 2nd order trotter)
Uf = np.reshape(lin.expm(-1j*H*delta) ,(2,2,2,2))
Uh = np.reshape(lin.expm(-1j*H*delta/2) ,(2,2,2,2))
######################################################
#Alternative: Can diagonalize at this stage.
#w,v = np.linalg.eig(H)
#Uf = np.reshape(np.dot(np.dot(v,np.diag(np.exp(-1j*delta*(w)))),np.transpose(v)),(2,2,2,2))
#Uh = np.reshape(np.dot(np.dot(v,np.diag(np.exp(-1j*(delta/2.0)*(w)))),np.transpose(v)),(2,2,2,2))
######################################################
#Initialise Observables and Diagnostic Variables 

#Observables
norm = np.zeros(cycle_times)
mag = np.zeros(cycle_times,dtype = complex)
obs = np.zeros(cycle_times,dtype = complex)
obs2 = np.zeros(cycle_times)
obs3 = np.zeros(cycle_times)
entang = np.zeros(cycle_times)
entang2 = np.zeros(cycle_times)
#Diagnostics
chiA_store = np.zeros(cycle_times)
largest = np.zeros(cycle_times)
mini = np.zeros(cycle_times)
sums = np.zeros(cycle_times)

#########################################################
#Perform time evolution using 2nd order trotter algorithm

kk = 0 #keeps track of physical time, whereas step is number of gate layers
# Perform the time evolution alternating on A and B bonds (ABAABAAB..i.e. gates UhUfUhUhUfUh..) 

for step in range(0,times):
    #2nd order trotter 
    if np.mod(step,3) == 1:
        A = 1
        B = 0
        U = copy.deepcopy(Uf)
    else:
        A = 0
        B = 1
        U = copy.deepcopy(Uh)

    chiA = np.size(l[A][:])
    chiB = np.size(l[B][:])
    
    # Construct theta
    theta = np.tensordot(np.diag(l[B][:]),G[A][:,:,:],axes=(1,1)) 
    theta = np.tensordot(theta,np.diag(l[A][:],0),axes=(2,0))
    theta = np.tensordot(theta,G[B][:,:,:],axes=(2,1))
    theta = np.tensordot(theta,np.diag(l[B][:],0),axes=(3,0))
    # Apply imaginary-time evolution operator U
    theta = np.tensordot(theta,U,axes=([1,2],[0,1]));
    # Perform singular-value decomposition
    theta = np.reshape(np.transpose(theta,(2,0,3,1)),(d*chiB,d*chiB));
    X, Y, Z = np.linalg.svd(theta); Z = Z.T   #U,S,Vh. Z transposed for convenience in manip.
    # Truncate the bond dimension back to chi and normalize the state
    
    chiA = np.size(Y) 
    l[A] = Y/np.sqrt(sum(Y**2))
    
    if chiA > chiM: #truncate bond dimension 
        chiA = chiM
        l[A] = np.zeros(chiA)
        l[A]= Y[0:chiA]/np.sqrt(sum(Y[0:chiA]**2))
        print('bond dim: max size truncate',chiA)
  
    aa = 0
    for ff in range(0,chiA): #remove small schmidt values
        if aa == 0:
            if (l[A][ff] < 10**(-15)):
                chiA = ff
                l[A]= l[A][0:chiA]/np.sqrt(sum(l[A][0:chiA]**2))
                aa = 1 
                print('bond dim: small elements truncate',chiA)


    X=np.reshape(X[0:d*chiB,0:chiA],(d,chiB,chiA))
    G[A]= np.zeros((np.shape(X))) 
    G[B]= np.zeros((np.shape(Z))) 
    G[A]=np.transpose(np.tensordot(lin.inv(np.diag(l[B][:])),X,axes=(1,1)),(1,0,2));
    Z=np.transpose(np.reshape(Z[0:d*chiB,0:chiA],(d,chiB,chiA)),(0,2,1)) #transpose operations are for treatment of Z like X, so have to transpose back.
    G[B] =np.tensordot(Z,lin.inv(np.diag(l[B][:])),axes=(2,0));

    A_mat = []
    A_mat= A_mat + [G[A]]
    A_mat= A_mat + [G[B]]
    B_mat = []
    B_mat= B_mat + [X]
    B_mat= B_mat + [Z]


    if np.mod(step,3)==2: #After succession of gates we make a `measurement'
        print(step,kk)        

        
        if np.mod(kk,2)==0:
            chiA_store[kk] = chiA 
        else: 
            chiA_store[kk] = chiA_store[kk-1]

        norm[kk] = mps.Normalisation_Infinite(B_mat,l,A)
        mag[kk] = mps.Mag_Infinite(B_mat,l,A)
        obs[kk] = mps.Obs_Infinite(B_mat,l,A,Sz,Sz)  
        obs2[kk] = mps.Obs_Infinite(B_mat,l,A,Sx,Sx)  
        obs3[kk] = mps.Obs_Infinite(B_mat,l,A,np.identity(2),Sz)
        entang[kk] = -np.sum((np.square(l[A]))*np.log(np.square(l[A])))
        entang2[kk] = -np.sum((np.square(l[B]))*np.log(np.square(l[B])))
        #entang[step] = -np.log(np.sum(theta**2))/delta/2
        largest[kk] = np.max(l[A])    
        mini[kk] = np.min(l[A])    
        sums[kk] = np.sum(np.square(l[A]))    
        kk = kk + 1


#Plot Results
if plotting == 1:
    plt.plot(delta*np.arange(1,np.size(mag)+1),norm,'o',label = 'N')
    plt.plot(delta*np.arange(1,np.size(mag)+1),obs,'o',label = 'ZZ')
    plt.plot(delta*np.arange(1,np.size(mag)+1),obs2,'om',label = 'XX')
    plt.plot(delta*np.arange(1,np.size(mag)+1),mag,'o',label = 'M')
    plt.plot(delta*np.arange(1,np.size(mag)+1),entang,'o',label = 'S')
    plt.plot(delta*np.arange(1,np.size(mag)+1),entang2,'o',label = 'S')

    #ED
    ed = np.load('/home/samuel/Desktop/recentZ.npy')
    edC = np.load('/home/samuel/Desktop/recentcorr.npy')
    edx = np.load('/home/samuel/Desktop/recentcorrX.npy')
    ed_entropy = np.load('/home/samuel/Desktop/entropy.npy')
    ed_dt = 0.1
    plt.plot(ed_dt*np.arange(0,np.size(ed)),ed,'x',label='ED')
    plt.plot(ed_dt*np.arange(0,np.size(ed)),edC,'x',label = 'EDcorr')
    plt.plot(ed_dt*np.arange(0,np.size(ed)),edx,'x',label = 'EDcorrX')
    plt.plot(ed_dt*np.arange(0,np.size(ed)),ed_entropy,'x',label = 'EDentropy')

    #MPO-W1
    mpo_mag = np.genfromtxt('/home/samuel/ITensor-3/sample/mpoW_magz.txt')
    mpo_corr = np.genfromtxt('/home/samuel/ITensor-3/sample/mpoW_corrz.txt')
    mpo_entropy = np.genfromtxt('/home/samuel/ITensor-3/sample/mpoW_entropy.txt')
    mpo_obs = np.genfromtxt('/home/samuel/ITensor-3/sample/mpoW_obs.txt')
    dt_mpo = 0.01
    plt.plot(dt_mpo*np.arange(1,np.size(mpo_mag)+1),mpo_mag,label='MPO-mag')
    plt.plot(dt_mpo*np.arange(1,np.size(mpo_obs)+1),mpo_obs,label='MPO-obs')
    plt.plot(dt_mpo*np.arange(1,np.size(mpo_mag)+1),mpo_corr,label='MPO-corr')
    plt.plot(dt_mpo*np.arange(1,np.size(mpo_mag)+1),mpo_entropy,label='MPO-entropy')
    plt.xlim(0,delta*cycle_times)
    plt.ylim(-1,1)
    plt.legend()
    plt.show()

    #Plot Simulation Parameters
    plt.plot(delta*np.arange(1,np.size(mag)+1),chiA_store,label = 'bond_dimension')
    plt.plot(delta*np.arange(1,np.size(mag)+1),sums,label= 'sums')
    plt.plot(delta*np.arange(1,np.size(mag)+1),largest,label= 'largest')
    plt.plot(delta*np.arange(1,np.size(mag)+1),mini,label= 'small')
    plt.show()

