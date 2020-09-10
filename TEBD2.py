#TEBD2 
#Author: Samuel Begg 
#Date: 10/09/2020
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

#Initialise Pauli Matrices
Sx = 0.5*np.asarray([[0,1.0],[1.0,                                              0]])
Sy = 0.5*np.asarray([[0,-1.0j],[1.0j,                                           0]])
Sz = 0.5*np.asarray([[1.0,0],[0,-1.0]])

##################################################################
#Control Panel

cycle_times = 1000 #cycle_times*delta = physical simulation time
times = 3*cycle_times #number of steps
chiM = 15 #max bond dimension
d = 2; #physical dimension
delta = 0.01; #time-step 
N = 100 #system size
cut = int(N/2)-1 #measure entanglement, schimdt values
cut_off =  10**(-15) #Discard singular values smaller than this.

#OBC Nearest-Neighbour Hamiltonian Parameters
J = 1.0
Jx = 0.5
Jy = 1.3
hx = 0.0

plotting = 1 #Would you like to observe results? Yes:1, No:0
##################################################################
#Initialize MPS
G  =[] #MPS
l = [] #List of bond elements 
chi = []  #List of bond-dimensions size

#Default all-up state 
for ii in range(0,N):
    G = G +  [np.asarray(np.zeros((d,1,1),dtype = complex))]
    l = l + [[1.0]]
    G[ii][0,0,0]=1.0
    G[ii][1,0,0]=0.0
    chi = chi + [np.size(l[ii])]
    #Can also initialize random tensor
    #G = G + [np.random.rand(d,2,2)] 
    #l = l + [np.random.rand(2)] 

##################################################################
#Create 2-site unitary gate 
H = -J*np.kron(Sz,Sz) - Jx*np.kron(Sx,Sx) - Jy*np.kron(Sy,Sy)- 0.5*hx*np.kron(np.identity(2),Sx) - 0.5*hx*np.kron(Sx,np.identity(2))
#non-interacting parts split between odd and even gates

Uf = np.reshape(lin.expm(-1j*H*delta) ,(2,2,2,2))
Uh = np.reshape(lin.expm(-1j*H*delta/2) ,(2,2,2,2))

##################################################################
#Alternative: 
#w,v = np.linalg.eig(H)
#Uf = np.reshape(np.dot(np.dot(v,np.diag(np.exp(-1j*delta*(w)))),np.transpose(v)),(2,2,2,2))
#Uh = np.reshape(np.dot(np.dot(v,np.diag(np.exp(-1j*(delta/2.0)*(w)))),np.transpose(v)),(2,2,2,2))

##################################################################
#Initialize Observables and  Diagnostic Parameters

norm = np.zeros(cycle_times)
mag = np.zeros(cycle_times,dtype = complex)
obs = np.zeros(cycle_times,dtype = complex)
obs2 = np.zeros(cycle_times)
obs3 = np.zeros(cycle_times)
entang = np.zeros(cycle_times)
largest = np.zeros(cycle_times)
mini = np.zeros(cycle_times)
sums = np.zeros(cycle_times)

##################################################################
#Time Evolution
#2nd order trotter algorithm

kk = 0 #variable updating the physical time, +1 after each succession of gates (every 3rd step)
for step in range(0,times):
    print(step)
    if np.mod(step,3) == 1: #Even Gates

        for ii in range(0,N-1):

            theta = np.tensordot(G[ii],G[ii+1],axes=((2),(1))) #theta dims. d D1 d D3

            if ii%2 == 1:
                U = copy.deepcopy(Uf)
                
                #Sum over the physical dims with U
                theta = np.tensordot(U,theta,axes=((0,1),(0,2))); #apply U to two-site tensor.
                #new dims d d D1 D3 

                #Reshape to d D1 d D3
                theta = np.transpose(theta,(0,2,1,3)) 

            #Move to d*D1 and d*D3
            theta = np.reshape(theta,(d*chi[(ii-1)%N],d*chi[ii+1])); #combine bond indices and site indices to create matrix (dxD1,dxD3) 
            X, Y, Z = np.linalg.svd(theta,full_matrices = False)  #U,S,Vh. dims (d x D1 , chi) (chi,chi) and (chi, dxD3)

            chi[ii] = np.size(Y)
            l[ii] =  Y/np.sqrt(sum(Y**2))  
            G[ii] = np.reshape(X,(d,chi[(ii-1)%N],chi[ii]))       
            Z = np.tensordot(np.diag(l[ii]),Z,axes=(1,0)) #Absorb schmidt into Z
            Z = np.reshape(Z,(chi[ii],d,chi[ii+1]))
            Z = np.transpose(Z,(1,0,2)) #maps to (d,chi_i,chi_i+1)
            G[ii+1] = Z  

            
            if chi[ii] > chiM: #truncate bond dimension 
                chi[ii] = chiM
                l[ii]= Y[0:chi[ii]]/np.sqrt(sum(Y[0:chi[ii]]**2))
                print('bond dim: max size truncate',chi[ii])
            aa = 0
            for ff in range(0,chi[ii]): #remove small schmidt values
                if aa == 0:
                    if (l[ii][ff] < cut_off):
                        chi[ii] = ff
                        l[ii]= l[ii][0:chi[ii]]/np.sqrt(sum(l[ii][0:chi[ii]]**2))
                        aa = 1 
                        print('bond dim: small elements truncate',chi[ii])
        
            G[ii]= G[ii][:,:,0:chi[ii]]
            G[ii+1]= G[ii+1][:,0:chi[ii],:]
            
            del X,Y,Z 
             
    else: #Odd Gates
        
        
        for ii in range(0,N-1):
            
            theta = np.tensordot(G[ii],G[ii+1],axes=(2,1)) #theta dims. d D1 d D3
                 
            if ii%2 == 0:
                U = copy.deepcopy(Uh)
        
                #Sum over the physical dims with U
                theta = np.tensordot(U,theta,axes=((0,1),(0,2))); #apply U to two-site tensor.
                #new dims d d D1 D3


                #Reshape to d D1 d D3
                theta = np.transpose(theta,(0,2,1,3))

            #Move to d*D1 and d*D3
            theta = np.reshape(theta,(d*chi[(ii-1)%N],d*chi[ii+1])); #combine bond indices and site indices to create matrix (dxD1,dxD3)
           
            X, Y, Z = np.linalg.svd(theta,full_matrices = False);   #U,S,Vh. dims (d x D1 , chi) (chi,chi) and (chi, dxD3)
           
            chi[ii] = np.size(Y)
            
            l[ii] =  Y/np.sqrt(sum(Y**2)) 

            G[ii] = np.reshape(X,(d,chi[(ii-1)%N],chi[ii]))       
            
            Z = np.tensordot(np.diag(l[ii]),Z,axes=(1,0)) #Absorb schmidt into Z

            Z = np.reshape(Z,(chi[ii],d,chi[ii+1]))

            Z = np.transpose(Z,(1,0,2)) #maps to (d,chi_i,D3)
            
            G[ii+1] = Z  
            
            if chi[ii] > chiM: #truncate bonddimension 
                chi[ii] = chiM
                l[ii]= Y[0:chi[ii]]/np.sqrt(sum(Y[0:chi[ii]]**2))
                print('bond dim: max size truncate',chi[ii])
          
            aa = 0
            for ff in range(0,chi[ii]): #remove small schmidt values
                if aa == 0:
                    if (l[ii][ff] < cut_off):
                        chi[ii] = ff
                        l[ii]= l[ii][0:chi[ii]]/np.sqrt(sum(l[ii][0:chi[ii]]**2))
                        aa = 1 
                        print('bond dim: small elements truncate',chi[ii])
        
            G[ii]= G[ii][:,:,0:chi[ii]]
            G[ii+1]= G[ii+1][:,0:chi[ii],:]
          
            del X,Y,Z 

    #After each application performs the right-sweep of canonicalisation. Appears to be necessary to prevent issues with schmidt values on site 0, despite not needed for observables.
    G = mps.Inversion(G)
    G,schmidt,normer = mps.Canonicalise(G,chiM*2+1,1)
    G = mps.Inversion(G)


    if np.mod(step,3)==2: #After succession of 3-gates (1 time-step) evaluate observables 

        G = mps.Inversion(G)
        G,schmidt,normer = mps.Canonicalise(G,chiM*2+1,1)
        G = mps.Inversion(G)

        obs[kk] = mps.Observable_two_site_LEFTZIP(G,Sz,cut,Sz,cut+1)
        norm[kk] = mps.Normalisation_LEFTZIP(G)
        mag[kk] = mps.Magnetisation_LEFTZIP(G)
        obs2[kk] = mps.Observable_two_site_LEFTZIP(G,Sx,cut,Sx,cut+1)
        entang[kk] = -np.sum((np.square(l[cut]))*np.log(np.square(l[cut])))
        largest[kk] = np.max(l[cut])    
        mini[kk] = np.min(l[cut])    
        sums[kk] = np.sum(np.square(l[cut]))    
        kk = kk + 1


##################################################################
#Plot Analysis

if plotting == 1:
    #Plot Results
    plt.plot(delta*np.arange(1,np.size(mag)+1),norm,'o',label = 'N')
    #plt.plot(delta*np.arange(1,np.size(mag)+1),obs,'o',label = 'ZZ')
    plt.plot(delta*np.arange(1,np.size(mag)+1),mag,'o',label = 'M')
    plt.plot(delta*np.arange(1,np.size(mag)+1),obs,'o',label = 'ZZ')
    plt.plot(delta*np.arange(1,np.size(mag)+1),obs2,'om',label = 'XX')
    plt.plot(delta*np.arange(1,np.size(mag)+1),entang,'o',label = 'S')

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
    mpo_corr = np.genfromtxt('/home/samuel/ITensor-3/sample/mpoW_corrz_centre.txt')
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

    plt.plot(entang,label = 'entang')
    plt.plot(largest,label = 'largest')
    plt.plot(mini,label = 'smallest')
    plt.plot(sums,label = 'sums')
    plt.legend()
    plt.show()

