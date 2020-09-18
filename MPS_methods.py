import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
import numpy.random as rand
import scipy.linalg as lin
import scipy.io as spio
import copy

Sx = 0.5*np.asarray([[0,1.0],[1.0,                                              0]])
Sy = 0.5*np.asarray([[0,-1.0j],[1.0j,                                           0]])
Sz = 0.5*np.asarray([[1.0,0],[0,-1.0]])

#############################################################################################
#Methods for MPS
def random_state(physical,sites,bond): 
         #Generate a random matrix product state
        inputMPS = []
        d_dim = np.zeros(sites,dtype = 'I')
        for kk in range(0,sites):
                d_dim[kk] = bond	
        inputMPS.append(rand.normal(0,(1.0/float(sites)),[physical,1,d_dim[0]]))
        for kk in range(1,sites-1):
                inputMPS.append(rand.normal(0,(1.0/float(sites)),[physical,d_dim[kk-1],d_dim[kk]]))
        inputMPS.append(rand.normal(0,(1.0/float(sites)),[physical,d_dim[sites-1],1]))
        return inputMPS

##################################################################################
#Act on l matrices, unclear which class to join to

def entangle_entropy(bondmat):
    a = copy.deepcopy(bondmat)
    summer = 0
    cheke = 0
    for kk in range(0,np.size(bondmat)):
        cheke = cheke + a[kk]**2
        summer = summer  - a[kk]**2*(np.log(a[kk]**2))
    return summer

#Requires U,S,Vh and cut-off

def bond_element_cutoff(U,S,Vh,bond_cutoff):
    checker = 0
    for kk in range(0,np.size(S,0)):
        if checker == 0:
            if S[kk] < bond_cutoff:
                checker = 1
                eff_dim = kk
                S_new = S[0:eff_dim]
                U_new = U[:,0:eff_dim]		
                Vh_new = Vh[0:eff_dim,:]

    if checker == 0: #if no elements cutoff 
        U_new = U
        S_new = S
        Vh_new = Vh

    return U_new,S_new,Vh_new

#Requires U,S,Vh and bond-dim

def bond_truncation(bond_dim,U,S,Vh):

    if np.size(S) > bond_dim:
            S_new = S[0:bond_dim]
            U_new = U[:,0:bond_dim]		
            Vh_new = Vh[0:bond_dim,:]

    else: 

        U_new = U
        S_new = S
        Vh_new = Vh

    return U_new, S_new, Vh_new




def Left_Canonical_Check(MPS):
    
    outputMPS = np.dot(np.conj(np.transpose(MPS[0,:,:])),MPS[0,:,:]) + np.dot(np.conj(np.transpose(MPS[1,:,:])),MPS[1,:,:]) 

    outputMPS[np.abs(outputMPS)<10**(-15)] =0 
    return outputMPS

def Right_Canonical_Check(MPS):
    
    outputMPS = np.dot(MPS[0,:,:],np.conj(np.transpose(MPS[0,:,:]))) + np.dot(MPS[1,:,:],np.conj(np.transpose(MPS[1,:,:]))) 
    outputMPS[np.abs(outputMPS)<10**(-15)] =0 

    return outputMPS


def Inversion(inputMPS):
    
    sites = len(inputMPS)
    outputMPS = []
    
    for kk in range(0,sites):
        outputMPS = outputMPS + [np.transpose(inputMPS[sites-kk-1],(0,2,1))] 

    return outputMPS


def Canonicalise(inputMPS,bond_dim,normalise_output): #Left canonical
    
    sites = len(inputMPS)
    physical = np.size(inputMPS[0][:,0,0])
    #Canonicalises MPS in left canonical form, normalised but also outputs the norm 
    outputMPS = []
    schmidt_values = []

     
    for kk in range(0,sites-1): #all sites except last dealt with in loop
            if kk == 0:	
                    Ashaped = inputMPS[kk].reshape((np.size(inputMPS[kk],0)*np.size(inputMPS[kk],1),np.size(inputMPS[kk],2)))
            else:
                    del(Ashaped,U,S,Vh) 
                    Ashaped = tempMPS.reshape((np.size(tempMPS,0)*np.size(tempMPS,1),np.size(tempMPS,2)))
                    
            U,S,Vh = lin.svd(Ashaped,full_matrices = False)		

            if np.size(S) > bond_dim:
                    S_new = S[0:bond_dim]
                    U_new = U[:,0:bond_dim]		
                    Vh_new = Vh[0:bond_dim,:]
                    S = S_new
                    U = U_new
                    Vh = Vh_new
            

            schmidt_values.append(S)
            if kk == 0:
                    outputMPS.append(np.reshape(U,(np.size(inputMPS[kk],0),np.size(inputMPS[kk],1),np.size(S,0))))   #sigma,d1 and the new bond dimension
            else:
                    outputMPS.append(np.reshape(U,(np.size(tempMPS,0),np.size(tempMPS,1),np.size(S,0))))   #sigma,d1 and the new bond dimension
                    del(tempMPS)
            tempMPS = np.zeros([physical,np.size(S),np.size(inputMPS[(kk+1)%sites],2)],dtype = complex)

            tempMPS[0,:,:] = np.dot(np.dot(np.diag(S),Vh),inputMPS[(kk+1)%sites][0,:,:])  #MA becomes the new A
            tempMPS[1,:,:] = np.dot(np.dot(np.diag(S),Vh),inputMPS[(kk+1)%sites][1,:,:])  #MA becomes the new A

    #Last site
    del(Ashaped,U,S,Vh)
    Ashaped = tempMPS.reshape((np.size(tempMPS,0)*np.size(tempMPS,1),np.size(tempMPS,2)))
    U,S,Vh = lin.svd(Ashaped,full_matrices = False)
    outputMPS.append(np.reshape(U,(np.size(tempMPS,0),np.size(tempMPS,1),np.size(S,0))))   #sigma,d1 and the new bond dimension
    norm = S*Vh
    if normalise_output == 0:  #Does not normalise (multiplies normalised result by its norm)
            outputMPS[sites-1][:,:,0] = (norm)*outputMPS[sites-1][:,:,0]	
    #Note that the last site A = U S Vh was defined as U with S Vh ignored since they are just the norm 

    return [outputMPS,schmidt_values,norm]

def Canonicalise_pbc(inputMPS,bond_dim,normalise_output,cut_off): #Left canonical
    
    sites = len(inputMPS)
    physical = np.size(inputMPS[0][:,0,0])
    #Canonicalises MPS in left canonical form, normalised but also outputs the norm 
    outputMPS = []
    schmidt_values = []

     
    for kk in range(0,sites): #all sites except last dealt with in loop
            if kk == 0:	
                    Ashaped = inputMPS[kk].reshape((np.size(inputMPS[kk],0)*np.size(inputMPS[kk],1),np.size(inputMPS[kk],2)))
            else:
                    del(Ashaped,U,S,Vh) 
                    Ashaped = tempMPS.reshape((np.size(tempMPS,0)*np.size(tempMPS,1),np.size(tempMPS,2)))
                    
            U,S,Vh = lin.svd(Ashaped,full_matrices = False)		

            if np.size(S) > bond_dim:
                    S_new = S[0:bond_dim]
                    U_new = U[:,0:bond_dim]		
                    Vh_new = Vh[0:bond_dim,:]
                    S = S_new
                    U = U_new
                    Vh = Vh_new
            aa = 0 
            for ff in range(0,np.size(S)): #remove small schmidt values
            	if aa == 0:
                	if (S[ff] < cut_off):
                        	S = S[0:ff]/np.sqrt(sum(S[0:ff]**2))
                        	aa = 1 
			        #print('bond dim: small elements truncate',chi[ii])
                        	U_new = U[:,0:ff]		
                        	Vh_new = Vh[0:ff,:]
                        	U = U_new
                        	Vh = Vh_new

            schmidt_values.append(S)
            if kk == 0:
                    outputMPS.append(np.reshape(U,(np.size(inputMPS[kk],0),np.size(inputMPS[kk],1),np.size(S,0))))   #sigma,d1 and the new bond dimension
            else:
                    outputMPS.append(np.reshape(U,(np.size(tempMPS,0),np.size(tempMPS,1),np.size(S,0))))   #sigma,d1 and the new bond dimension
                    del(tempMPS)
            #print(sites,'HELLO')
            #print((kk+1)%sites,'printer')
            tempMPS = np.zeros([physical,np.size(S),np.size(inputMPS[(kk+1)%sites],2)],dtype = complex)
            tempMPS[0,:,:] = np.dot(np.dot(np.diag(S),Vh),inputMPS[(kk+1)%sites][0,:,:])  #MA becomes the new A
            tempMPS[1,:,:] = np.dot(np.dot(np.diag(S),Vh),inputMPS[(kk+1)%sites][1,:,:])  #MA becomes the new A
    
    #normalize S
    S =  S/np.sqrt(sum(S**2))  
    norm = np.dot(np.diag(S),Vh)
    #if np.size(norm) > 1:
    	#outputMPS[0] = np.transpose(np.tensordot(norm*outputMPS[0],axes = ((1),(1))),(1,0,2))	
    X = np.tensordot(norm,outputMPS[0],axes = ((1),(1)))	
    #Note that the last site A = U S Vh was defined as U with S Vh ignored since they are just the norm
   
    outputMPS[0]= np.transpose(X,(1,0,2))	

    return [outputMPS,schmidt_values,norm]

def Observable_one_site_LEFTZIP(MPS,Obs1,site1):  #Does not assume canonical form but zips from left.
    
    ii = 0
    if site1 == 0:
        left_site = Obs1[0,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + Obs1[1,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:]) + Obs1[0,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][0,:,:]) + Obs1[1,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][1,:,:])
    else:
        left_site = np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:])

    for jj in range(1,len(MPS)):

            if jj == site1:
                left_site =  Obs1[0,0]*np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][0,:,:])) +  Obs1[1,1]*np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][1,:,:])) +  Obs1[1,0]*np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][0,:,:])) + Obs1[0,1]*np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][1,:,:]))    
            
            else:
                new_left_up = np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][0,:,:]))
                new_left_down = np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][1,:,:])) #Bond contraction for down index

                left_site = new_left_up + new_left_down 

    obs = np.trace(left_site)
    return obs

def Observable_two_site_LEFTZIP_pbc(MPS,Obs1,site1,Obs2,site2):  #Does not assume canonical form but zips from left.
    
    ii = 0
    if site1 == 0:
        left_site = Obs1[0,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + Obs1[1,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:]) + Obs1[0,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][0,:,:]) + Obs1[1,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][1,:,:])
    else:
        left_site = np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:])

    for jj in range(1,len(MPS)):

            if jj == site1:
                left_site =  Obs1[0,0]*np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][0,:,:])) +  Obs1[1,1]*np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][1,:,:])) +  Obs1[1,0]*np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][0,:,:])) + Obs1[0,1]*np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][1,:,:]))    
            elif jj == site2:
                left_site =  Obs2[0,0]*np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][0,:,:])) +  Obs2[1,1]*np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][1,:,:])) +  Obs2[1,0]*np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][0,:,:])) + Obs2[0,1]*np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][1,:,:]))  
            else:
                new_left_up = np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][0,:,:]))
                new_left_down = np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][1,:,:])) #Bond contraction for down index

                left_site = new_left_up + new_left_down 
    obs = np.trace(left_site)
    return obs

def Observable_two_site_LEFTZIP(MPS,Obs1,site1,Obs2,site2):  #Does not assume canonical form but zips from left.
    
    ii = 0
    if site1 == 0:
        left_site = Obs1[0,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + Obs1[1,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:]) + Obs1[0,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][0,:,:]) + Obs1[1,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][1,:,:])
    else:
        left_site = np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:])

    for jj in range(1,len(MPS)):

            if jj == site1:
                left_site =  Obs1[0,0]*np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][0,:,:])) +  Obs1[1,1]*np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][1,:,:])) +  Obs1[1,0]*np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][0,:,:])) + Obs1[0,1]*np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][1,:,:]))    
            elif jj == site2:
                left_site =  Obs2[0,0]*np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][0,:,:])) +  Obs2[1,1]*np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][1,:,:])) +  Obs2[1,0]*np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][0,:,:])) + Obs2[0,1]*np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][1,:,:]))   
            
            else:
                new_left_up = np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][0,:,:]))
                new_left_down = np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][1,:,:])) #Bond contraction for down index

                left_site = new_left_up + new_left_down 

    obs = np.real(left_site[0,0])
    return obs




def Normalisation_LEFTZIP(MPS):  #Does not assume canonical form. Note that norm in canonical form is trivial
    ii = 0
    left_site = np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:])

    for jj in range(1,len(MPS)):
            new_left_up = np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][0,:,:]))
            new_left_down = np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][1,:,:])) #Bond contraction for down index
            left_site = new_left_up + new_left_down 
    norm = np.real(left_site[0,0])
    return norm


def Magnetisation_LEFTZIP(MPS):
    szloc = np.zeros(len(MPS))
    for kk in range(0,len(MPS)):
        szloc[kk] = Observable_one_site_LEFTZIP(MPS,Sz,kk)

    mag = np.mean(szloc)

    return mag

def Overlap(MPS): #must manually choose correct element to overlap with. Down state here.

    running = MPS[0][1,:,:] 
    for kk in range(1,len(MPS)):
    	running = np.dot(running,MPS[kk][1,:,:])	
    
    overlap = np.trace(running)	
    return overlap

################################################################################################

#Methods for iMPS
def Obs_Infinite(MPS,l,pos_A,Obs1,Obs2):  
    #Physical sum on each site separetely
    A = pos_A

    #left_spin  
    ii = 0
    left_site = Obs1[0,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + Obs1[1,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:])+Obs1[0,1]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][1,:,:])+Obs1[1,0]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][0,:,:]) 
    ii = 1
    right_site = Obs2[0,0]*np.dot(MPS[ii][0,:,:],np.conj(np.transpose(MPS[ii][0,:,:]))) + Obs2[1,1]*np.dot(MPS[ii][1,:,:],np.conj(np.transpose(MPS[ii][1,:,:]))) + Obs2[1,0]*np.dot(MPS[ii][0,:,:],np.conj(np.transpose(MPS[ii][1,:,:]))) + Obs2[0,1]*np.dot(MPS[ii][1,:,:],np.conj(np.transpose(MPS[ii][0,:,:])))
    #Sum over bond dimension then trace
    #TauL = np.dot(np.diag(l[A])**2,left_site)
    TauL = np.dot(np.dot(np.diag(l[A]),left_site),np.diag(l[A]))
    TauR = right_site
    mag1 = np.trace(np.dot(TauL,TauR))
    
    #right_spin 

    return mag1 



def Normalisation_Infinite(MPS,l,posA): 
    A = posA
    ii = 0
    left_site = np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:]) 
    ii = 1
    right_site = np.dot(MPS[ii][0,:,:],np.conj(np.transpose(MPS[ii][0,:,:]))) + np.dot(MPS[ii][1,:,:],np.conj(np.transpose(MPS[ii][1,:,:])))
    TauL = np.dot(np.dot(np.diag(l[A]),left_site),np.diag(l[A]))
    TauR = right_site
    norm = np.trace(np.dot(TauL,TauR))
    return norm 

def Mag_Infinite(MPS,l,pos_A):  
    #Physical sum on each site separetely
    A = pos_A

    #left_spin  
    ii = 0
    left_site = Sz[0,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + Sz[1,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:])+Sz[0,1]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][1,:,:])+Sz[1,0]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][0,:,:]) 
    ii = 1
    right_site = np.dot(MPS[ii][0,:,:],np.conj(np.transpose(MPS[ii][0,:,:]))) + np.dot(MPS[ii][1,:,:],np.conj(np.transpose(MPS[ii][1,:,:])))
    #Sum over bond dimension then trace
    TauL = np.dot(np.dot(np.diag(l[A]),left_site),np.diag(l[A]))
    TauR = right_site
    mag1 = np.trace(np.dot(TauL,TauR))
    
    #right_spin 
    ii = 0
    left_site = np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:]) 
    ii = 1
    right_site = Sz[0,0]*np.dot(MPS[ii][0,:,:],np.conj(np.transpose(MPS[ii][0,:,:]))) + Sz[1,1]*np.dot(MPS[ii][1,:,:],np.conj(np.transpose(MPS[ii][1,:,:]))) + Sz[1,0]*np.dot(MPS[ii][0,:,:],np.conj(np.transpose(MPS[ii][1,:,:]))) + Sz[0,1]*np.dot(MPS[ii][1,:,:],np.conj(np.transpose(MPS[ii][0,:,:])))
    #Sum over bond dimension then trace
    TauL = np.dot(np.dot(np.diag(l[A]),left_site),np.diag(l[A]))
    TauR = right_site
    mag2 = np.trace(np.dot(TauL,TauR))

    mag = 0.5*(mag1+mag2) 

    return mag 


###############################################################################################
#Methods for SDE_MPS

def Canonicalise_Parallel(inputMPS,bond_dim,normalise_output):#left canonical but not returning the same values
	
	
	sites = len(inputMPS)
	physical = np.size(inputMPS[0][:,0,0])
	#Canonicalises MPS in left canonical form, normalised but also outputs the norm 
	outputMPS = []
	schmidt_values = []

	
	for kk in range(0,sites-1): #all sites except last dealt with in loop
		if kk == 0:	
			Ashaped = inputMPS[kk].reshape((np.size(inputMPS[kk],0)*np.size(inputMPS[kk],1),np.size(inputMPS[kk],2)))
			#print('A',Ashaped)	
		else:
			del(Ashaped,U,S,Vh) 
			Ashaped = tempMPS.reshape((np.size(tempMPS,0)*np.size(tempMPS,1),np.size(tempMPS,2)))
			
		U,S,Vh = lin.svd(Ashaped,full_matrices = False)		

		if np.size(S) > bond_dim:
			S_new = S[0:bond_dim]
			U_new = U[:,0:bond_dim]		
			Vh_new = Vh[0:bond_dim,:]
			S = S_new
			U = U_new
			Vh = Vh_new
		

		schmidt_values.append(S)
		if kk == 0:
			outputMPS.append(np.reshape(U,(np.size(inputMPS[kk],0),np.size(inputMPS[kk],1),np.size(S,0))))   #sigma,d1 and the new bond dimension
			#print('output',outputMPS[0][1,:,:])
			#print('sch',S)
		else:
			outputMPS.append(np.reshape(U,(np.size(tempMPS,0),np.size(tempMPS,1),np.size(S,0))))   #sigma,d1 and the new bond dimension
			del(tempMPS)
		tempMPS = np.zeros([physical,np.size(S),np.size(inputMPS[(kk+1)%sites],2)],dtype = complex)

		tempMPS[0,:,:] = np.dot(np.dot(np.diag(S),Vh),inputMPS[(kk+1)%sites][0,:,:])  #MA becomes the new A
		tempMPS[1,:,:] = np.dot(np.dot(np.diag(S),Vh),inputMPS[(kk+1)%sites][1,:,:])  #MA becomes the new A

	#Last site
	del(Ashaped,U,S,Vh)
	Ashaped = tempMPS.reshape((np.size(tempMPS,0)*np.size(tempMPS,1),np.size(tempMPS,2)))
	U,S,Vh = lin.svd(Ashaped,full_matrices = False)
	outputMPS.append(np.reshape(U,(np.size(tempMPS,0),np.size(tempMPS,1),np.size(S,0))))   #sigma,d1 and the new bond dimension
	norm = S*Vh
	if normalise_output == 0:  #Does not normalise (multiplies normalised result by its norm)
		outputMPS[sites-1][:,:,0] = (norm)*outputMPS[sites-1][:,:,0]	
	#Note that the last site A = U S Vh was defined as U with S Vh ignored since they are just the norm 


	
	return outputMPS


def compressor(MPS_list,bond_dim,batches,output_norm):  #All samples are added together and then compressed

	for pp in range(0,batches):
		if pp == 0:
			MPS_batch1 = MPS_list[0]
		elif pp == 1:
			MPS_batch2 = MPS_list[1]
			MPS_running = MPS_Merge_Both_Normalise(MPS_batch1,MPS_batch2,batches)	
		else:
			#print(np.size(MPS_running[0]))	
			MPS_batch = MPS_list[pp]
			MPS_running = MPS_Merge_Single_Normalise(MPS_running,MPS_batch,batches)	
			del(MPS_batch)
	
	MPS_fin,s,n = Canonicalise(MPS_running,bond_dim,output_norm) 
	return MPS_fin,s,n


#Unverified
def Observable_one_site_LCF_LEFTZIP(MPS,Obs1,site1):  #Does not assume canonical form but zips from left.
    
    ii = 0
    if site1 == 0:
        left_site = Obs1[0,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + Obs1[1,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:]) + Obs1[0,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][0,:,:]) + Obs1[1,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][1,:,:])
    else: 
       left_site = np.identity(np.size(MPS[site1][0,:,:],0))
    
    for jj in range(1,len(MPS)):
            if jj == site1:
                left_site =  Obs1[0,0]*np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][0,:,:])) +  Obs1[1,1]*np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][1,:,:])) +  Obs1[1,0]*np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][0,:,:])) + Obs1[0,1]*np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][1,:,:])) 
            
            if jj > site1:
                new_left_up = np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][0,:,:]))
                new_left_down = np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][1,:,:])) #Bond contraction for down index

                left_site = new_left_up + new_left_down 

    obs = np.real(left_site[0,0])
    return obs



def MPS_Merge_Both_Normalise(MPS_1,MPS_2,batches):  #MPS_1 and MPS2 are divided by the batches, assumes MPS_1 properly normalised
	MPS_N = []
	sites = len(MPS_1) #Assuming MPS_2 is the same
	phys = 2 #np.size(MPS_1[0][:,0,0]) # The physical dimension is the first element
	for tt in range(0,sites):
		if tt == 0:
			new_mat = np.zeros([2,1,np.size(MPS_1[tt],2) + np.size(MPS_2[tt],2)],dtype = complex)
		elif tt == sites -1:
			new_mat = np.zeros([2,np.size(MPS_1[tt],1) + np.size(MPS_2[tt],1),1],dtype = complex)	
		else:
			new_mat = np.zeros([2,np.size(MPS_1[tt],1) + np.size(MPS_2[tt],1),np.size(MPS_1[tt],2) + np.size(MPS_2[tt],2)],dtype = complex)
			
		for phys in range(0,2):
			if tt == 0:
				new_mat[phys,0,0:np.size(MPS_1[tt],2)] = (1.0/float(batches))*MPS_1[tt][phys,0,:]  #MPS2 is added to MPS1 not the other way
				new_mat[phys,0,np.size(MPS_1[tt],2):np.size(new_mat,2)] = (1.0/float(batches))*MPS_2[tt][phys,0,:]
			
			elif tt == sites-1: #Note that this entry is considered 1D for some reason. Perhaps because new mat is dimension 2,X,1 and the 1 is on the end so is ignored?
				new_mat[phys,0:np.size(MPS_1[tt],1),0] = MPS_1[tt][phys,:,0]
				new_mat[phys,np.size(MPS_1[tt],1):np.size(new_mat,1),0] = MPS_2[tt][phys,:,0]

			else:
				new_mat[phys,0:np.size(MPS_1[tt],1),0:np.size(MPS_1[tt],2)] = MPS_1[tt][phys,:,:]
				new_mat[phys,np.size(MPS_1[tt],1):,np.size(MPS_1[tt],2):] = MPS_2[tt][phys,:,:]
		MPS_N.append(new_mat)

	return MPS_N

def MPS_Merge_Single_Normalise(MPS_1,MPS_2,batches):  #MPS_2 is added to MPS_1, MPS_2 is divided by the batches, assumes MPS_1 properly normalised
	MPS_N = []
	sites = len(MPS_1) #Assuming MPS_2 is the same
	phys = np.size(MPS_1[0][:,0,0]) # The physical dimension is the first element
	for tt in range(0,sites):
		if tt == 0:
			new_mat = np.zeros([2,1,np.size(MPS_1[tt],2) + np.size(MPS_2[tt],2)],dtype = complex)
		elif tt == sites -1:
			new_mat = np.zeros([2,np.size(MPS_1[tt],1) + np.size(MPS_2[tt],1),1],dtype = complex)	
		else:
			new_mat = np.zeros([2,np.size(MPS_1[tt],1) + np.size(MPS_2[tt],1),np.size(MPS_1[tt],2) + np.size(MPS_2[tt],2)],dtype = complex)
			
		for phys in range(0,2):
			if tt == 0:
				new_mat[phys,0,0:np.size(MPS_1[tt],2)] = MPS_1[tt][phys,:,:]  #MPS2 is added to MPS1 not the other way
				new_mat[phys,0,np.size(MPS_1[tt],2):np.size(new_mat,2)] = (1.0/float(batches))*MPS_2[tt][phys,:,:]
			
			elif tt == sites-1: #Note that this entry is considered 1D for some reason. Perhaps because new mat is dimension 2,X,1 and the 1 is on the end so is ignored?
				new_mat[phys,0:np.size(MPS_1[tt],1)] = MPS_1[tt][phys,:,:]
				new_mat[phys,np.size(MPS_1[tt],1):np.size(new_mat,1)] = MPS_2[tt][phys,:,:]

			else:
				new_mat[phys,0:np.size(MPS_1[tt],1),0:np.size(MPS_1[tt],2)] = MPS_1[tt][phys,:,:]
				new_mat[phys,np.size(MPS_1[tt],1):,np.size(MPS_1[tt],2):] = MPS_2[tt][phys,:,:]
		MPS_N.append(new_mat)

	return MPS_N


#############################################################################################
#Old Codes: keep Magnetisation_LEFTZIP as relevant for SDE_MPS algorithm I wrote
#def Magnetisation_LEFTZIP(MPS):  #Assumes Left Canonical Form. See Scholwock page 36 for refresh on the contraction order.
#	n = 0
#	for ii in range(0,len(MPS)):
#		if ii == len(MPS)-1:  #evaluate magnetisation on RH site so just use the fact that left-canonical. #The end of the chain is unique. Can use AxA^*
#			end_chain = Sz[0,0]*np.dot(np.conj(np.transpose(MPS[len(MPS)-1][0,:,:])),MPS[len(MPS)-1][0,:,:]) + Sz[1,1]*np.dot(np.conj(np.transpose(MPS[len(MPS)-1][1,:,:])),MPS[len(MPS)-1][1,:,:]) + Sz[0,1]*np.dot(np.conj(np.transpose(MPS[len(MPS)-1][0,:,:])),MPS[len(MPS)-1][1,:,:]) + Sz[1,0]*np.dot(np.conj(np.transpose(MPS[len(MPS)-1][1,:,:])),MPS[len(MPS)-1][0,:,:]) 
#			n =  n + 1.0/len(MPS)*end_chain	
#		else: #Otherwise, scan from the right until the site of interest (left site)
#			left_site = Sz[0,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + Sz[1,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:]) + Sz[0,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][0,:,:]) + Sz[1,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][1,:,:])
# 
#			for jj in range(ii+1,len(MPS)):
#				new_left_up = np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][0,:,:]))
#				new_left_down = np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][1,:,:])) #Bond contraction for down index
#				left_site = new_left_up + new_left_down   #Physical contraction
#            			#tadpole = np.dot(left_site,right_site)
#
#			n = n + (1.0/len(MPS))*left_site #np.trace(tadpole) 
#	return n 


