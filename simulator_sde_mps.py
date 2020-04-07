
##########VERSION_NOTES#########################
#Author: Samuel Begg, KCL Dept. of Phys
#Date: 04/04/19
#Computes Observables in SDE method 
#####EDITS_and_CAPABILITIES#####################
#Mar22
#First version of the code to be cleaned
#Apr4
#Current capability:
#---any initial condition verified
#---longitudinal field also verfied
#---able to save all trajectories into a file using StateSave
#---can evaluate local observables
#---bond noise (verified)
#---PBC and OBC verified for bond noise and site noise (need to supply inputs yourself for site noise)
#---sample killing and duplication based on norms (unverified)


#Does not have (to add):
#---XXZ code unverified
#Untrusty worthy capabilities:
#---statesave and forks not compatible 
#---splits and forks believed to be compatible only if the norm truncations is justified
#---Retain loop no longer implements norm since want to do this on MPS level later. And braket norm is not correct. 

######################################
#NOTES
#####################################
#22/03/19
#Code only works in parralel up to 24 cores due to naive imlementation, need to use more sophisticated mpi to get more cores


from joblib import Parallel, delayed
import multiprocessing as mp
#import matplotlib
#matplotlib.use('Agg')
import MPS as mps
import numpy as np
import numpy.random
import cmath
import matplotlib.pyplot as plt
import time as time
import scipy.io
from datetime import datetime
startTime = datetime.now()
print(str(startTime))
import copy

shift = 158
printf = "port" + str(shift)
timegap = 1


def controlpanel(shiftin):
	####MODEL_SELECTION###############
	ising = 1
	xxz = 0 #XXZ code yet to be verified 
	GaussInit = [0.0,0.0,0.0,0.0,0.0] #Starting parametrisation, Gauss (0) or alternate `bar' variables (1)
	a = 1.0 #determines the starting point on the Bloch sphere. a = 1.0 is down-state if GaussInit = 0 (or up state if set GaussInit = 1)
	Initxiz = -2.0*np.log(a)
	Initxiplus = np.sqrt(1.0 - a*np.conj(a))/np.abs(a) #determines the starting point  
	sites = 5 #number of sites 
	Ferro = 1 # 0 for AntiFerromagnet and 1 for Ferromagnet.
	Real = 1
	PBC = 0 # periodic boundary conditions, only valid for bond noise. For site noise the input file itself must be changed.

	systems = [ising,GaussInit,Initxiplus,Initxiz,sites,xxz,Ferro,Real,PBC]  


	####COMPUTATION_VARIABLES#Splits and forks not necessarily compatible
	bond_noise = 1 # 1 for bond noise or 0 for site noise (must supply correct noise matrix file in second case) 
	supercomp = 0 
	cores = 24
	totalsamp = 1000 #samples for averagin
	splits = 1  #divides the samples into batches.
	samp = int(totalsamp/splits) #samples per batch
	Dual = 1 #Dual = 1 if two parametrisations are used or 0 is only want Gauss parametrisation 
	sampleshift = shiftin*totalsamp
	t = 30 #number of time steps
	dt = 0.1 # time step
	forktime = int(t/5)
	forks= int(t/forktime)   #not compatible with statesave == 1 currently
	retain = totalsamp #int(totalsamp/2)
	####OPTIONAL_EXTENSIONS_FOR_RECORDING###########
	local_observable_site = 3 #Sz measured on this site (physical site index, chain starts 1..to..N)
	StateSave = 1  #save the samples. The data is saved in a matrix that can be analysed elsewhere
	Gap = 1 #save samples in matrix at these times (if StateSave = 1). The data is saved every 'Gap' time steps in the above matrix.
	DoubleBraKet = 2 #LEAVE: doubles hubbard stratonovich fields. D = 2 is necessary for decoupling the bra and ket separately. The code simply creates a second chain (implemented by doubling the number of sites and getting different noises for each half of this doubled chain). For efficiency, Loschmidt there are just twice as many samples as there would otherwise be (i.e only the bra is decoupled but since they obey the same SDE's that is ok). Running D=1 for Loschmidt echo no longer possible. D = 4 was trialled for entanglement entropy but should not be trusted. 
	reduced_den_site = 3 #LEAVE: This functionality may not be currently used in this version of the code. Will re-instate if I wanted to consider entanglement entropy of a reduced subsystem. Consider scheduling for removal
	Sym = 1 #LEAVE: implements a symmetry algorithm, 1 means nothing is implemented. This functionality should not be trusted and is scheduled for removal.
	
	compvars = [t,dt,forks,forktime,Dual,totalsamp,samp,sampleshift,splits,Sym,StateSave,Gap,DoubleBraKet,supercomp,cores,reduced_den_site,bond_noise,local_observable_site,retain]

	#############MODEL_VARIABLES##############
	hx = -0.3 
	hy = 0.0 
	hz = 0.0

	############POTENTIAL_VARIABLES_IN_EXTENSIONS_(XXZ)#########
	#XXZ has not been verified. Variables are not used if Ising is run 
	j = 1.0 #LEAVE: will not change anything. Scaled to 1 in the code. Use ferro anti-ferro code
	delta = 1.0 #anisotropy
	gx = 1.0
	gy = 1.0
	
	if xxz == 1:
		gz = delta
	else:
		gz = 1.0 #gz is used in the code and changing it here will scale J, it is not clear why someone would do this though as easier to scale to 1.
	
	modvars = [j,hx,hy,hz,gx,gy,gz,delta]

	return [systems, compvars, modvars]

def GaussMAP(Gaus,A,B,GaussInit): #map observable into Gauss param. GaussInit scheduled for removal here.
	if Gaus == 0.0:
		Ag = A	
		Bg = B
	else:	
		Ag = 1.0/A		
		Bg = B - 2.0*np.log(A)  		
	
	return [Ag,Bg]


def Loschmidt(Gaus,A,B,size,GaussInit):

	La = np.zeros((2,size), dtype = complex) #La [(all init/all flip), time]	
	A2 = np.zeros(size, dtype = complex)	
	B2 = np.zeros(size, dtype = complex)	
	for mm in range(0,size):
		[A2[mm],B2[mm]] = GaussMAP(Gaus[mm],A[mm],B[mm],GaussInit[mm]) #change the entries in the positions that are mapped.
		La[0,mm] = np.exp(-B2[mm]/2.0) #overlap with initial state 
		La[1,mm] = A2[mm]*np.exp(-B2[mm]/2.0) #overlap with FLIP state 
		#else:
		#	La[mm] = A[mm]*np.exp(-B[mm]/2.0) 

	LE_tra = np.prod(La,1)
	
	return LE_tra

def Magnetisation_Local(Gaus,Gausbra,A,B,Abra,Bbra,sites,GaussInit,special_site):

	A2 = np.zeros(sites, dtype = complex)	
	A2b = np.zeros(sites, dtype = complex)	
	B2 = np.zeros(sites, dtype = complex)	
	B2b = np.zeros(sites, dtype = complex)
	
	for mm in range(0,sites):
		[A2[mm],B2[mm]] = GaussMAP(Gaus[mm],A[mm],B[mm],GaussInit[mm]) #Map variables back to the Gauss param
	for mm in range(0,sites):
		[A2b[mm],B2b[mm]] = GaussMAP(Gausbra[mm],Abra[mm],Bbra[mm],GaussInit[mm]) #Map the bra variables back to the Gauss param
	
		f = np.zeros(sites,dtype = complex)
	coef = np.ones(sites)
	coef[special_site-1] = -1.0	
	f = -0.5*np.exp(-(np.sum(B2,0) + np.sum(np.conj(B2b),0))/2)*np.prod(1+coef*A2*np.conj(A2b),0)
	return f	

def Magnetisation(Gaus,Gausbra,A,B,Abra,Bbra,sites,GaussInit):

	A2 = np.zeros(sites, dtype = complex)	
	A2b = np.zeros(sites, dtype = complex)	
	B2 = np.zeros(sites, dtype = complex)	
	B2b = np.zeros(sites, dtype = complex)
	
	for mm in range(0,sites):
		[A2[mm],B2[mm]] = GaussMAP(Gaus[mm],A[mm],B[mm],GaussInit[mm]) #Map variables back to the Gauss param
	for mm in range(0,sites):
		[A2b[mm],B2b[mm]] = GaussMAP(Gausbra[mm],Abra[mm],Bbra[mm],GaussInit[mm]) #Map the bra variables back to the Gauss param
	
		f = np.zeros(sites,dtype = complex)
	for ii in range(0,sites):	
		coef = np.ones(sites)
		coef[ii] = -1	
		f[ii] = -0.5*np.exp(-(np.sum(B2,0) + np.sum(np.conj(B2b),0))/2)*np.prod(1+coef*A2*np.conj(A2b),0)
	Mag = 1.0/sites*sum(f,0)
	return Mag	

def EntanglementEntropy(Gaus,Gausbra,Gaus3,Gaus4,A,B,Abra,Bbra,A3,B3,A4,B4,sites,reduced_sites,GaussInit):	
	#Derivation was based on regular log not matrix log, so this function no longer works.
	A1b = np.zeros(sites, dtype = complex)	
	A2b = np.zeros(sites, dtype = complex)	
	B1b = np.zeros(sites, dtype = complex)	
	B2b = np.zeros(sites, dtype = complex)	
	
	A3b = np.zeros(sites, dtype = complex)	
	A4b = np.zeros(sites, dtype = complex)	
	B3b = np.zeros(sites, dtype = complex)	
	B4b = np.zeros(sites, dtype = complex)	
	
	for mm in range(0,sites):
		[A1b[mm],B1b[mm]] = GaussMAP(Gaus[mm],A[mm],B[mm],GaussInit[mm]) #change the entries in the positions that are mapped.
	for mm in range(0,sites):
		[A2b[mm],B2b[mm]] = GaussMAP(Gausbra[mm],Abra[mm],Bbra[mm],GaussInit[mm]) #change the entries in the positions that are mapped.
	for mm in range(0,sites):
		[A3b[mm],B3b[mm]] = GaussMAP(Gaus3[mm],A3[mm],B3[mm],GaussInit[mm]) #change the entries in the positions that are mapped.
	for mm in range(0,sites):
		[A4b[mm],B4b[mm]] = GaussMAP(Gaus4[mm],A4[mm],B4[mm],GaussInit[mm]) #change the entries in the positions that are mapped.
	
	#unit_log = np.exp(-(B2[0:reduced_sites] + np.conj(B2b[0:reduced_sites]))/2)*(1+A2[0:reduced_sites]*np.conj(A2b[0:reduced_sites]))
	#log_comp = -np.prod(unit_log*np.log2(unit_log),0)	
	#TRACED PART	
	F12 = np.exp(-(np.sum(B1b[reduced_sites:sites],0) + np.sum(np.conj(B2b[reduced_sites:sites]),0))/2)*np.prod(1+A1b[reduced_sites:sites]*np.conj(A2b[reduced_sites:sites]),0)
	F34 = np.exp(-(np.sum(B3b[reduced_sites:sites],0) + np.sum(np.conj(B4b[reduced_sites:sites]),0))/2)*np.prod(1+A3b[reduced_sites:sites]*np.conj(A4b[reduced_sites:sites]),0)
	chi12 = np.exp(-(B1b[0:reduced_sites] + np.conj(B2b[0:reduced_sites]))/2)*F12/float(sites)
	chi34 = np.exp(-(B3b[0:reduced_sites] + np.conj(B4b[0:reduced_sites]))/2)*F34/float(sites)
	#stot = chi*np.log2(np.abs(chi)) + chi*A2[0:reduced_sites]*np.log2(np.abs(chi*np.conj(A2b[0:reduced_sites]))) + chi*np.conj(A2b[0:reduced_sites])*np.log2(np.abs(chi*A2[0:reduced_sites])) + chi*np.conj(A2b[0:reduced_sites]*A2[0:reduced_sites])*np.log2(np.abs(chi*np.conj(A2b[0:reduced_sites])*A2[0:reduced_sites]))	
	#DOUBLE ABS PROVIDES DECENT APPROXIMATION
	#stot = np.abs(chi12)*np.log2(np.abs(chi)) + np.abs(chi*A2[0:reduced_sites])*np.log2(np.abs(chi*np.conj(A2b[0:reduced_sites]))) + np.abs(chi*np.conj(A2b[0:reduced_sites]))*np.log2(np.abs(chi*A2[0:reduced_sites])) + np.abs(chi*np.conj(A2b[0:reduced_sites])*A2[0:reduced_sites])*np.log2(np.abs(chi*np.conj(A2b[0:reduced_sites])*A2[0:reduced_sites]))	
	#ORIGINAL, DOES NOT WORK	
	stot = chi12*np.log2(chi34) + chi12*A1b[0:reduced_sites]*np.log2(chi34*np.conj(A4b[0:reduced_sites])) + chi12*np.conj(A2b[0:reduced_sites])*np.log2(chi34*A3b[0:reduced_sites]) + chi12*np.conj(A2b[0:reduced_sites])*A1b[0:reduced_sites]*np.log2(chi34*np.conj(A4b[0:reduced_sites])*A3b[0:reduced_sites])
	#ABS VAL	
	#stot = np.abs(chi12)*np.log2(np.abs(chi34)) + np.abs(chi12*A1b[0:reduced_sites])*np.log2(np.abs(chi34*np.conj(A4b[0:reduced_sites]))) + np.abs(chi12*np.conj(A2b[0:reduced_sites]))*np.log2(np.abs(chi34*A3b[0:reduced_sites])) + np.abs(chi12*np.conj(A2b[0:reduced_sites])*A1b[0:reduced_sites])*np.log2(np.abs(chi34*np.conj(A4b[0:reduced_sites])*A3b[0:reduced_sites]))	
	
	EE = np.prod(stot,0)

	return EE	

def Normalisation(Gaus,Gausbra,A,B,Abra,Bbra,sites,GaussInit):
	#calculates the norm, which is a complex variable like all variables in the formalism. Later, you may choose to analyse the absolute value or the real part.
	A2 = np.zeros(sites, dtype = complex)	
	A2b = np.zeros(sites, dtype = complex)	
	B2 = np.zeros(sites, dtype = complex)	
	B2b = np.zeros(sites, dtype = complex)	
	for mm in range(0,sites):
		[A2[mm],B2[mm]] = GaussMAP(Gaus[mm],A[mm],B[mm],GaussInit[mm]) #map back to Gauss param
	for mm in range(0,sites):
		[A2b[mm],B2b[mm]] = GaussMAP(Gausbra[mm],Abra[mm],Bbra[mm],GaussInit[mm]) #map back to Gauss param
	
		f = np.zeros(sites,dtype = complex)
	Norm = np.exp(-(np.sum(B2,0) + np.sum(np.conj(B2b),0))/2)*np.prod(1+A2*np.conj(A2b),0)
	return Norm	

def Correlation(Gaus,Gausbra,A,B,Abra,Bbra,sites,GaussInit):

	A2 = np.zeros(sites, dtype = complex)	
	A2b = np.zeros(sites, dtype = complex)	
	B2 = np.zeros(sites, dtype = complex)	
	B2b = np.zeros(sites, dtype = complex)	
	for mm in range(0,sites):
		[A2[mm],B2[mm]] = GaussMAP(Gaus[mm],A[mm],B[mm],GaussInit[mm]) #change the entries in the positions that are mapped.
	for mm in range(0,sites):
		[A2b[mm],B2b[mm]] = GaussMAP(Gausbra[mm],Abra[mm],Bbra[mm],GaussInit[mm]) #change the entries in the positions that are mapped.
	
		f = np.zeros(sites,dtype = complex)
	for ii in range(0,sites):	
		coef = np.ones(sites)
		coef[ii] = -1.0 #note that this is the only difference between the expression for magnetisation and the normalisation	
		coef[(ii+1)%sites] = -1.0 #note that this is the only difference between the expression for magnetisation and the normalisation	
		f[ii] = 0.25*np.exp(-(np.sum(B2,0) + np.sum(np.conj(B2b),0))/2)*np.prod(1+coef*A2*np.conj(A2b),0)
	Corr = 1.0/sites*sum(f,0)
	return Corr	

def StochMeth(shiftin,printfile,timegapsave,system,compvar,modvar):    

	[ising,GaussInit,Initxiplus,Initxiz,sites,xxz,Ferro,Real,PBC] =  system 

	[t,dt,forks,forktime,Dual,totalsamp,samp,sampleshift,splits,Sym,StateSave,Gap,DoubleBraKet,supercomp,cores,red_site,bond_noise,local_observable_site,retain] = compvar
	
	[j,hx,hy,hz,gx,gy,gz,delta] = modvar

	
	saveUNI = 'Ising' + 'sites' + str(sites) + 'overlapmat_' + 'dt_' + str(dt) +'delta_' + str(delta) + 'ferro_' +str(Ferro) 
	saveUNI = 'means'
	###############Testing###################
	plotED = 0 
	tester = 0
	siteplot = 0
	#########################################

	#input the noise matrix and the eigenvectors. Note that noise matrix is yet to be rescaled, but has been diagonalised.
	if supercomp == 0:
		num_cores = mp.cpu_count()

	if supercomp == 1:
		num_cores = cores

	if bond_noise == 0: 
		if supercomp == 0:
		
			Pmat = np.genfromtxt('/home/samuelbegg/Documents/Simulation/DiagData/P' + str(sites) +'.dat',dtype = complex, delimiter = '')
			Diag = np.genfromtxt('/home/samuelbegg/Documents/Simulation/DiagData/D' + str(sites) +'.dat',dtype = complex, delimiter = '')
			num_cores = mp.cpu_count()
		
		if supercomp == 1:
			Pmat = np.genfromtxt('DiagData/P' + str(sites) +'.dat',dtype = complex, delimiter = '')
			Diag = np.genfromtxt('DiagData/D' + str(sites) +'.dat',dtype = complex, delimiter = '')
			num_cores = cores

		#This manipulation is done since the diagonalised noise matrix is complex and has to be rescaled
		Dhaf = np.zeros([sites,sites],dtype = complex)
		for kk in range(0,sites):
			Dhaf[kk,kk] = 1.0/(cmath.sqrt(Diag[kk])) 

		Pscaled = np.dot(Pmat,Dhaf) 
		Kmat = np.zeros([sites],dtype = complex)
		Ps2 = Pscaled**2
		for ii in range(0,sites):
			for kk in range(0,sites):
				Kmat[ii] = Kmat[ii] + Ps2[ii,kk]
	else:
		Pscaled = 1   #dummy, since need to initialise as gets put into the pararellisation. Ideally would remove that capability. Unsure how to ensure flexibility simply.

	#########################################
	#This will print in the log file so that there is a record of the simulation inputs. May need to make adaptions over time as the functionality of the code changes 
	print(printfile, "_Ising" + str(ising), "_J" + str(j), "_hx" + str(hx),"_hy" + str(hy),"_hz" + str(hz),"_anisotropy" + str(delta), "_sites" + str(sites), "_ferro" + str(Ferro), "_real" + str(Real), "_steps" + str(t),"_dt"+str(dt),"_totalsamp" + str(totalsamp), "_splits" + str(splits), "_samp" + str(samp),"GaussInit_" + str(GaussInit[0]), "_init_xiplus_" + str(Initxiplus), "_init_xiz_" + str(Initxiz), "_date" + str(startTime), "_dual" + str(Dual))

	###############INITIALISE################
	##############NORM_SELECTION#############
	#this code is currently inactive
	#initialise_xiplus = Initxiplus*np.ones([samp,sites*2], dtype = complex)                                                                  
	#initialise_xiz = Initxiz*np.ones([samp,sites*2], dtype = complex) 
	#############OBSERVABLES_INITIALISATION_FOR_ALL_SAMPLES###########################  

		
	mps_norm = np.zeros(int(t/Gap),dtype = complex)
	mps_mag = np.zeros(int(t/Gap),dtype = complex)
	entanglement = np.zeros((t,splits),dtype = complex)  
	normalisation = np.zeros((t,splits),dtype = complex)  
	correlations = np.zeros((t,splits),dtype = complex)  
	magnetisation = np.zeros((t,splits),dtype = complex)  
	magnetisation_local = np.zeros((t,splits),dtype = complex)  
	overlap = np.zeros((t,splits),dtype = complex)  
	overlapFLIP = np.zeros((t,splits),dtype = complex)  
	le = np.zeros((t),dtype = complex) 
	tvec = np.abs(j)*dt*np.arange(0,t)
	burn = np.zeros((samp),dtype = int) 

	for mem in range(1,splits+1):   #Splits the computation into a series of smaller computations by samples. I.e. so that data is saved every 50,000 samples rather than 2 million (which could result in data loss if there was a crash)	
		initialise_xiplus = np.zeros([samp,sites*2], dtype = complex)
		initialise_xiz = np.zeros([samp,sites*2], dtype = complex)
		initialise_Gauss = np.zeros([samp,sites*2])  #	
		
		for forker in range(1,forks+1):		
			prodstate = np.zeros((DoubleBraKet*samp, sites, 2, int(forktime/Gap)),dtype = complex) #samples,sites,spin dimension,times 

			pool = mp.Pool(num_cores)
			results = [pool.apply(GenNoise,args = [system,compvar,modvar,vv,tester,siteplot,Pscaled,forker,initialise_xiplus[vv % samp,:],initialise_xiz[vv % samp,:],initialise_Gauss[vv % samp,:]]) for vv in range(samp*(mem-1),samp*(mem))] #Runs the SDE algorithm GenNoise in parallel
			pool.close()

			magnetsamps = np.zeros((forktime,samp),dtype = complex) 
			magnetsamps_local = np.zeros((forktime,samp),dtype = complex) 
			#magnetsampsDIAG = np.zeros((t,samp),dtype = complex) 
			normsamps = np.zeros((forktime,samp),dtype = complex)  
			#normsampsDIAG = np.zeros((t,samp),dtype = complex)  
			entangsamps = np.zeros((forktime,samp),dtype = complex) 
			corrsamps = np.zeros((forktime,samp),dtype = complex) 
			LE_tr = np.zeros((2,forktime,samp),dtype = complex) #first index is the allup/alldown for the symmetrised ground state 
			
			new_samples_norm = np.zeros(int(retain),dtype = complex)
		
	
			
			kk = 0 
			for jj in range(0,samp):
				burn[jj] = results[jj][1]
				if burn[jj] == 1:	
					[LE_tr[:,:,kk],magnetsamps[:,kk],magnetsamps_local[:,kk],normsamps[:,kk],entangsamps[:,kk],corrsamps[:,kk]] = results[jj][0] #loop across sample index to put into friendly form 
					
					if StateSave == 1:
						
						placeholder = results[jj][5] #want to do 2 samples at a time here 	
						placeholder2 = results[jj][6]	
						#samples,sites,spin dimension,times		
						prodstate[kk,:,0,:] = placeholder[0:sites] #want to do 2 samples at a time here 	
						prodstate[kk,:,1,:] = placeholder2[0:sites]
						
						if DoubleBraKet == 2:	
							prodstate[samp+kk,:,0,:] = placeholder[sites:2*sites] #Since we save wavefunctions rather than the density matrix the second set of variables from the bra just become extra samples for the kets.	
							prodstate[samp+kk,:,1,:] = placeholder2[sites:2*sites]	
						
					kk = kk + 1 
			#print(samp - kk, 'Total # Burns')  #if this is non-zero then the algorithm for avoiding divergences is not working
	
			phys_dim = 2
			batches = 2
			blocks = 2
			mega_blocks = 2
			batch_size = int(2*samp/(batches*blocks*mega_blocks)) #but what is the size of the block? The bond dimension of the output array. 
			bond_dim = 20
			print('check')	
			for tloop in range(0,int(forktime/Gap)):
				MPS_Mega = []
				#data = prodstate[:,:,:,(forker-1)*(forktime/Gap)+tloop]
				for meg in range(0,mega_blocks):
					MPS_Block = []
					for  block_index in range(0,blocks):

						#for vv in range(0,batches):
						#	print(tloop,vv + batches*block_index + batches*blocks*meg + 1)
	
						MPS_List = Parallel(n_jobs = num_cores)(delayed(mps.Data_to_MPS)(prodstate[(vv + batches*block_index + batches*blocks*meg)*batch_size:(vv + batches*block_index + batches*blocks*meg + 1)*batch_size,:,:,tloop ],sites,phys_dim,batch_size) for vv in range(0,batches))  	
						MPS_List_C = Parallel(n_jobs=num_cores)(delayed(mps.Canonicalise_Parallel)(MPS_List[vv],bond_dim,0) for  vv in range(0,batches)) 		
						MpS_C,s,norm = mps.compressor_post_parallel_normop(MPS_List_C,bond_dim,batches,0)  	
						#MpS = mps.Data_to_MPS(data,sites,2,samp)
						#MpS_C,schmidt,norm = mps.Canonicalise_Normed(MpS)
						MPS_Block.append(MpS_C)	
					Mblock_C,s,norm = mps.compressor_post_parallel_normop(MPS_Block,bond_dim,blocks,0)  	
					MPS_Mega.append(Mblock_C)	
	
				Mfinal,s,norm = mps.compressor_post_parallel_normop(MPS_Mega,bond_dim,mega_blocks,1)  

				mps_norm[(forker-1)*(int(forktime/Gap))+tloop] = norm
				mps_mag[(forker-1)*(int(forktime/Gap))+tloop] = mps.Magnetisation_LEFTZIP(Mfinal)
			#Must take the mean across the batches
			#ECHOES	
			Observable1 = LE_tr[0,:,0:(kk)] 
			overlap[forktime*(forker-1):forktime*forker,splits-1]	= np.mean(Observable1,1) 
			Observable2 = LE_tr[1,:,0:(kk)] 
			overlapFLIP[forktime*(forker-1):forktime*forker,splits-1] = np.mean(Observable2,1)
			
			#MAGNETISATION(forker-1)*(forktime/Gap)+tloop
			magnetisation[forktime*(forker-1):forktime*forker,splits-1] = np.mean(magnetsamps,1)
		
			magnetisation_local[forktime*(forker-1):forktime*forker,splits-1] = np.mean(magnetsamps_local,1)

			#NORMALISATION
			#print(normsamps)
			normalisation[forktime*(forker-1):forktime*forker,splits-1] = np.mean(normsamps,1)
			#normDIAG = np.mean(normsampsDIAG,1)
			#magDIAG = np.mean(magnetsampsDIAG,1)	

			#ENTANGLEMENT
			entanglement[forktime*(forker-1):forktime*forker,splits-1] = -np.mean(entangsamps,1)

			#CORRELATIONS
			correlations[forktime*(forker-1):forktime*forker,splits-1] = np.mean(corrsamps,1)		
	
			if forks > 1 and forker < forks:		
	
				index = Rank_Norms(normsamps[forktime-1,:])

				norm_scale = 0
				
					
							
				for kk in range(0,retain):	
					new_samples_norm[kk] = normsamps[forktime-1,index[kk]]
				reduced_samples_norm = np.mean(new_samples_norm)
				#print(np.percentile(np.abs(np.real(normsamps[forktime-1,:])),25),'1st quartile')
				#print(np.percentile(np.abs(np.real(normsamps[forktime-1,:])),50),'2nd quartile')
				#print(np.percentile(np.abs(np.real(normsamps[forktime-1,:])),75),'3rd quartile')
				#print(np.mean(np.abs(np.real(normsamps[forktime-1,:]))),'mean')
				#print(np.median(np.abs(np.real(normsamps[forktime-1,:]))),'mean')
				#print(np.std(np.abs(np.real(normsamps[forktime-1,:]))),'standard deviation')
				#plt.hist(np.log(np.abs(np.real(normsamps[forktime-1,:]))),bins = 100)
				#plt.show()
				#print(reduced_samples_norm,'reduced_samples_norm')
				
				for jj in range(0,retain):  #scan over the #remain largest norms
					for cc in range(0,int(totalsamp/retain)): #if the ratio is 2 this will create the second lot of trajectories to replace the discards
						initialise_xiplus[jj + retain*cc,:] = results[index[jj]][2]    
						initialise_xiz[jj + retain*cc,:] = results[index[jj]][3] # + (1.0/sites)*np.log(np.real(reduced_samples_norm))                                                     
						initialise_Gauss[jj + retain*cc,:] = results[index[jj]][4] 
					#initialise_Gauss[jj + retain*cc,:] = results[index[jj]][4] 



	
		if supercomp == 1: 
			#Save the observables (after taking mean above)
			np.save('/home/mmm0308/Scratch/'+str(printfile)+'/overlap_' + str(mem), overlap[::timegapsave,splits-1])
			np.save('/home/mmm0308/Scratch/'+str(printfile)+'/FLIPoverlap_' + str(mem), overlapFLIP[::timegapsave,splits-1])
			np.save('/home/mmm0308/Scratch/'+str(printfile)+'/magnetisation_' + str(mem), magnetisation[::timegapsave,splits-1])
			np.save('/home/mmm0308/Scratch/'+str(printfile)+'/magnetisation_local_' + str(mem), magnetisation_local[::timegapsave,splits-1])
			#np.save('/home/mmm0308/Scratch/'+str(printfile)+'/magDIAG_' + str(mem), magDIAG[::timegapsave])
			np.save('/home/mmm0308/Scratch/'+str(printfile)+'/normalisation_' + str(mem), normalisation[::timegapsave,splits-1])
			#np.save('/home/mmm0308/Scratch/'+str(printfile)+'/normDIAG_' + str(mem), normDIAG[::timegapsave])
			np.save('/home/mmm0308/Scratch/'+str(printfile)+'/correlations_' + str(mem), correlations[::timegapsave,splits-1])
			np.save('/home/mmm0308/Scratch/'+str(printfile)+'/mps_norm_' + str(mem), mps_norm[::timegapsave])
			np.save('/home/mmm0308/Scratch/'+str(printfile)+'/mps_mag' + str(mem), mps_mag[::timegapsave])

			#if StateSave == 1:
			#	for uu in range(0,2):	
			#		scipy.io.savemat('/home/mmm0308/Scratch/' + str(printfile) +'/prodstate' + str(mem) + '_half_' + str(uu),{'vect':prodstate[:,:,:,uu*int(0.5*t/Gap):(uu+1)*int(0.5*t/Gap)]})             

		else:  #Not on supercomputer. Note that observables are not currently saved here. You can save by duplicating the above code. Can still look at plots though.
			#Save the data 
			if StateSave == 1:			
				for uu in range(0,2):	
		#			scipy.io.savemat('/home/k1623105/my_local_scratch/Storage_MPS/TestBatch/prodstate' + str(mem) + '_half_' + str(uu),{'vect':prodstate[:,:,:,uu*int(0.5*t/Gap):(uu+1)*int(0.5*t/Gap)]})         		
					print('not saving currently')    

				#Not currently used but potential solution to output in mat form for matlab:
					#matfile = '/home/k1623105/Documents/PhD/MPS_Samples/test_mat_'+str(mem)+'.mat'
					# Write the array to the mat file. For this to work, the array must be the value
					# corresponding to a key name of your choice in a dictionary
					#scipy.io.savemat(matfile, mdict={'out': prodstate}, oned_as='row')

	if supercomp == 0: #Various plots for comparison with ED or other method.
#
#		mag_average = np.zeros(t,dtype = complex) 
#		for kk in range(1,np.size(mag_average)):
#			mag_average[kk] = np.mean(np.real(magnetisation[1:kk])/np.abs(normalisation[1:kk]))
#		#f, (ax2,ax4) = plt.subplots(2, 1, sharex=True)                                                                                                                                                  
		recent = np.load('/home/samuelbegg/Desktop/recentZ.npy')                                                                                                                                              
#		#plt.plot(dt*np.arange(0,np.size(magnetisation)),np.real(magnetisation),label = 'Ensemble')
		
		plt.plot((Gap*dt)*np.arange(1,np.size(mps_mag)+1),mps_mag,label = 'MPS_Mag')
		plt.plot((Gap*dt)*np.arange(1,np.size(mps_norm)+1),mps_norm,label = 'MPS_Norm')
		plt.plot(dt*np.arange(1,np.size(magnetisation)+1),np.real(magnetisation),label = 'Magnetisation')
		#plt.plot(dt*np.arange(0,np.size(magnetisation)),np.real(magnetisation)/np.abs(normalisation),label = 'Magnetisation SDE')
		#plt.plot(dt*np.arange(0,np.size(magnetisation)),normalisation,label = 'Normalisation')
		plt.ylim(-2,2)
#		#plt.plot(dt*np.arange(0,np.size(magDIAG)),np.real(magDIAG)/np.abs(normDIAG),label = 'Diag')
#		plt.plot(dt*np.arange(0,np.size(mag_average)),np.real(mag_average),label = 'Time_Average')
#		plt.legend()	
		plt.plot(0.01*np.arange(0,np.size(recent)),recent,label = 'ED')
		plt.legend()
		plt.show()



def Rank_Norms(Norm_Mat):									     

	indexed_mat = list(enumerate(-np.abs(np.real(Norm_Mat))))                                    
	order = [i[0] for i in sorted(indexed_mat, key=lambda x:x[1])]                       
									     
	return order    



def GenNoise(systems,compvars,modvars,ww,tester,siteplot,Pscaled,forker,initialise_Xiplus,initialise_Xiz,initialise_gauss):  	

	[Ising,GaussInit,Initxiplus,Initxiz,numsites0,XXZ,ferro,real,PBC] = systems 
	[timesteps,Dt,forks,forktime,dual,totalsamp,samp,sampshift,splits,symm,StateSave,gap,DOUBLEbraket,supercomp,cores,red_site,bond_noise,local_observable_site,retain] = compvars
	[J,hx,hy,hz,gX,gY,gZ,delta] = modvars

	timesteps = int(forktime) #this is a hack that should be reconciled.

	GaussO = list(GaussInit)	
	numsites = DOUBLEbraket*numsites0 #trick for doubling the number of samples for both bra and ket decoupling	

	#Number of symmetry loops and symmetry variables	
	LE_prodSym = np.zeros((2,timesteps,symm),dtype = complex) #symm index should be 1 and therefore not currently used

	#Generate the random numbers. Shifts take into account the seed (caused by shift choice at top of the file). ww takes into account the different batches. 
	if XXZ == 1:
		np.random.seed(ww + sampshift*(1+forker))
		dRx =  numpy.random.multivariate_normal(np.zeros(numsites),np.identity(numsites), timesteps)
		np.random.seed(ww + sampshift*(2+forker))
		dRy =  numpy.random.multivariate_normal(np.zeros(numsites),np.identity(numsites), timesteps)
	if bond_noise == 0:
		np.random.seed(ww + sampshift*(3+forker))
		dRz =  numpy.random.multivariate_normal(np.zeros(numsites),np.identity(numsites), timesteps)
	if bond_noise == 1:
		np.random.seed(ww + sampshift*(3+forker))
		dRz1 =  numpy.random.multivariate_normal(np.zeros(numsites),np.identity(numsites), timesteps)
		np.random.seed(ww + sampshift*(6+forker))
		dRz2 =  numpy.random.multivariate_normal(np.zeros(numsites),np.identity(numsites), timesteps)
	
	if StateSave == 1:	
			upstore = np.zeros((numsites,int(timesteps/gap)),dtype = complex)
			downstore = np.zeros((numsites,int(timesteps/gap)),dtype = complex) 
			bb = np.zeros(numsites, dtype = int) #for the upstore/downstore matrices loop later on	

	
	if DOUBLEbraket > 1:
		if forker >1:
			Gauss = initialise_gauss	
		else:				
			Gauss = list(GaussO)	
			for vv in range(0,DOUBLEbraket-1):
				Gauss = np.append(Gauss,GaussO) #Double the cahin for the bra/ket decouplings
	else:
		Gauss = list(GaussInit)
	
	LE_prod = np.zeros((2,timesteps),dtype = complex)  
	Magnet = np.zeros((timesteps),dtype = complex)  
	Magnet_Local = np.zeros((timesteps),dtype = complex)  
	#MagnetDIAG = np.zeros((timesteps),dtype = complex)  
	Entang = np.zeros((timesteps),dtype = complex)  
	Norm = np.zeros((timesteps),dtype = complex)  
	#NormDIAG = np.zeros((timesteps),dtype = complex)  
	Correl = np.zeros((timesteps),dtype = complex)  

		
	success = 1 # Used as a flag. Should always be 1. If there is an infinity then success = 0 will be set in the code further down. Which will then print that this has occurred and there is an issue. Generally a way to detect whether the two parametrisation mapping is working, since this is the way infinities arise at poles.

	#INITIALISATION

	#Initialise the SDE variables for the Stratonovich algorithm
	#xi_plus
	AT = np.zeros((timesteps+1,numsites),dtype = complex) 
	ATest = np.zeros((timesteps+1,numsites),dtype = complex) 
	dAT = np.zeros((timesteps+1,numsites),dtype = complex)
	dATest = np.zeros((timesteps+1,numsites),dtype = complex) 

	#xi_z	
	BT = np.zeros((timesteps+1,numsites),dtype = complex) 
	BTest = np.zeros((timesteps+1,numsites),dtype = complex) 
	dBT = np.zeros((timesteps+1,numsites),dtype = complex) 
	dBTest = np.zeros((timesteps+1,numsites),dtype = complex)

	if forker == 1:
		AT[0,:] = Initxiplus*np.ones((numsites),dtype = complex) 
		BT[0,:] = Initxiz*np.ones((numsites),dtype = complex)
	else:
		AT[0,:]  = initialise_Xiplus
		BT[0,:] = initialise_Xiz		
	#If you want to find the distribution of xi_minis in the Gauss paramtrisation then these must be initialised. Has not been verified
	#CT = np.zeros((timesteps,numsites),dtype = complex) #r^2 where r = |xi_+| in paper
	#dCT = np.zeros((timesteps,numsites),dtype = complex) #r^2 where r = |xi_+| in paper
	#CT[0::,:] = 0.0 + 0.0*1.0j

	
	if bond_noise == 0:	
		if XXZ == 1: #need extra noises if the XXZ algorithm is running, has not been verified	
			dWx = np.zeros((np.size(dRx,0),np.size(dRx,1)),dtype = complex)
			dWy = np.zeros((np.size(dRy,0),np.size(dRy,1)),dtype = complex)
		dWz = np.zeros((np.size(dRz,0),np.size(dRz,1)),dtype = complex)
		for jj in range(0,DOUBLEbraket):
			if XXZ == 1: #has not been verified
				dWx[:,(jj*numsites0):(jj+1)*numsites0] = np.dot(dRx[:,(jj*numsites0):(jj+1)*numsites0],np.transpose(Pscaled))
				dWy[:,(jj*numsites0):(jj+1)*numsites0] = np.dot(dRy[:,(jj*numsites0):(jj+1)*numsites0],np.transpose(Pscaled))
			dWz[:,(jj*numsites0):(jj+1)*numsites0] = np.sqrt(gZ)*np.dot(dRz[:,(jj*numsites0):(jj+1)*numsites0],np.transpose(Pscaled))
	if bond_noise == 1:
		dWz = np.zeros((np.size(dRz1,0),np.size(dRz1,1)),dtype = complex)
		for ee in range(0,numsites):
			#need to be careful here. Labelling end of chain noise to be indexed with the last site rather than first. 
			if ee == 0: #for the kets start of chain
				if PBC == 1:
					dWz[:,ee] = ((1.0/2.0)**(0.5))*np.sqrt(gZ)*(dRz1[:,numsites0-1] + dRz1[:,ee] - 1j*dRz2[:,numsites0-1] + 1j*dRz2[:,ee])  #looks good,wraps around
				else:
					dWz[:,ee] = ((1.0/2.0)**(0.5))*np.sqrt(gZ)*(dRz1[:,ee] + 1j*dRz2[:,ee])  #drop the wrap around chain
			elif ee == numsites0: #for the bras start of chain
				if PBC == 1: 
					dWz[:,ee] = ((1.0/2.0)**(0.5))*np.sqrt(gZ)*(dRz1[:,numsites-1] + dRz1[:,ee] - 1j*dRz2[:,numsites-1] + 1j*dRz2[:,ee]) #wraps around on the second set of sites (bra)
				else:
					dWz[:,ee] = ((1.0/2.0)**(0.5))*np.sqrt(gZ)*(dRz1[:,ee] +  1j*dRz2[:,ee])   #drops the wrapped around noise
			
			elif ee == numsites0-1: #for the kets end of chain
				if PBC == 1:
					dWz[:,ee] = ((1.0/2.0)**(0.5))*np.sqrt(gZ)*(dRz1[:,ee-1] + dRz1[:,ee] - 1j*dRz2[:,ee-1] + 1j*dRz2[:,ee])  #appears to be correct
				else:
					dWz[:,ee] = ((1.0/2.0)**(0.5))*np.sqrt(gZ)*(dRz1[:,ee-1] - 1j*dRz2[:,ee-1])  #drop the last bond that is on end of chain
			elif ee == numsites-1: #for the bras end of chain
				if PBC == 1:
					dWz[:,ee] = ((1.0/2.0)**(0.5))*np.sqrt(gZ)*(dRz1[:,ee-1] + dRz1[:,ee] - 1j*dRz2[:,ee-1] + 1j*dRz2[:,ee])  #appears correct
				else:
					dWz[:,ee] = ((1.0/2.0)**(0.5))*np.sqrt(gZ)*(dRz1[:,ee-1] - 1j*dRz2[:,ee-1])  #drops the noise on the end of the chain
			
			else:				
				dWz[:,ee] = ((1.0/2.0)**(0.5))*np.sqrt(gZ)*(dRz1[:,ee-1] + dRz1[:,ee] - 1j*dRz2[:,ee-1] + 1j*dRz2[:,ee])
	
	if ferro == 0: #AntiFerro case
		if real == 0:
			f = 1.0
			q = f

		if real == 1: 	
			f = (1.0/np.sqrt(2.0))*(1.0 - 1.0j)                                                                                                         
			q = f	
	
	if ferro == 1: #Ferromagnetic case 
		if real == 0:
			f = 1.0	
			q = -1.0j*f
		if real == 1:
			f = (1.0/np.sqrt(2.0))*(1.0 + 1.0j)                                
			q = f        

		
	for i in range(0,timesteps):                                              
		
		for mm in range(0,numsites):
			hzG = copy.deepcopy(hz)	
			hyG = copy.deepcopy(hy)	
			if Gauss[mm] == 1:  #AG param. Reverse Noise directions and magnetic fields
				dWz[i,mm] = -dWz[i,mm]
				hzG = -copy.deepcopy(hz)
				hyG = -copy.deepcopy(hy)
				if XXZ == 1:
					dWy[i,mm] = -dWy[i,mm]
				
			if success == 1:				
				
				if Ising == 1:	

					 
					dATest[i+1,mm] = (-1.0j*0.5*(hyG)*(-1.0j - 1.0j*AT[i,mm]**2.0))*Dt +  (-1.0j*0.5*hx*(1.0 - AT[i,mm]**2.0))*Dt  + (q*np.sqrt(Dt)*dWz[i,mm])*AT[i,mm] - 1.0j*hzG*Dt*AT[i,mm]  	
					dBTest[i+1,mm] =  1.0j*AT[i,mm]*(hx + 1.0j*hyG)*Dt + q*np.sqrt(Dt)*dWz[i,mm] - 1.0j*hzG*Dt 
					#dCT[i+1,mm] =  -1.0j*np.exp(BT[i+1,mm])*0.5*hx*Dt 
				
					ATest[i+1,mm] = AT[i,mm] + dATest[i+1,mm]               						 
					BTest[i+1,mm] = BT[i,mm] + dBTest[i+1,mm]
					#CT[i+1,mm] = CT[i,mm] + dCT[i+1,mm]

					Aestcontr = (-1.0j*0.5*(hyG)*(-1.0j - 1.0j*ATest[i+1,mm]**2.0))*Dt +  (-1.0j*0.5*hx*(1.0 - ATest[i+1,mm]**2.0))*Dt  + (q*np.sqrt(Dt)*dWz[i,mm])*ATest[i+1,mm] -1.0j*hzG*Dt*ATest[i+1,mm]	
					Aleftcontr =  (-1.0j*0.5*(hyG)*(-1.0j - 1.0j*AT[i,mm]**2.0))*Dt  +  (-1.0j*0.5*hx*(1.0 - AT[i,mm]**2.0))*Dt  + (q*np.sqrt(Dt)*dWz[i,mm])*AT[i,mm] - 1.0j*hzG*Dt*AT[i,mm]	
					AT[i+1,mm] = AT[i,mm] + 0.5*(Aestcontr + Aleftcontr)	

					Bestcontr = 1.0j*ATest[i+1,mm]*(hx + 1.0j*hyG)*Dt + q*np.sqrt(Dt)*dWz[i,mm]  - 1.0j*hzG*Dt  
					Bleftcontr = 1.0j*AT[i,mm]*(hx + 1.0j*hyG)*Dt + q*np.sqrt(Dt)*dWz[i,mm]  - 1.0j*hzG*Dt  
					BT[i+1,mm] = BT[i,mm] + 0.5*(Bestcontr + Bleftcontr)	


				if XXZ == 1: #Has not been verified 
	
					dATest[i+1,mm] = (q/2.0)*(1.0-AT[i,mm]**2.0)*np.sqrt(Dt)*dWx[i,mm] - 1.0j*(q/2.0)*(1.0+AT[i,mm]**2.0)*np.sqrt(Dt)*dWy[i,mm] + q*(AT[i,mm])*np.sqrt(Dt)*dWz[i,mm]                          
					dBTest[i+1,mm] = (q)*np.sqrt(Dt)*dWz[i,mm] - (q)*AT[i,mm]*np.sqrt(Dt)*dWx[i,mm] - q*1.0j*AT[i,mm]*np.sqrt(Dt)*dWy[i,mm]
					#dCT[i+1,mm]  = (q/4.0)*np.exp(BT[i,mm])*(np.sqrt(Dt)*dWx[i,mm] + 1.0j*np.sqrt(Dt)*dWy[i,mm]) 
					
					ATest[i+1,mm] = AT[i,mm] + dATest[i+1,mm]                       
					BTest[i+1,mm] = BT[i,mm] + dBTest[i+1,mm]
					#CT[i+1,mm] = CT[i,mm] + dCT[i+1,mm]	
				
					Aestcontr = (q/2.0)*(1.0-ATest[i+1,mm]**2.0)*np.sqrt(Dt)*dWx[i,mm] - 1.0j*(q/2.0)*(1.0+ATest[i+1,mm]**2.0)*np.sqrt(Dt)*dWy[i,mm] + q*(ATest[i+1,mm])*np.sqrt(Dt)*dWz[i,mm]                          
					Aleftcontr = dATest[i+1,mm]				
					AT[i+1,mm] = AT[i,mm] + 0.5*(Aestcontr + Aleftcontr)	

					Bestcontr = (q)*np.sqrt(Dt)*dWz[i,mm] - (q)*ATest[i+1,mm]*np.sqrt(Dt)*dWx[i,mm] - q*1.0j*ATest[i+1,mm]*np.sqrt(Dt)*dWy[i,mm]
					Bleftcontr = dBTest[i+1,mm]			
					BT[i+1,mm] = BT[i,mm] + 0.5*(Bestcontr + Bleftcontr)	


			if (success == 1):


				if StateSave == 1: #Saving the data. Code maps back to the Gauss param. Saves as elements in wavefunction rather than as the parameters in the SDE's. Can move back easily later though as simply related.
					if Gauss[mm]==0:
						
						if ((i+1) % gap == 0) :
							upstore[mm,bb[mm]] = AT[i+1,mm]*np.exp(-BT[i+1,mm]/2.0) 											
							downstore[mm,bb[mm]] = np.exp(-BT[i+1,mm]/2.0) 					
							bb[mm] = bb[mm] + 1
 
					if Gauss[mm] ==1:	
						if ((i+1) % gap == 0) :
							downstore[mm,bb[mm]] = AT[i+1,mm]*np.exp(-BT[i+1,mm]/2.0) 					
							upstore[mm,bb[mm]] = np.exp(-BT[i+1,mm]/2.0) 	
							bb[mm] = bb[mm] + 1			
					
				if (np.abs(AT[i+1,mm]) > 1.0) and (dual == 1):
					
					BT[i+1,mm] = BT[i+1,mm] - 2.0*np.log(AT[i+1,mm])  		
					AT[i+1,mm] = 1.0/AT[i+1,mm]	
					Gauss[mm] = (Gauss[mm] + 1) % 2 

	#END OF SPATIAL LOOP, mm
	#############################################################################################################################################
	#Observables Plugins
		
		LE_regular = Loschmidt(Gauss[0:numsites0],AT[i+1,0:numsites0],BT[i+1,0:numsites0],numsites0,GaussInit)
		
		if DOUBLEbraket > 1:
			LE_tilde = Loschmidt(Gauss[numsites0:numsites],AT[i+1,numsites0:numsites],BT[i+1,numsites0:numsites],numsites0,GaussInit)
			LE_prod[:,i] = 0.5*(LE_regular + LE_tilde)  #Doubles the number of samples for the Loschmidt is Double bra ket = 2
		else: 	
			LE_prod[:,i] = LE_regular

		#LE_prodSym[:,i+1,pp] = LE_prod[:,i+1] #Deactivate ready for decommissioning
		
		if DOUBLEbraket >= 2:		
			Magnet[i] = Magnetisation(Gauss[0:numsites0],Gauss[numsites0:2*numsites0],AT[i+1,0:numsites0],BT[i+1,0:numsites0],AT[i+1,numsites0:2*numsites0],BT[i+1,numsites0:2*numsites0],numsites0,GaussInit)	
			Magnet_Local[i] = Magnetisation_Local(Gauss[0:numsites0],Gauss[numsites0:2*numsites0],AT[i+1,0:numsites0],BT[i+1,0:numsites0],AT[i+1,numsites0:2*numsites0],BT[i+1,numsites0:2*numsites0],numsites0,GaussInit,local_observable_site)	
			#MagnetDIAG[i+1] = Magnetisation(Gauss[0:numsites0],Gauss[0:numsites0],AT[i+1,0:numsites0],BT[i+1,0:numsites0],AT[i+1,0:numsites0],BT[i+1,0:numsites0],numsites0,GaussInit)	
			Norm[i] = Normalisation(Gauss[0:numsites0],Gauss[numsites0:2*numsites0],AT[i+1,0:numsites0],BT[i+1,0:numsites0],AT[i+1,numsites0:2*numsites0],BT[i+1,numsites0:2*numsites0],numsites0,GaussInit)		   
			#NormDIAG[i+1] = Normalisation(Gauss[0:numsites0],Gauss[0:numsites0],AT[i+1,0:numsites0],BT[i+1,0:numsites0],AT[i+1,0:numsites0],BT[i+1,0:numsites0],numsites0,GaussInit)		   
			Correl[i] = Correlation(Gauss[0:numsites0],Gauss[numsites0:2*numsites0],AT[i+1,0:numsites0],BT[i+1,0:numsites0],AT[i+1,numsites0:2*numsites0],BT[i+1,numsites0:2*numsites0],numsites0,GaussInit)	
		if DOUBLEbraket == 4:	
			Entang[i+1] = EntanglementEntropy(Gauss[0:numsites0],Gauss[numsites0:2*numsites0],Gauss[2*numsites0:3*numsites0],Gauss[3*numsites0:4*numsites0],AT[i+1,0:numsites0],BT[i+1,0:numsites0],AT[i+1,numsites0:2*numsites0],BT[i+1,numsites0:2*numsites0],AT[i+1,2*numsites0:3*numsites0],BT[i+1,2*numsites0:3*numsites0],AT[i+1,3*numsites0:4*numsites0],BT[i+1,3*numsites0:4*numsites0],numsites0,red_site,GaussInit)	
###############	
	#LE_out = np.mean(LE_prodSym,2) #Deactivate ready for decommissioning
	var = [LE_prod,Magnet,Magnet_Local,Norm,Entang,Correl]


	#####Possible point for saving individual noise realisation and phase space variable#####
	
	#np.save('Noise.npy',dRz)
	#np.save('ATsingle.npy',np.abs(AT))
	
	#END OF THE TIME LOOP.
	############################################################################################################################################
	if forks > 1:
		if StateSave == 1:
			return	[var,success,AT[i+1,:],BT[i+1,:],Gauss,upstore,downstore]
		else:
			return [var,success,AT[i+1,:],BT[i+1,:],Gauss]
		

	if StateSave == 1:	
		return [var,success,upstore,downstore]
	else:	
		return [var,success]


[system, compvar, modvar] = controlpanel(shift)
StochMeth(shift,printf,timegap,system,compvar,modvar)


print(datetime.now() - startTime)



