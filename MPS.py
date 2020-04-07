import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
import scipy.linalg as lin
import scipy.io as spio

#convention for MPS objects: MPS[sites][physical,bond,bond])

#To do:
#Right canonicalise for the entanglement entropy

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


def Canonicalise_Normed(inputMPS):

	
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
		schmidt_values.append(S)
		if kk == 0:
			outputMPS.append(np.reshape(U,(np.size(inputMPS[kk],0),np.size(inputMPS[kk],1),np.size(S,0))))   #sigma,d1 and the new bond dimension
			#print('output',outputMPS[0][1,:,:])
			#print('sch',S)
		else:
			outputMPS.append(np.reshape(U,(np.size(tempMPS,0),np.size(tempMPS,1),np.size(S,0))))   #sigma,d1 and the new bond dimension
			del(tempMPS)
		tempMPS = np.zeros([physical,np.size(S),np.size(inputMPS[kk+1],2)],dtype = complex)

		tempMPS[0,:,:] = np.dot(np.dot(np.diag(S),Vh),inputMPS[kk+1][0,:,:])  #MA becomes the new A
		tempMPS[1,:,:] = np.dot(np.dot(np.diag(S),Vh),inputMPS[kk+1][1,:,:])  #MA becomes the new A

	#Last site
	del(Ashaped,U,S,Vh)
	Ashaped = tempMPS.reshape((np.size(tempMPS,0)*np.size(tempMPS,1),np.size(tempMPS,2)))
	U,S,Vh = lin.svd(Ashaped,full_matrices = False)
	outputMPS.append(np.reshape(U,(np.size(tempMPS,0),np.size(tempMPS,1),np.size(S,0))))   #sigma,d1 and the new bond dimension
	norm = S*Vh
	#outputMPS[sites-1][:,:,0] = (norm)*outputMPS[sites-1][:,:,0]
	#Commenting the above line prevents normalisation	
	#Note that the last site A = U S Vh was defined as U with S Vh ignored since they are just the norm 


	
	return [outputMPS,schmidt_values,norm]

def Canonicalise_NoTruncation(inputMPS):

	
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

		schmidt_values.append(S)
		if kk == 0:
			outputMPS.append(np.reshape(U,(np.size(inputMPS[kk],0),np.size(inputMPS[kk],1),np.size(S,0))))   #sigma,d1 and the new bond dimension
			#print('output',outputMPS[0][1,:,:])
			#print('sch',S)
		else:
			outputMPS.append(np.reshape(U,(np.size(tempMPS,0),np.size(tempMPS,1),np.size(S,0))))   #sigma,d1 and the new bond dimension
			del(tempMPS)
		tempMPS = np.zeros([physical,np.size(S),np.size(inputMPS[kk+1],2)],dtype = complex)

		tempMPS[0,:,:] = np.dot(np.dot(np.diag(S),Vh),inputMPS[kk+1][0,:,:])  #MA becomes the new A
		tempMPS[1,:,:] = np.dot(np.dot(np.diag(S),Vh),inputMPS[kk+1][1,:,:])  #MA becomes the new A

	#Last site
	del(Ashaped,U,S,Vh)
	Ashaped = tempMPS.reshape((np.size(tempMPS,0)*np.size(tempMPS,1),np.size(tempMPS,2)))
	U,S,Vh = lin.svd(Ashaped,full_matrices = False)
	outputMPS.append(np.reshape(U,(np.size(tempMPS,0),np.size(tempMPS,1),np.size(S,0))))   #sigma,d1 and the new bond dimension
	norm = S*Vh
	outputMPS[sites-1][:,:,0] = (norm)*outputMPS[sites-1][:,:,0]	
	#Note that the last site A = U S Vh was defined as U with S Vh ignored since they are just the norm 


	
	return [outputMPS,schmidt_values,norm]


def Canonicalise(inputMPS,bond_dim,normalise_output):
	
	
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
		tempMPS = np.zeros([physical,np.size(S),np.size(inputMPS[kk+1],2)],dtype = complex)

		tempMPS[0,:,:] = np.dot(np.dot(np.diag(S),Vh),inputMPS[kk+1][0,:,:])  #MA becomes the new A
		tempMPS[1,:,:] = np.dot(np.dot(np.diag(S),Vh),inputMPS[kk+1][1,:,:])  #MA becomes the new A

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

def Canonicalise_Parallel(inputMPS,bond_dim,normalise_output):
	
	
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
		tempMPS = np.zeros([physical,np.size(S),np.size(inputMPS[kk+1],2)],dtype = complex)

		tempMPS[0,:,:] = np.dot(np.dot(np.diag(S),Vh),inputMPS[kk+1][0,:,:])  #MA becomes the new A
		tempMPS[1,:,:] = np.dot(np.dot(np.diag(S),Vh),inputMPS[kk+1][1,:,:])  #MA becomes the new A

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


def Right_Canonicalise(inputMPS):

	
	sites = len(inputMPS)
	physical = np.size(inputMPS[0][:,0,0])
	#Canonicalises MPS in left canonical form, normalised but also outputs the norm 
	outputMPS = []
	schmidt_values = []

	
	for kk in range(sites-1,0,-1): #all sites except last dealt with in loop
		if kk == sites-1:	
			Ashaped = inputMPS[kk].reshape(np.size(inputMPS[kk],1),np.size(inputMPS[kk],2)*(np.size(inputMPS[kk],0)))
			#print('A',Ashaped)	
		else:
			del(Ashaped,U,S,Vh) 
			Ashaped = tempMPS.reshape(np.size(tempMPS,1),np.size(tempMPS,2)*np.size(tempMPS,0))
		print(kk)	
		U,S,Vh = lin.svd(Ashaped,full_matrices = False)
		schmidt_values = [S] + schmidt_values 
		if kk == sites-1:
			outputMPS.append(np.reshape(Vh,(np.size(inputMPS[kk],0),np.size(S,0),np.size(inputMPS[kk],2))))   #sigma,d1 and the new bond dimension
			#print('output',outputMPS[0][1,:,:])
			#print('sch',S)
		else:
			outputMPS = [np.reshape(Vh,(np.size(tempMPS,0),np.size(S,0),np.size(tempMPS,2)))] + outputMPS   #sigma,d1 and the new bond dimension
			del(tempMPS)
		tempMPS = np.zeros([physical,np.size(inputMPS[kk-1],1),np.size(S)],dtype = complex)

		tempMPS[0,:,:] = np.dot(inputMPS[kk-1][0,:,:],np.dot(U,np.diag(S)))  #MA becomes the new A
		tempMPS[1,:,:] = np.dot(inputMPS[kk-1][1,:,:],np.dot(U,np.diag(S)))  #MA becomes the new A

	#Last site
	del(Ashaped,U,S,Vh)
	Ashaped = tempMPS.reshape(np.size(tempMPS,1),np.size(tempMPS,2)*np.size(tempMPS,0))
	U,S,Vh = lin.svd(Ashaped,full_matrices = False)
	outputMPS = [(np.reshape(Vh,(np.size(tempMPS,0),np.size(S,0),np.size(tempMPS,2))))] + outputMPS   #sigma,d1 and the new bond dimension
	norm = U*S
	outputMPS[0][:,:,0] = (norm)*outputMPS[0][:,:,0]	
	#Note that the last site A = U S Vh was defined as U with S Vh ignored since they are just the norm 


	
	return [outputMPS,schmidt_values,norm]

def Data_to_MPS(data,sites,phys_dim,samples):
	MPS  = []
	for sz in range(0,sites):
		#print(sz,'sz')
		if sz == 0: #first site
			Amat = np.zeros([phys_dim,1,samples],dtype = complex) #For a single time, and redefined in loop every site
		
		elif sz == sites-1:  #last site
			Amat = np.zeros([phys_dim,samples,1],dtype = complex) #For a single time, and redefined in loop every site

		else:
			Amat = np.zeros([phys_dim,samples,samples],dtype = complex) #For a single time, and redefined in loop every site

		for ii in range(0,samples):
			#print(ii,'samples')
			#print(Amat[:,0,ii],'samples')
			if sz == 0: #first site
				Amat[:,0,ii] = (1.0/samples)*data[ii,sz,:] #do both physical dims at same time 
				
			elif sz == sites-1: #last site
				Amat[:,ii,0] = data[ii,sz,:] #do both physical dims at same time 

			else: 
				Amat[:,ii,ii] = data[ii,sz,:] #do both physical dims at same time 
		MPS.append(Amat)
	return MPS


def MPS_Merge_Pair(MPS_1,MPS_2):
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
				new_mat[phys,0,0:np.size(MPS_1[tt],2)] = (1.0/2.0)*MPS_1[tt][phys,:,:]
				new_mat[phys,0,np.size(MPS_1[tt],2):np.size(new_mat,2)] = (1.0/2.0)*MPS_2[tt][phys,:,:]
			
			elif tt == sites-1: #Note that this entry is considered 1D for some reason. Perhaps because new mat is dimension 2,X,1 and the 1 is on the end so is ignored?
				new_mat[phys,0:np.size(MPS_1[tt],1)] = MPS_1[tt][phys,:,:]
				new_mat[phys,np.size(MPS_1[tt],1):np.size(new_mat,1)] = MPS_2[tt][phys,:,:]

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

def MPS_Merge_Both_Normalise_Fork(MPS_1,MPS_2,batches):  #SIZE EDIT
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
				new_mat[phys,0,0:np.size(MPS_1[tt],2)] = (1.0/float(batches))*MPS_1[tt][phys,:,:]  #MPS2 is added to MPS1 not the other way
				new_mat[phys,0,np.size(MPS_1[tt],2):np.size(new_mat,2)] = (1.0/float(batches))*MPS_2[tt][phys,:,:]
			
			elif tt == sites-1: #Note that this entry is considered 1D for some reason. Perhaps because new mat is dimension 2,X,1 and the 1 is on the end so is ignored?
				new_mat[phys,0:np.size(MPS_1[tt],1)] = MPS_1[tt][phys,:,:]
				new_mat[phys,np.size(MPS_1[tt],1):np.size(new_mat,1)] = MPS_2[tt][phys,:,:]

			else:
				new_mat[phys,0:np.size(MPS_1[tt],1),0:np.size(MPS_1[tt],2)] = MPS_1[tt][phys,:,:]
				new_mat[phys,np.size(MPS_1[tt],1):,np.size(MPS_1[tt],2):] = MPS_2[tt][phys,:,:]
		MPS_N.append(new_mat)

	return MPS_N

def compressor_svd_running(data,sites,phys_dim,batch_size,batches):  #Compression before addition

	for pp in range(0,batches-1):
		if pp == 0:
			MPS_batch1 = Data_to_MPS(data[0*batch_size:(1)*batch_size,:,:,time],sites,phys_dim,batch_size)
			#MPS_batch1,s,n = Canonicalise(MPS_batch1)

			MPS_batch2 = Data_to_MPS(data[(1)*batch_size:(2)*batch_size,:,:,time],sites,phys_dim,batch_size)
			#MPS_batch2,s,n = Canonicalise(MPS_batch2)

			MPS_running = MPS_Merge_Both_Normalise(MPS_batch1,MPS_batch2,batches)	
			MPS_running,s,n = Canonicalise(MPS_running)	
		else:
			MPS_batch = Data_to_MPS(data[(pp+1)*batch_size:(pp+2)*batch_size,:,:,time],sites,phys_dim,batch_size)
			#MPS_batch,s,n = Canonicalise(MPS_batch)
			MPS_running = MPS_Merge_Single_Normalise(MPS_running,MPS_batch,batches)	
			MPS_running,s,n = Canonicalise(MPS_running) 
			del(MPS_batch)

	#MPS_running,s,n = Canonicalise_Normed(MPS_running)  #At the last stage the norm is taken
	return MPS_running,s,n

def compressor_svd_static(data,sites,phys_dim,batch_size,batches,bond_dim):  #All samples are added together and then compressed

	for pp in range(0,batches):
		if pp == 0:
			MPS_batch1 = Data_to_MPS(data[0*batch_size:(1)*batch_size,:,:,time],sites,phys_dim,batch_size)
			#print(MPS_batch1[0][0,:,:],'In')	
			MPS_batch1,s,n = Canonicalise(MPS_batch1,bond_dim,0)
			#print(MPS_batch1[0][0,:,:],'1')
			#print(np.shape(MPS_batch1[1]))	
			#print(MPS_batch1[0][0,:,:],'running')
		elif pp == 1:
			MPS_batch2 = Data_to_MPS(data[1*batch_size:(2)*batch_size,:,:,time],sites,phys_dim,batch_size)
			MPS_batch2,s,n = Canonicalise(MPS_batch2,bond_dim,0)
			#print(np.shape(MPS_batch2[1]))	
			#print(MPS_batch2[0][0,:,:],'running')
			#print(MPS_batch1[0][0][0,:,:],'1')	
			#print(MPS_batch2[0][0,:,:],'2')	
			MPS_running = MPS_Merge_Both_Normalise(MPS_batch1,MPS_batch2,batches)	
			#print(np.shape(MPS_running[1]))	
			#print(MPS_running[0][0,:,:],'running')
		else:
			#print(np.size(MPS_running[0]))	
			MPS_batch = Data_to_MPS(data[(pp)*batch_size:(pp+1)*batch_size,:,:,time],sites,phys_dim,batch_size)
			MPS_batch,s,n = Canonicalise(MPS_batch,bond_dim,0)
			MPS_running = MPS_Merge_Single_Normalise(MPS_running,MPS_batch,batches)	
			del(MPS_batch)
	
	MPS_fin,s,n = Canonicalise(MPS_running,bond_dim,1) 
	return MPS_fin,s,n

def compressor_post_parallel(MPS_list,bond_dim,batches):  #All samples are added together and then compressed

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
	
	MPS_fin,s,n = Canonicalise(MPS_running,bond_dim,1) 
	return MPS_fin,s,n

def compressor_post_parallel_normop(MPS_list,bond_dim,batches,output_norm):  #All samples are added together and then compressed

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



# How to pararrelise object. Must Canonicalise all and save. The loop is done separately.
# Use pararrelise with basic Canonicalisation. Then sweep through, adding iteratively, which can be done in a loop if feed in all the canonicalised MPS.
 


##def Data_to_MPS(dataMAT):
Sx = 0.5*np.asarray([[0,1.0],[1.0,                                              0]])                                                           
Sy = 0.5*np.asarray([[0,-1.0j],[1.0j,                                           0]])                                                     
Sz = 0.5*np.asarray([[1.0,0],[0,-1.0]])       

#def Magnetisation_Test(MPS):
#
#    n = 0
#    for ii in range(0,len(MPS)):
#        if ii == len(MPS)-1:  #evaluate magnetisation on RH site so just use the fact that left-canonical
#            end_chain = Sz[0,0]*np.dot(MPS[6][0,:,:],np.conj(np.transpose(MPS[6][0,:,:]))) + Sz[1,1]*np.dot(MPS[6][1,:,:],np.conj(np.transpose(MPS[6][1,:,:]))) + Sz[0,1]*np.dot(MPS[6][0,:,:],np.conj(np.transpose(MPS[6][1,:,:]))) + Sz[1,0]*np.dot(MPS[6][1,:,:],np.conj(np.transpose(MPS[6][0,:,:])))
#            n = np.trace(end_chain)
#            print(1.0/float(sites)*np.trace(end_chain),'end')
#
#       else: #Otherwise, scan from the right until the site of interest (left site)
#            left_site = Sz[0,0]*np.dot(MPS[ii][0,:,:],np.conj(np.transpose(MPS[ii][0,:,:]))) + Sz[1,1]*np.dot(MPS[ii][1,:,:],np.conj(np.transpose(MPS[ii][1,:,:]))) + Sz[0,1]*np.dot(MPS[ii][0,:,:],np.conj(np.transpose(MPS[ii][1,:,:]))) + Sz[1,0]*np.dot(MPS[ii][1,:,:],np.conj(np.transpose(MPS[ii][0,:,:]))) 
#            print('check')
#            print(np.shape(left_site),'left size')
#            for jj in range(len(MPS)-1,ii,-1):
#                print(jj,'jj')
#                if jj == len(MPS)-1:  #starting at the end
#                    right_site = np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:]) + np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][0,:,:]) + np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][1,:,:])
#                else:  #take product with the next site. Check order of contractions
#                    right_site = (np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:]) + np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][0,:,:]) + np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][1,:,:]))*right_site
#                print(np.shape(right_site),'right')
#            tadpole = np.dot(left_site,right_site)
#            n = n + (1.0/len(MPS))*tadpole 
#    return n 

#def Magnetisation(MPS):  #Assumes Left Canonical Form. See Scholwock page 36 for refresh on the contraction order.

#    n = 0
#    for ii in range(0,len(MPS)):
#        if ii == len(MPS)-1:  #evaluate magnetisation on RH site so just use the fact that left-canonical. #The end of the chain is unique. Can use AxA^*
#            end_chain = Sz[0,0]*np.dot(np.conj(np.transpose(MPS[len(MPS)-1][0,:,:])),MPS[len(MPS)-1][0,:,:]) + Sz[1,1]*np.dot(np.conj(np.transpose(MPS[len(MPS)-1][1,:,:])),MPS[len(MPS)-1][1,:,:]) + Sz[0,1]*np.dot(np.conj(np.transpose(MPS[len(MPS)-1][0,:,:])),MPS[len(MPS)-1][1,:,:]) + Sz[1,0]*np.dot(np.conj(np.transpose(MPS[len(MPS)-1][1,:,:])),MPS[len(MPS)-1][0,:,:]) 
#	    n =  n + 1.0/len(MPS)*end_chain
	    #print(1.0/len(MPS)*end_chain,'end')
		
#	else: #Otherwise, scan from the right until the site of interest (left site)
#            left_site = Sz[0,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + Sz[1,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:]) + Sz[0,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][0,:,:]) + Sz[1,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][1,:,:]) 
#            for jj in range(len(MPS)-1,ii,-1):
#                if jj == len(MPS)-1:  #starting at the end. Do the physical contraction.
#                   right_site = np.dot(MPS[jj][0,:,:],np.conj(np.transpose(MPS[jj][0,:,:]))) + np.dot(MPS[jj][1,:,:],np.conj(np.transpose(MPS[jj][1,:,:])))
#		    #print(np.shape(right_site),'right')
#                else:  #Triangle Contraction,
#		    new_right_up = np.dot(np.dot(MPS[jj][0,:,:],right_site),np.conj(np.transpose(MPS[jj][0,:,:])))  #Bond contraction for up index
#		    new_right_down = np.dot(np.dot(MPS[jj][1,:,:],right_site),np.conj(np.transpose(MPS[jj][1,:,:]))) #Bond contraction for down index
#		    right_site = new_right_up + new_right_down   #Physical contraction
#            tadpole = np.dot(left_site,right_site)
#            n = n + (1.0/len(MPS))*np.trace(tadpole) 

    #	print(np.shape(tadpole),'tadpole')
#    return n 


def Magnetisation_LEFTZIP(MPS):  #Assumes Left Canonical Form. See Scholwock page 36 for refresh on the contraction order.
	n = 0
	for ii in range(0,len(MPS)):
		if ii == len(MPS)-1:  #evaluate magnetisation on RH site so just use the fact that left-canonical. #The end of the chain is unique. Can use AxA^*
			end_chain = Sz[0,0]*np.dot(np.conj(np.transpose(MPS[len(MPS)-1][0,:,:])),MPS[len(MPS)-1][0,:,:]) + Sz[1,1]*np.dot(np.conj(np.transpose(MPS[len(MPS)-1][1,:,:])),MPS[len(MPS)-1][1,:,:]) + Sz[0,1]*np.dot(np.conj(np.transpose(MPS[len(MPS)-1][0,:,:])),MPS[len(MPS)-1][1,:,:]) + Sz[1,0]*np.dot(np.conj(np.transpose(MPS[len(MPS)-1][1,:,:])),MPS[len(MPS)-1][0,:,:]) 
			n =  n + 1.0/len(MPS)*end_chain	
		else: #Otherwise, scan from the right until the site of interest (left site)
			left_site = Sz[0,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][0,:,:]) + Sz[1,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][1,:,:]) + Sz[0,1]*np.dot(np.conj(np.transpose(MPS[ii][1,:,:])),MPS[ii][0,:,:]) + Sz[1,0]*np.dot(np.conj(np.transpose(MPS[ii][0,:,:])),MPS[ii][1,:,:])
 
			for jj in range(ii+1,len(MPS)):
				new_left_up = np.dot(np.conj(np.transpose(MPS[jj][0,:,:])),np.dot(left_site,MPS[jj][0,:,:]))
				new_left_down = np.dot(np.conj(np.transpose(MPS[jj][1,:,:])),np.dot(left_site,MPS[jj][1,:,:])) #Bond contraction for down index
				left_site = new_left_up + new_left_down   #Physical contraction
            			#tadpole = np.dot(left_site,right_site)

			n = n + (1.0/len(MPS))*left_site #np.trace(tadpole) 
	return n 

def Unravel(MPS):
	for ii in range(0,len(MPS)-1):
		if ii == 0:
			c_mat = np.dot(MPS[ii][:,:,:],MPS[ii+1][:,:,:])
		else:
			c_mat = np.dot(c_mat,MPS[ii+1][:,:,:])
	
	return cmat



#from joblib import Parallel, delayed
#import multiprocessing
#num_cores =  4
#time = 0
#samples = 5000
#phys_dim = 2
#sites = 7
#batches = 5
#bond_dim = 10
#(data,sites,phys_dim,samples)
#batch_size = samples/batches
#data1 = np.load('/home/k1623105/my_local_scratch/Storage_MPS/TestBatch/prodstate1_half_0.npy')
#MPS_List = Parallel(n_jobs = num_cores)(delayed(Data_to_MPS)(data1[vv*batch_size:(vv+1)*
#batch_size,:,:,time],sites,phys_dim,batch_size) for vv in range(0,batches))
#results = Parallel(n_jobs=num_cores)(delayed(Canonicalise_Parallel)(MPS_List[vv],bond_dim,0) for #vv in range(0,batches)#)
#MPS_List_C = results

#print(np.shape(MPS_List_C[0][0]),'shape')
#MPS_Total,s,norm = compressor_post_parallel(MPS_List_C,batches) 
#print(len(MPS_List))
#print(len(MPS_List_C))
#print(len(MPS_Total))
#print(np.shape(MPS_Total[0]),'shape')

#Data_to_MPS(data1[vv*batch_size:(vv+1)*batch_size,:,:,time],sites,phys_dim,batch_size)
#number = 30Canonicalise)
#biner = np.binary_repr(number)
#print(str(biner))
#bin_str = [int(d) for d in str(biner)]
#print(bin_str)

#print(int(d) for d in str(np.binary_repr(number)))
#data = np.load('/home/k1623105/my_local_scratch/Storage_MPS/TestBatch/prodstate1_half_0.npy')
#data1 = np.load('/home/k1623105/my_local_scratch/Storage_MPS/TestBatch/prodstate1_half_0.npy')
#data2 = np.load('/home/k1623105/my_local_scratch/Storage_MPS/TestBatch/prodstate1_half_1.npy')
#print(np.shape(data1))  #samp,sites,phys_dim,time

#phys_dim = 2
#time = 0
#samples = 5000
#sites = 7
#batches = 1
#bond_dim = 10

#batch_size = samples/batches
#Divide all elements of 1 matrix by samples to normalise

#MPS = random_state(2,7,100)
#MPS_C,sch,Norm = Canonicalise(MPS,bond_dim,1)
#Unravel(MPS_C)

#MPS = Data_to_MPS(data1[0:samples,:,:,time],sites,phys_dim,samples)
#print(Norm,'left')
#print(sch[0],'MPSC')



#mag = Magnetisation_LEFTZIP(MPS_C)
#print(mag,'mag')
#MPS_D,s,N = Right_Canonicalise(MPS)
#print(N,'right')
#print(s[5],'MPSD')

#print(np.dot(np.conj(np.transpose(MPS_C[1][0,:,:])),MPS_C[1][0,:,:]) + np.dot(np.conj(np.transpose(MPS_C[1][1,:,:])),MPS_C[1][1,:,:]),'check')
#print(Norm)
#print(mag,'magnetisation')
#print(sch[0])
#/np.sqrt(np.dot(sch[0],sch[0])), 'first sch')
#print(Norm)
#Magnet = np.zeros(20,dtype = complex)
#for time in range(0,20):
#        print(time) #MPS = Data_to_MPS(data[0:samples,:,:,time],sites,phys_dim,samples)
#	MPS_C,sch,Norm = Canonicalise(MPS)
#	Magnet[time] = Magnetisation(MPS_C)
#	del(MPS,MPS_C,sch,Norm)
#print(MPS_C[1][0,:,:],'MPSC')
#print(np.dot(np.conj(np.transpose(MPS_C[1][0,:,:])),MPS_C[1][0,:,:]) + np.dot(np.conj(np.transpose(MPS_C[1][1,:,:])),MPS_C[1][1,:,:]),'check')
#print(Norm)
#print(Magnet,'mag')

#2nd SVD appears to implement a gauge transformation....
#MPS_D, sch2,n2 = Canonicalise(MPS_C)
#print(MPS_D[1][0,:,:],'MPSD')
#print(sch2[0])  #Should be all 1's
#/np.sqrt(np.dot(sch2[0],sch2[0])), 'first sch')
##print(MPS_D[0][1,:,:],'MPSD')
#print(n2)

#MPS_E, sch2,n2 = Canonicalise(MPS_D)
#print(MPS_E[1][0,:,:],'MPSE')
#print(sch2[0])  #Should be all 1's
#/np.sqrt(np.dot(sch2[0],sch2[0])), 'first sch')
##print(MPS_D[0][1,:,:],'MPSD')
#print(n2)
#print(np.dot(np.transpose(MPS_D[1][:,0,0]),MPS_D[1][:,0,0]),'check')
#print(np.dot(np.conj(np.transpose(MPS_D[1][0,:,:])),MPS_D[1][0,:,:]) + np.dot(np.conj(np.transpose(MPS_D[1][1,:,:])),MPS_D[1][1,:,:]),'check')
#Check an observable as may be the same up to Gauge. Norm being the same suggests no additional compression.




#print(MPS_C[0],'first elemend D')
#print(MPS_D[5][0,0:10,0],'last elemend D')
#print(sch2[1]/np.sqrt(np.dot(sch2[1],sch2[1])),'follow')
#print(n, 'follow')
#MPS_D,sch1,Norm = Canonicalise(MPS_C)

#print(sch1[2]/np.sqrt(np.dot(sch1[2],sch1[2])),'2nd sch')
#print(Norm,'second')

#MPS_New = MPS_Merge_Both_Normalise(MPS_1,MPS_2,2)
#MPS_New_C,sch,n = Canonicalise(MPS_New)



#print(n)
test_lab = 0
if test_lab == 1:
	#data1 = np.load('/home/k1623105/my_local_scratch/Storage_MPS/TestBatch/prodstate1_half_0.npy')
	#data2 = np.load('/home/k1623105/my_local_scratch/Storage_MPS/TestBatch/prodstate1_half_1.npy')
	#print(np.shape(data1))  #samp,sites,phys_dim,time
	phys_dim = 2
	#time = 10
	samples = 5000
	sites = 7
	batches = 5
	bond_dim = 10
	batch_size = (samples/batches)


	Magnet1 = np.zeros(40,dtype = complex)
	Norm1 = np.zeros(40,dtype = complex)
	for time in range(0,20):
		print(time)
		MPS = Data_to_MPS(data1[0:samples,:,:,time],sites,phys_dim,samples)
		MPS_final,sch,norm = Canonicalise(MPS,bond_dim,1)
		#MPS_final, s, norm = compressor_svd_static(data1,sites,phys_dim,batch_size,batches,bond_dim)
		Norm1[time] = norm
		Magnet1[time] = Magnetisation_LEFTZIP(MPS_final)
		del(MPS_final,sch,norm)
						   # (data,sites,phys_dim,batch_size,batches)
	for time in range(0,20):
		MPS = Data_to_MPS(data2[0:samples,:,:,time],sites,phys_dim,samples)
		MPS_final,sch,norm = Canonicalise(MPS,bond_dim,1)
		print(time)
		#MPS_final, s, norm = compressor_svd_static(data2,sites,phys_dim,batch_size,batches,bond_dim)
		Norm1[20 +  time] = norm

		Magnet1[20 + time] = Magnetisation_LEFTZIP(MPS_final)
		del(MPS_final,sch,norm)
	bond_dim = 128
	Magnet2 = np.zeros(40,dtype = complex)
	Norm2 = np.zeros(40,dtype = complex)
	for time in range(0,20):
		print(time)
		MPS = Data_to_MPS(data1[0:samples,:,:,time],sites,phys_dim,samples)
		MPS_final,sch,norm = Canonicalise(MPS,bond_dim,1)
		#MPS_final, s, norm = compressor_svd_static(data1,sites,phys_dim,batch_size,batches,bond_dim)
		Norm2[time] = norm
		Magnet2[time] = Magnetisation_LEFTZIP(MPS_final)
		del(MPS_final,sch,norm)
						   # (data,sites,phys_dim,batch_size,batches)
	for time in range(0,20):
		MPS = Data_to_MPS(data2[0:samples,:,:,time],sites,phys_dim,samples)
		MPS_final,sch,norm = Canonicalise(MPS,bond_dim,1)
		print(time)
		#MPS_final, s, norm = compressor_svd_static(data2,sites,phys_dim,batch_size,batches,bond_dim)
		Norm2[20 +  time] = norm

		Magnet2[20 + time] = Magnetisation_LEFTZIP(MPS_final)
		del(MPS_final,sch,norm)

	batches = 5
	batch_size = (samples/batches)
	bond_dim = 20
	Magnet3 = np.zeros(40,dtype = complex)
	Norm3 = np.zeros(40,dtype = complex)
	for time in range(0,20):
		print(time)
		MPS = Data_to_MPS(data1[0:samples,:,:,time],sites,phys_dim,samples)
		#MPS_final,sch,norm = Canonicalise(MPS,bond_dim,1)
		MPS_final, sch, norm = compressor_svd_static(data1,sites,phys_dim,batch_size,batches,bond_dim)
		Norm3[time] = norm
		Magnet3[time] = Magnetisation_LEFTZIP(MPS_final)
		del(MPS_final,sch,norm)
						   # (data,sites,phys_dim,batch_size,batches)
	for time in range(0,20):
		MPS = Data_to_MPS(data2[0:samples,:,:,time],sites,phys_dim,samples)
		#MPS_final,sch,norm = Canonicalise(MPS,bond_dim,1)
		print(time)
		MPS_final, sch, norm = compressor_svd_static(data2,sites,phys_dim,batch_size,batches,bond_dim)
		Norm3[20 +  time] = norm

		Magnet3[20 + time] = Magnetisation_LEFTZIP(MPS_final)
		del(MPS_final,sch,norm)
	bond_dim = 20
	Magnet4 = np.zeros(40,dtype = complex)
	Norm4 = np.zeros(40,dtype = complex)
	for time in range(0,20):
		print(time)
		MPS = Data_to_MPS(data1[0:samples,:,:,time],sites,phys_dim,samples)
		MPS_final,sch,norm = Canonicalise(MPS,bond_dim,1)
		#MPS_final, s, norm = compressor_svd_static(data1,sites,phys_dim,batch_size,batches,bond_dim)
		Norm4[time] = norm
		Magnet4[time] = Magnetisation_LEFTZIP(MPS_final)
		del(MPS_final,sch,norm)
						   # (data,sites,phys_dim,batch_size,batches)
	for time in range(0,20):
		MPS = Data_to_MPS(data2[0:samples,:,:,time],sites,phys_dim,samples)
		MPS_final,sch,norm = Canonicalise(MPS,bond_dim,1)
		print(time)
		#MPS_final, s, norm = compressor_svd_static(data2,sites,phys_dim,batch_size,batches,bond_dim)
		Norm4[20 +  time] = norm

		Magnet4[20 + time] = Magnetisation_LEFTZIP(MPS_final)
		del(MPS_final,sch,norm)
	#print(MPS_final[0][0,0:10,:],'batch')
	#Data_to_MPS(data[0*batch_size:(1)*batch_size,:,:,time],sites,phys_dim,batch_size)
	#MPS = Data_to_MPS(data[0:2*samples,:,:,time],sites,phys_dim,2*samples)


	plt.plot(Magnet4,label = '20')
	plt.plot(Magnet3,label = '20batch')
	plt.plot(Magnet1,label ='10')
	plt.plot(Magnet2,label = '128')
	plt.legend()
	#plt.savefig('bond_dimension_plot_directcomp.png')
	plt.show()
	plt.plot(Norm4,label = '128')
	plt.plot(Norm3,label = '30')
	plt.plot(Norm1,label = '10')
	plt.plot(Norm2,label ='128')
	plt.legend()
	plt.show()

#print(Magnet,'mag')
#print(Magnet-Magnet1,'diff')
#Merge Pair divides by 2. 
#Merge Batch should divide by batches.
#batch = samples/batches


#MPS_final, s, n = compressor(data,sites,phys_dim,batch_size)


#print(n)

#print(s[2])



#WRITE A FUNCTION TO DO THIS AS A TEST OF THE BOND DIMENSION
#print(np.shape(MPS_running[0]))
#print(np.shape(MPS_running[2]))
#print(np.shape(MPS_running[3]))
#print(np.shape(MPS_running[4]))
#print(np.shape(MPS_running[5]))
#print(np.shape(MPS_running[6]))



#Need batch normalisation.
#During initial import into an MPS it is divided by total number of samples (data to mps)
#perhaps better to add a new MPS to the old MPS, rather than in pairs.      


# Now say samples are split up into 



#MPS_1 = Data_to_MPS(data[0:2],:,:,time],sites,phys_dim,2)
#MPS_1 = Data_to_MPS(data,sites,phys_dim,2)
#MPS_2 = Data_to_MPS(data,sites,phys_dim,10)

#MPS_New = MPS_Merge(MPS_1,MPS_2)
#print(MPS_New[0][0,:,:])










