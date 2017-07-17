import itertools
import numpy as np
from scipy.special import expit, entr
from math import ceil

import tempfile
import shutil
import os
from joblib import Parallel, delayed
from joblib import load, dump        

def pattern_maker(N):
    """
    Outputs the arrays holding all the binary patterns
    
    Inputs:
        N - The number of neurons
    Ouputs:
        P_unsigned - The array holding all the {0,1} patterns.  Of shape (2^N,N)
        P_signed - The {-1,1} version
    """
    
    lst = map(list, itertools.product([0, 1], repeat=N))
    P_unsigned = np.array(lst)
    P_signed = 2.0*P_unsigned - 1.0
    return P_unsigned, P_signed

def soft_max(H):
    """
    Carries out the stabilized softmax operation to compute the probability distribution over \vec{r}:
    
    Inputs:
        H - The value of the hamiltonian.  Shape (2^N,Num_samples)
    Ouputs:
        P_r - The distribution over \vec{r}.  Shape (2^N,Num_samples)
    """
    
    max_H = np.amax(H,axis=0)
    #max_H *= 0.0
    mid = np.exp(H-max_H[np.newaxis,:])
    return mid/(np.sum(mid,axis=0)[np.newaxis,:])

def get_H_0(pattern, alpha_vec, beta_vec, J_mat):
    """
    Computes the part of the hamiltonian that is independent of the stimuli
    
    Inputs:
        pattern - The array holding all the signed patterns.  Shape is (2^N,N)
        alpha_vec - The array holding all the alpha values
        beta_vec - The array holding all the beta values
        J_mat - The matrix of couplings. Shape is (N,N)
        
    Ouputs:
        H_0 - The hamiltonian part that is independent of stimuli. Shape is (2^N)
    """
    
    N = np.shape(pattern)[1]
    if np.amax(np.abs(J_mat)) > 0.0:
        J_part = np.sum(np.dot(pattern,J_mat)*pattern,axis=1)
    else:
        J_part = 0.0
    
    alpha_part = -np.sum(pattern*beta_vec[np.newaxis,:]*alpha_vec[np.newaxis,:],axis=1)
    
    
    return J_part + alpha_part

def compute_r_dist(pattern,alpha_vec,beta_vec,J_mat,w_array,stimuli):
    """
    Computes distribution over patterns of r
    
    Inputs:
        pattern - The array holding all the signed patterns.  Shape is (2^N,N)
        alpha_vec - The array holding all the alpha values
        beta_vec - The array holding all the beta values
        J_mat - The matrix of couplings. Shape is (N,N)
        w_array - The set of linear filters.  Shape (N,D)
        stimuli - The set of stimuli.  Shape (D,Num_samples)
    Ouputs:
        r_dist - The distribution over all 2^N patterns
    """
    N = np.size(beta_vec)
    total = 2**N
    H_0 = get_H_0(pattern, alpha_vec, beta_vec, J_mat)
    
    #calculate number of batches
    Num_samples = np.shape(stimuli)[1]
    
    current_proj = np.dot(w_array,stimuli)
    current_proj *= beta_vec[:,np.newaxis]
    current_H = np.einsum('ij,jk',pattern,current_proj)
    current_H += H_0[:,np.newaxis]
    r_dist = soft_max(current_H)
    return r_dist

def tuning_curves(pattern,alpha_vec,beta_vec,J_mat,w_array,stimuli):
    """
    Computes P(r_i = 1|s) for all i as a function of s
    
    Inputs:
        pattern - The array holding all the signed patterns.  Shape is (2^N,N)
        alpha_vec - The array holding all the alpha values
        beta_vec - The array holding all the beta values
        J_mat - The matrix of couplings. Shape is (N,N)
        w_array - The set of linear filters.  Shape (N,D)
        stimuli - The set of stimuli.  Shape (D,Num_samples)
    Ouputs:
        mean_mat - The array holding P(r_i = 1|s).  Shape is (N,num_samples)
    """
    
    #r_dist has shape (2^N,Num_samples)
    r_dist = compute_r_dist(pattern,alpha_vec,beta_vec,J_mat,w_array,stimuli)
    pattern_un = (1.0 + pattern)/2.0
    mean_mat = np.sum(pattern_un[:,:,np.newaxis] * r_dist[:,np.newaxis,:],axis=0)
    return mean_mat

def mini_batch_job(pattern,alpha_vec,beta_vec,H_0,w_array,stimuli,start,stop):
    N = np.size(beta_vec)
    total = 2**N
    
    #calculate number of samples for this job
    Num_samples = stop - start
    
    #allocate memory
    r_dist_current = np.zeros((total,Num_samples))
    r_marg = np.zeros(total)
    r_ce = 0.0
    
    current_proj = np.dot(w_array,stimuli[:,start:stop])
    current_proj *= beta_vec[:,np.newaxis]
    current_H = np.einsum('ij,jk',pattern,current_proj)
    current_H += H_0[:,np.newaxis]
    r_dist_current = soft_max(current_H)
    #update marginal distribution using current batch
    r_marg += np.sum(r_dist_current,axis=1)
    #update conditional entropy using current batch
    r_ce += np.sum(np.sum(entr(r_dist_current),axis=0))
    return (r_marg,r_ce)

def master_thread_I(pattern,alpha_vec,beta_vec,J_mat,w_array,stimuli,batch_size,num_threads=4):
    """
    Computes mutual information in a multithreaded fashion
    
    Inputs:
        pattern - The array holding all the signed patterns.  Shape is (2^N,N)
        alpha_vec - The array holding all the alpha values
        beta_vec - The array holding all the beta values
        J_mat - The matrix of couplings. Shape is (N,N)
        w_array - The set of linear filters.  Shape (N,D)
        stimuli - The set of stimuli.  Shape (D,Num_samples)
        batch_size - The size of each minibatch.  Must be greater than 1 and evenly divide the number of stimuli
        num_threads- The number of threads to use for parallel computation
    Ouputs:
        I_r - The mutual information in nats between \vec{s} and \vec{r}
        
    """
    
    N = np.size(beta_vec)
    total = 2**N
    H_0 = get_H_0(pattern, alpha_vec, beta_vec, J_mat)
    #print H_0
    #calculate number of batches
    Num_samples = np.shape(stimuli)[1]
    Num_batches = int(ceil(float(Num_samples) / float(batch_size)))
    
    #allocate memory
    r_marg = np.zeros(total)
    r_ce = 0.0
    
    
    #start memmapping large stuff
    folder = tempfile.mkdtemp()
    
    pattern_name = os.path.join(folder, 'pattern')
    stimuli_name = os.path.join(folder, 'stimuli')
    H_0_name = os.path.join(folder, 'H_0')
    
    dump(stimuli,stimuli_name)
    dump(pattern,pattern_name)
    dump(H_0,H_0_name)
    
    pattern = load(pattern_name,mmap_mode='r')
    stimuli = load(stimuli_name,mmap_mode='r')
    H_0 = load(H_0_name,mmap_mode='r')
    
    #make arrays of start and stop indices
    ind_array = np.arange(Num_batches)
    start_array = batch_size * ind_array
    stop_array = np.clip(start_array + batch_size,0,Num_samples)
               
    results = Parallel(n_jobs=num_threads)(delayed(mini_batch_job)(pattern,alpha_vec,beta_vec,H_0,w_array,stimuli,start_array[i],stop_array[i]) for i in range(Num_batches))
    
    #for item in results:
        #r_marg += item[0]/float(Num_samples)
        #r_ce += item[1]/float(Num_samples)
    
    margs,ce_vals = zip(*results)
    r_marg = sum(margs)/float(Num_samples)
    r_ce = sum(ce_vals)/float(Num_samples)
    
    r_H = np.sum(entr(r_marg))
    
    try:
        shutil.rmtree(folder)
    except:
        print("Failed to delete: " + folder)
        
    I_r = (r_H - r_ce)
    return I_r

def I_wrapper(alpha_vec,beta_vec,J_mat,w_array,stimuli,batch_size,num_threads=4):
    """
    Wrapper to the compute_I function
    
    Inputs:
        alpha_vec - The array holding all the alpha values
        beta_vec - The array holding all the beta values
        J_mat - The matrix of couplings. Shape is (N,N)
        w_array - The set of linear filters.  Shape (N,D)
        stimuli - The set of stimuli.  Shape (D,Num_samples)
        batch_size - The size of each minibatch.  Must be greater than 1.
        num_threads - The number of threads to use for parallel computation
    Ouputs:
        I_r - The mutual information in nats between \vec{s} and \vec{r}
    """
    
    N = np.size(alpha_vec)
    pattern_un, pattern = pattern_maker(N)
    return master_thread_I(pattern,alpha_vec,beta_vec,J_mat,w_array,stimuli,batch_size,num_threads=4)