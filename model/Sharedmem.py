from mpi4py import MPI 
import numpy as np 

comm = MPI.COMM_WORLD 
rank = comm.rank
workers  = comm.Get_size()

def SharedArray(self,shape,dtype=np.float32):
		bytesize = np.dtype(dtype).itemsize
		nbytes = sum(bytesize * dim for dim in shape)
		win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm) 
		buf, itemsize = win.Shared_query(0) 
		return np.ndarray(buffer=buf, dtype=np.float32, shape=shape) 

