from schwimmbad import MPIPool
import numpy as np

def fun(inputs):
	print(f"Hi! This is job number {inputs}!")
	return 0
	
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

nranks = pool.comm.Get_size() - 1
pool.map(fun, np.arange(nranks))