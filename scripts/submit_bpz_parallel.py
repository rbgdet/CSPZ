import numpy as np
import subprocess as sp
import copy
batch_size = 3
n_batches = 2

f = open("/home/raulteixeira/repos/CSPZ/scripts/submit_bpz_batch.sh", "r")
baselines = f.readlines()

for i in range(n_batches):
	start = i*batch_size
	end = (i+1)*batch_size
	print('python script START: ', start, '\n', 'python script END: ', end)
	list_lines = copy.deepcopy(baselines)
	list_lines[2] = list_lines[2].replace('i=0;i<1', f'i={start}'+f';i<{end}')
	print(list_lines[2])
	#sp.run(f'cp /home/raulteixeira/repos/CSPZ/scripts/submit_bpz_batch.sh /home/raulteixeira/repos/CSPZ/scripts/submit_bpz_batch_{i}.sh', shell = True)
	
	script = open(f'/home/raulteixeira/repos/CSPZ/scripts/submit_bpz_batch_{i}.sh', mode='w')
	script.writelines(list_lines) 
	script.close()
	sp.run(f'bash /home/raulteixeira/repos/CSPZ/scripts/submit_bpz_batch_{i}.sh', shell = True)
	
	#sp.run(f'rm /home/raulteixeira/repos/CSPZ/scripts/submit_bpz_batch_{i}.sh', shell = True)
