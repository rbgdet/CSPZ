import numpy as np
import os
import sys

print('Download START')

i = int(sys.argv[1])

metadata = np.genfromtxt('tile_DR3_1_1.csv', dtype='str', delimiter=",")[1:][i]

tile = metadata[0][2:-1]
path = metadata[1][2:-1]
print('this tile will be downloaded: ', tile)
#print(path.split('/')[1:])
Taiga_path = 'DEC_Taiga/'+'/'.join(path.split('/')[1:])

print("metadata=", metadata)
command = 'wget --user=decade --password=decaFil3s  --recursive --no-parent --auth-no-challenge  https://decade.ncsa.illinois.edu/deca_archive/'+Taiga_path+'/cat/'

os.system(command)
print('this tile was be downloaded: ', tile)
print('Download END')
