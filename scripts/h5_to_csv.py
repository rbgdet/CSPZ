import pandas as pd
import time

start_time = time.time()
for i in range(1,50):
    df = pd.read_hdf(f'/project2/chihway/raulteixeira/data/bdf_photometry/dr3_1_1_bdf_0000{i:02}.h5')
    df.to_csv(f'/project2/chihway/raulteixeira/data/bdf_photometry/dr3_1_1_bdf_0000{i:02}_v2.csv.gz')
    print(i, 'file:', start_time-time.time(), 'seconds')