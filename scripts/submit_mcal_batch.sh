#!/bin/bash

for ((i=0;i<1;i++))

do
echo $i

cd /scratch/midway2/raulteixeira/CosmicShearData/
mkdir tile_${i}
cd tile_${i}
cp /home/raulteixeira/repos/CSPZ/scripts/download_tile.py ./.
cp /home/raulteixeira/repos/CSPZ/scripts/bpzh5fitsrun.py ./. #
cp /home/raulteixeira/repos/shearcat/code/measurement/tile_DR3_1_1.csv ./.

python download_tile.py ${i}

echo "#!/bin/sh
#SBATCH -t 10:00:00
#SBATCH --partition=broadwl
#SBATCH --account=pi-chihway
#SBATCH --job-name=BPZ_${i}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28

python bpzh5fitsrun.py ${i}

mv metacal_output_*fits /project2/chihway/data/decade/shearcat_v1/.
rm -rf /scratch/midway2/raulteixeira/CosmicShearData/tile_${i}/decade.ncsa.illinois.edu
rm /scratch/midway2/raulteixeira/CosmicShearData/tile_${i}/*py
rm /scratch/midway2/raulteixiera/CosmicShearData/tile_${i}/tile_DR3_1_1.csv
rm /scratch/midway2/raulteixeira/CosmicShearData/tile_${i}/table.h5

">submit

sbatch submit

done

