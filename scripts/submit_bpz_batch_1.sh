#!/bin/bash
for ((i=4;i<8;i++))

do
echo $i

cd /scratch/midway2/raulteixeira/CosmicShearData/
mkdir tile_${i}
cd tile_${i}
cp /home/raulteixeira/repos/CSPZ/scripts/download_tile.py ./.
cp /home/raulteixeira/repos/CSPZ/scripts/bpztilerun_template.py ./.
cp /home/raulteixeira/repos/CSPZ/scripts/tile_DR3_1_1.csv ./.

python download_tile.py ${i}

echo "#!/bin/sh
#SBATCH -t 00:20:00
#SBATCH --partition=chihway
#SBATCH --account=pi-chihway
#SBATCH --job-name=BPZ_${i}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
python bpztilerun_template.py --tilename ${i} --spectra CWWSB4.list --prior hdfn_gen --OutPath /scratch/midway2/raulteixeira/CosmicShearData/tile_${i}/pzout_${i}.h5 --dz 0.010000 --zmin 0.005000 --zmax 3.505000 --Nsamples 1

rm -rf /scratch/midway2/raulteixeira/CosmicShearData/tile_${i}/decade.ncsa.illinois.edu
rm /scratch/midway2/raulteixeira/CosmicShearData/tile_${i}/*py
rm /scratch/midway2/raulteixeira/CosmicShearData/tile_${i}/tile_DR3_1_1.csv
rm /scratch/midway2/raulteixeira/CosmicShearData/tile_${i}/table.h5">submit

sbatch submit

done
