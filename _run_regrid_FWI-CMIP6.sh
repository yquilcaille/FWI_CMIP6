#!/bin/bash 
module load conda/2021
source activate mesmer

for subindex_csl in {0..14}
do
    python _regrid_FWI-CMIP6.py $subindex_csl &
done




