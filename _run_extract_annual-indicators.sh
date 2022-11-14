#!/bin/bash 
module load conda/2021
source activate mesmer

for subindex_csl in {0..49}
do
    python _extract_annual-indicators.py $subindex_csl &
done




