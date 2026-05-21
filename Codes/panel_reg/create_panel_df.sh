#!/bin/bash

#SBATCH --partition=all
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=Fahim.Hasan@colostate.edu

python create_panel_df.py

