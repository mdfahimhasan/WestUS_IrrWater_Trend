#!/bin/bash

#SBATCH --partition=smi_all
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=Fahim.Hasan@colostate.edu

python run_panel_reg.py

