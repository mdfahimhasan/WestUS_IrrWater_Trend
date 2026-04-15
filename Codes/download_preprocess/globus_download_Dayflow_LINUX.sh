#!/bin/bash

## running the globus_download_LINUX.py script on CPU nodes

#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-0
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=Fahim.Hasan@colostate.edu

# Start Globus Connect Personal in background
~/globusconnectpersonal/globusconnectpersonal -start &
GCP_PID=$!      #  Globus Connect Personal Process ID 
sleep 15

# Make globus CLI available
export PATH="$HOME/.local/bin:$PATH"

# Run the Python transfer script
python globus_download_LINUX.py

# Stop Globus Connect Personal when done
kill $GCP_PID