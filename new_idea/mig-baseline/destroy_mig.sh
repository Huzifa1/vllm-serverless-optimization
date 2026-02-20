bash kill-gpu-procs.sh

# Distroy all existing MIG instances
sudo nvidia-smi mig -dci && sudo nvidia-smi mig -dgi
# Disable MIG
sudo nvidia-smi -i 0 -mig 0