bash destroy_mig.sh

# Enable MIG
sudo nvidia-smi -i 0 -mig 1

# Create 3 MIG slices with:
# 2x 2g.24gb (PID = 14)
# 1x 3g.47gb (PID = 9)
sudo nvidia-smi mig -cgi 9,14,14 -C