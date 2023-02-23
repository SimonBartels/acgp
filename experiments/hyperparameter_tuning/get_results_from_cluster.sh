#!/bin/bash
#sshpass -f "passwordfile" ssh cluster 'bash -c "cd ~/pac_determinant/sublinear_gpr/sublinear_gpr_code/experiments/output/results && tar --exclude=parameters* -czf ~/opt_results.tar.gz ."' &&
sshpass -f "passwordfile" scp cluster:opt_results.tar.gz . &&
#sshpass -f "passwordfile" ssh cluster 'bash -c "rm opt_results.tar.gz"' &&
rm ../output/results/ -r &&
mkdir ../output/results/ &&
tar -xzf opt_results.tar.gz -C ../output/results/
# && rm opt_results.tar.gz  # keep zip file, just in case
