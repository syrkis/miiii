# Push local changes to git
push = "git add . && git commit -m 'WIP' && git push"

# SSH into cluster, pull repo, run job
remote = "ssh nobr@hpc.itu.dk 'cd ~/miiii && git pull && sbatch script.sh'"

# Copy result from cluster to local machine
pull = "scp -r nobr@hpc.itu.dk:~/miiii/logs ."

# Chain it all
run:
    just push
    just remote
