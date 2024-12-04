Notes on Slurm job

* All files must be on shared filesystem

slurm_job1.sh
#This is a good first test as it does not use any file on the shared filesystem
$sbatch -p gpu slurm_job.sh
sbatch: warning: can't run 1 processes on 3 nodes, setting nnodes to 1
Submitted batch job 1180

"sbatch: warning: can't run 1 processes on 3 nodes, setting nnodes to 1" => The reason is the problem is written by leveraging MPI

$ squeue
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
              1180       gpu  rag_adv cuix@AD.  R       0:05      1 nyro-up-oitqaub06

#look at "--error" file and "--output" file under ~/
#~/%j.err => /home/<username>/%j.err is more clear

###########################################################################################
#!/bin/bash
#SBATCH --job-name=rag_adv       # create a short name for your job
#SBATCH --nodes=3                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus=1
#SBATCH --error=~/%j.err
#SBATCH --output=~/%j.out
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=cuix@sec.gov


conda activate hal-july2024
conda info
nvidia-smi
hostname
sleep 5m

####################################################################################################









https://leetcode.com/studyplan/leetcode-75/

8 Patterns for Leetcode
  https://www.youtube.com/watch?v=RYT08CaYq6A
  Chapters
0:00 - Intro
0:49 - Two Pointers
2:29 - Sliding Window
4:20 - Binary Search
7:34 - BFS
10:08 - DFS
13:08 - Backtracking
16:58 - Priority Queue (Heap)
19:01 - Dynamic Programming

Giw ti Sikve ANY LeetCode Problem
https://www.youtube.com/watch?v=OTNe0eV8418

8 pattens to solve 80% LeetCode problems
https://www.youtube.com/watch?v=xo7XrRVxH8Y

1. Two pointers
2. Binary Tree BFS
3. Topological Sort
4 Binary Tree DFS
5 Top K elements
6 Modified Binary Search
7 Subset
8 Sliding Window
