#!/bin/bash -l

# Set SCC project
#$ -P statnlp

# Set job name
#$ -N train_modanet

# Set when qsub system emails you
#$ -m beas

# Specify hard time limit for the job.
#$ -l h_rt=36:00:00

# Where to stream std out
#$ -o log/qsub.out.txt

# Where to stream std err
#$ -e log/qsub.err.txt

# CPU resources
#$ -pe omp 8

# GPU resources
#$ -l gpus=0.125

# Actual commands
# load required modules
module load python3/3.6.5
module load tensorflow/1.13.1
module load gcc/5.5.0
module load cuda/9.2
module load pytorch/1.0
module load fftw/3.3.4
module load tiff/4.0.6
module load openjpeg/2.1.2
module load imagemagick/7.0.3-5

sh train.sh
