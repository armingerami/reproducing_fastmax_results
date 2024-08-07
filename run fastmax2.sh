#!/bin/bash -x

#SBATCH --mem=32G
#SBATCH --qos=high     
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:rtxa5000:1


module load Python3
source ./venv/bin/activate
module load gcc
module load cuda/12.1.1
pip install -r requirements.txt
python setup.py install
# python examples/run_text_example\ fastmax.py --dropout 0.2 --token_length 20000 --batch 4
# python examples/run_text_example\ fastmax.py --dropout 0.2 --token_length 10000 --batch 8
python examples/run_text_example\ fastmax.py --dropout 0.2 --token_length 200 --batch 64