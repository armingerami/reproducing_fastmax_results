# fmm-attention

Instructions for running experiments with FMM-based attention on a minimalist transformer.

1. Install python requirements in a conda environment or a virtualenv:
    ```
    pip install -r requirements.txt
    ```
2. Run one of the examples:
    ```
    cd examples
    python run_image_example.py
    ```

3. Run on that example on the Nexus cluster by using `run_sbatch.sh`. First edit the line of the file that says "proj_root" to your own project folder, then run with:
   ```
   sbatch run_sbatch.sh
   ```

4. If you don't want to run jobs in the background with sbatch, you can open an interactive terminal to a compute node and then just directly run the example:
   ```
   srun --pty --gres=gpu:1 bash
   python examples/run_image_example.py
   ``` 

