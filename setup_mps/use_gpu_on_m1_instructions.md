# Steps to set-up PyTorch for MPS on Mac

1) Install homebrew for ARM64
2) Install wget
3) Download Miniforge for ARM64 (Python and Conda)

```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

4) Restart terminal
5) Initialise conda

```
conda init zsh
```

6) create virtual environment

```
conda create --name gpu-env python=3.10.5
conda activate gpu-env
```

7) install libraries

```
pip3 install torch torchvision torchaudio
pip3 install jupyter pandas numpy matplotlib scikit-learn tqdm
```

8) Create a jupyter notebook to test if it worked

```
jupyter notebook
```

9) Create a notebook in VSCode and copy and paste the URL Of the jupyter server in the VSCode python kernel. Run the code in the `test_mac_mps.ipynb` file. Also run `python setup_mps/test_mac_mps.py` on the terminal.

10) Deactivate virtual environment
```
conda deactivate
```