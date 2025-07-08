# Installing PyMC on Windows

Install miniforge

https://conda-forge.org/download/

Add the conda to the startup so that you will have access to the conda command

```bash
~/miniforge3/Scripts/conda init bash
```

Set conda so it will not activate when the shell starts.

```bash
conda config --set auto_activate_base false
```

Install 

```bash
conda create -c conda-forge -n pymc_env "pymc>=5"
conda activate pymc_env
pip install -r requirements.txt
```
