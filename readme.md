

# Science Cluster Stuff

* Make virtual environment
```bash
module load mamba
mamba activate ma-env
```

* Module Load, unload, list
```bash
module load a100
module unload a100
module list 
```

* Install GPU stuff:
```bash
nvidia-smi #to check cuda version
mamba install pytorch pytorch-cuda=<required-version> transformers deepspeed -c pytorch -c nvidia
mamba install -r requirements.txt
python -c 'import tensorflow as tf; print("Built with CUDA:", tf.test.is_built_with_cuda()); print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU"))); print("TF version:", tf.__version__)'
```


* Interactive session
```srun --pty -n 1 -c 2 --time=00:15:00 --gpus=A100:1 bash -l```

