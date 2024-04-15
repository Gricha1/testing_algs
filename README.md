# HRAC
This is a PyTorch implementation for our paper "[Generating Adjacency-Constrained Subgoals in Hierarchical Reinforcement Learning](https://arxiv.org/abs/2006.11485)" (NeurIPS 2020 spotlight).

## Dependencies
- Python 3.6
- PyTorch 1.3
- OpenAI Gym
- MuJoCo

~~Also, to run the MuJoCo experiments, a license is required (see [here](https://www.roboti.us/license.html)).~~

## Usage

**Update:** implementation for discrete control tasks is in the `discrete/` folder; please refer to the usage therein.

### Training
- Ant Gather
```
python main.py --env_name AntGather
```
- Ant Maze
```
python main.py --env_name AntMaze
```
- Ant Maze Sparse
```
python main.py --env_name AntMazeSparse
```
### Evaluation
- Ant Gather
```
python eval.py --env_name AntGather --model_dir [MODEL_DIR]
```
- Ant Maze
```
python eval.py --env_name AntMaze --model_dir [MODEL_DIR]
```
- Ant Maze Sparse
```
python eval.py --env_name AntMazeSparse --model_dir [MODEL_DIR]
```
Default `model_dir` is `pretrained_models/`.

## Pre-trained models

See `pretrained_models/` for pre-trained models on all tasks. The expected performances of the pre-trained models are as follows (averaged over 100 evaluation episodes):

|Ant Gather|Ant Maze|Ant Maze Sparse|
|:--:|:--:|:--:|
|3.0|96%|89%|

## Citation
If you find this work useful in your research, please cite:
```
@inproceedings{zhang2020generating,
  title={Generating adjacency-constrained subgoals in hierarchical reinforcement learning},
  author={Zhang, Tianren and Guo, Shangqi and Tan, Tian and Hu, Xiaolin and Chen, Feng},
  booktitle={NeurIPS},
  year={2020}
}
```




# Docker

## docker setup
docker run -it --gpus "device=0" --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -v $(pwd):/usr/home/workspace continuumio/miniconda3 /bin/bash -c "conda install python=3.8.5 -y && bash"

## deps
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorboard
pip install pandas
pip install gym
#pip install mujoco_py
#pip install Cython==3.0.0a10

apt-get update
apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    patchelf
pip install onnxruntime free-mujoco-py

## mujoco
apt-get update
apt-get install build-essential --yes
cd /root
mkdir .mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz -C .mujoco
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

## docker into
docker start HRAC
docker exec -it HRAC bash
cd /usr/home/workspace


# tensorboard
tensorboard --logdir logs --bind_all