# Interference-aware Deep Q-learning
This is the code for the paper
`IQ: Interference-aware Deep Q-learning`
by Tiantian Zhang, Xueqian Wang, Bin Liang, Xiu Li, and Bo Yuan (TNNLS 2022). 

### Setup
Start a virtualenv with these commands:

```
virtualenv -p python3 .
source ./bin/activate
```

Then install necessary packages: 

```
pip install -r requirements.txt
```

### Training
We provide as an example training a IQ_Rainbow agent (`IQ_rainbow.gin`) on `Breakout` but one can 
substitute below commands with `IQ_dqn.gin/dqn.gin/rainbow.gin/SRNN_dqn.gin/SRNN_rainbow.gin`.
Train the agent by executing,

```
python -m train_agent
  --gin_files=configs/IQ_rainbow.gin \
  --base_dir=results/Breakout/RBS=1000000/IQ_Rainbow_1
```

