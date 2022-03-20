# IQ-RE: Interference-aware Deep Q-learning with Random Encoder
This is the code for IQ-RE, 
and was adapted from the Google's RL research framework 
[`Dopamine`](https://github.com/google/dopamine).

## Running the Code
All of the commands below are run from the parent `IQ-RE` directory.

### Training
We provide as an example training a IQ_Rainbow agent (`IQ_rainbow.gin`) on `Breakout` but one can 
substitute below commands with `IQ_dqn.gin/dqn.gin/rainbow.gin/SRNN_dqn.gin/SRNN_rainbow.gin`.
Train the agent by executing,

```
python -m train_agent
  --gin_files=configs/IQ_rainbow.gin \
  --base_dir=results/Breakout/RBS=1000000/IQ_Rainbow_1
```

### Plotting Learning Curves
Plot the learning curves by executing,
```
python -m training_performance_plot
```

