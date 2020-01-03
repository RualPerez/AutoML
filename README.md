# AutoML
Deep Reinforcement Learning for Efficient Neural Architecture Search (ENAS) in PyTorch, i.e., AutoML. Code based on the paper https://arxiv.org/abs/1802.03268

How to run the pipeline:

1) Clone the whole repository
```git clone https://github.com/RualPerez/AutoML.git```

2) Create virtualenv
```virtualenv -p /usr/bin/python3.7 virtualenv/AutoML```

3) Activate virtualenv
```source virtualenv/AutoML/bin/activate```

4) Install libraries
```pip3 install -r requirements.txt```

5) Run the main script, for instance:
```python3 main.py --num_episodes 5 --batch 5 --possible_hidden_units 1 4```

Note that you can get a help of how to run the main script by:
```python3 main.py -h```

## File description

