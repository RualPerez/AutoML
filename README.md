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

Once the whole steps have run successfully, the next times you only need to run the last step 5. 

**Output**: The main script saves the trained policy/controller net as policy.pt

## File description

| File / Folder | Description |
| ----------- | ----------- |
| main.py | Main script with runs the AutoML experiment |
| *.py | An auxiliar script necessary to run main.py, its detailed description has been written as python documentation  |
| Policy_Gradient_AutoML.ipynb | Jupyter Notebook designed to help users to understand how this project has been developed |
| article.pdf | Basic article that describes the principles of this project (theory-related). Here it can be found the **results**. |
| requirements.txt | Version of the python libraries necessary to run the main script  |
| images/ | Images used for the article |




