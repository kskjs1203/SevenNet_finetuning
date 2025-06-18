This is the code for rehearsal

The additional input in input.yaml is

rehearsal: True
- This define the rehearsal options is on or off

load_memory_path: [ 'task1.extxyz' ]
- This input is file list that uses rehearsal

mem_batch_size: 8
- This is the hyper parameter that how much data is in training long new dataset.
For example, if uses batch_size : 8 for new dataset and mem_batch_size : 8, the memory data and  new data trained half.
If increases  mem_batch_size to 16, the old data, new data trained with ratio 2:1
So you should adjust this parameter to set how much learning importmation old data relative to new data


mem_ratio : 1
- This is the parameters how much data used to form memory.
The mini-batch for rehearsal is create from the fixed memory.
For example, if mem_ratio is set to 0.7,  then the memory is created from load_memory_path files with the ratio


ALERT:
The train batch is first learned then the memory batch is learned not shuffled state.
