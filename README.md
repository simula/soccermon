## Soccer Dataset 

### Description 
The repository restructures the PMSys dataset to a 
class based representation. The original data is provided in two different dimensions. The first dimension is player centered and provides 
a .csv file for each player with the according features. The second dimension is features centered and provides 
.csv files for each feature. The repository provides a scripts to transform and store the data into more convenient formats. 
The repository is meant to simplify the usage of the PMSys data for the end user. 


### Usage 

The script runs in conda environment which is initialised by 
<pre><code>conda env create -f environment.yml
</code></pre>

The data is not provided in this repo. Hence, the user needs to input the PMSys data in to the 
<pre><code>soccer_dataset/input/
</code></pre>
folder in order to run the scripts correctly.

The functions for processing and transforming the data into a class based representation can be found 
<pre><code>soccer_dataset/preprocessing/read_in_data.py
</code></pre>

We also provide a script to store the class based data representation as pickle files in 
<pre><code>soccer_dataset/preprocessing/run_team_pickle_generation.py
</code></pre>

