#Set slideflow backends
export SF_SLIDE_BACKEND=cucim
export SF_BACKEND=torch

#Set the config file
CONFIG_FILE=pathbench_configs/conf_classification1.yaml

#Run the program
python3 PathBench-MIL/main.py --config $CONFIG_FILE
