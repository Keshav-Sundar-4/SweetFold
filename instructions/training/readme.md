## Instructions:
1. Configure full.yaml to your liking and connect it to the actual file paths
2. Configure train.py properly and point train.sh to the correct files/file paths
3. Run train.sh. The output will be logged in the file path chosen in full.yaml

## File descriptions:
- full.yaml: A config file for glycan structure training. The full.yaml config provides intialization instructions for the confidence module as well as structure prediction moduless.
- train.py: A file that lays the foundation for training the Boltz-1x model
    - train.sh: Helper shell script used to run train.py
