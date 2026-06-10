## Instructions:
1. Download the original Boltz checkpoint from the Boltz-1x huggingface and convert it using convert_checkpoint.py
2. Use this new checkpoint path for the  checkpoint line in full.yaml. If you have already trained you can also resume from a checkpoint.
3. Configure full.yaml to your liking and connect it to the actual file paths
4. Configure train.py properly and point train.sh to the correct files/file paths
5. Run train.sh. The output will be logged in the file path chosen in full.yaml
6. When done training, use update_checkpoint.py with the proper file paths to update the checkpoint

## File descriptions:
- full.yaml: A config file for glycan structure training. The full.yaml config provides intialization instructions for the confidence module as well as structure prediction moduless.
- train.py: A file that lays the foundation for training the Boltz-1x model
    - train.sh: Helper shell script used to run train.py
