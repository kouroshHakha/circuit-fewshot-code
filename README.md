
# Running the experiments

You can now launch pre-training by running:
```
python scripts/pretrain.py --path <path_to_config> --gpus 1
```

For more information on the options run `python scripts/pretrain.py -h`.

And for fine-tuning with a pretrained checkpoint to do voltage prediction on a new circuit: 
```
scripts/pretrain.py --project=<wandb_project_name> --gpus=1 --max_steps=50000 --ckpt=<path_to_pretrained_network.ckpt> --train=0 --finetune=1 --path= <path_to_downstream_config>
```

For fine-tuning on a graph property prediction run:
```
python scripts/train_graph_prediction.py --project=<wandb_project_name> --gpus=1 --max_steps=5000 --gnn_ckpt=<gnn_backbone_ckpt> --path=<config_path>
```

