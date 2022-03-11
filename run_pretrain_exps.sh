
# pretrain deep_gen on vdc with 15 layers
CUDA_VISIBLE_DEVICES=0 python scripts/pretrain.py --gpus=1 --project=Circuit-PT --max_epoch 100 --path=configs/opamp/dc/deep_gen_net/15-layer/config.py


# finetune the pretrained gnn backbone on vdc prediction of a new opamp topology
CUDA_VISIBLE_DEVICES=0 python scripts/pretrain.py --gpus=1 --project=Circuit-PT --max_steps=50000 --ckpt=results/cgl/Circuit-PT/33c8jvqf/checkpoints/last.ckpt --run_name=ft --train=0 --finetune=1 --path=configs/opamp/biased_pmos/15-layer-ft-all-v1-0.1/config.py

# training from scratch for this task would be 
CUDA_VISIBLE_DEVICES=0 python scripts/pretrain.py --gpus=1 --project=Circuit-PT --max_steps=50000 --train=1 --finetune=0 --path=configs/opamp/biased_pmos/15-layer-scratch-0.1/config.py


# finetune the pretrained gnn backbone on gain prediction of a new opamp topology
CUDA_VISIBLE_DEVICES=0 python scripts/train_graph_prediction.py --gpus=1 --project=Circuit-PT --max_steps=50000 --gnn_ckpt=results/cgl/Circuit-PT/33c8jvqf/checkpoints/last.ckpt --run_name=gain_ft --path=configs/opamp/biased_pmos_gain/15-layer-ft-all-v1-0.1/config.py

# training from scratch for this task would be 
CUDA_VISIBLE_DEVICES=0 python scripts/train_graph_prediction.py --gpus=1 --project=Circuit-PT --max_steps=50000 --run_name=gain_scratch --path=configs/opamp/biased_pmos_gain/15-layer-scratch-0.1/config.py