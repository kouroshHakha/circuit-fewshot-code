
# pretrain deep_gen on vdc with 15 layers
CUDA_VISIBLE_DEVICES=0 python scripts/pretrain.py --gpus=1 --project=Circuit-PT --max_epoch 100 --path=configs/opamp/dc/deep_gen_net/15-layer/config.py