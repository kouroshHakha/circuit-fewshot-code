

CUDA_VISIBLE_DEVICES=0 python scripts_bagnet/run_exp.py --gnn_ckpt results/cgl/Circuit-PT/33c8jvqf/checkpoints/last.ckpt --seed 10 --run_name pt_gnn
CUDA_VISIBLE_DEVICES=1 python scripts_bagnet/run_exp.py --gnn_ckpt results/cgl/Circuit-PT/33c8jvqf/checkpoints/last.ckpt --seed 20 --run_name pt_gnn

CUDA_VISIBLE_DEVICES=2 python scripts_bagnet/run_exp.py --gnn_ckpt results/cgl/Circuit-PT/33c8jvqf/checkpoints/last.ckpt --rand_init --seed 10 --run_name gnn_rand_init
CUDA_VISIBLE_DEVICES=3 python scripts_bagnet/run_exp.py --gnn_ckpt results/cgl/Circuit-PT/33c8jvqf/checkpoints/last.ckpt --rand_init --seed 20 --run_name gnn_rand_init

CUDA_VISIBLE_DEVICES=4 python scripts_bagnet/run_exp.py --gnn_ckpt results/cgl/Circuit-PT/33c8jvqf/checkpoints/last.ckpt --freeze --seed 0  --run_name gnn_frozen
CUDA_VISIBLE_DEVICES=5 python scripts_bagnet/run_exp.py --gnn_ckpt results/cgl/Circuit-PT/33c8jvqf/checkpoints/last.ckpt --freeze --seed 10 --run_name gnn_frozen
CUDA_VISIBLE_DEVICES=6 python scripts_bagnet/run_exp.py --gnn_ckpt results/cgl/Circuit-PT/33c8jvqf/checkpoints/last.ckpt --freeze --seed 20 --run_name gnn_frozen

CUDA_VISIBLE_DEVICES=6 python scripts_bagnet/run_exp.py --seed 0  --run_name mlp
CUDA_VISIBLE_DEVICES=6 python scripts_bagnet/run_exp.py --seed 10 --run_name mlp
CUDA_VISIBLE_DEVICES=6 python scripts_bagnet/run_exp.py --seed 20 --run_name mlp
