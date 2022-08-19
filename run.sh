# CUDA_VISIBLE_DEVICES=6 nohup python run.py --rs --max-epochs 20 --loss 2 &

# --win-len --win-w --cls
CUDA_VISIBLE_DEVICES=7 python run.py --valid-freq 2 --win-len 2 --win-w 0.5