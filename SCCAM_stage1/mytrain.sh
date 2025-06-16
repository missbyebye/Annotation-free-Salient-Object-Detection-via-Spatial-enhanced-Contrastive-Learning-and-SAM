#source activate Yolox_3.10

####   train    ####
#OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2,3 python train_CCAM_DUTS.py --tag CCAM_DUTS_MOCO --batch_size 112 --pretrained mocov2 --alpha 0.25 --data_dir ~/MyWork/SalientVideo/DS-Net/dataset/DUTS-TR/DUTS-TR/DUTS-TR-Image/  --num_workers 0 --max_epoch 80

#OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2,3 python train_CCAM_DUTS.py --tag CCAM_DUTS_MOCO --batch_size 112 --pretrained mocov2 --alpha 0.25 --data_dir ~/MyWork/SalientVideo/DS-Net/dataset/DUTS-TR/DUTS-TE/DUTS-TE-Image/  --num_workers 0 --max_epoch 40

#OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0,1,2 python train_CCAM_DUTS.py --tag CCAM_DUTS_MOCO --batch_size 84 --pretrained mocov2 --alpha 0.25 --data_dir ~/MyWork/SalientVideo/DS-Net/dataset/DUTS-TR/DUTS-TR/DUTS-TR-Image/  --num_workers 0

OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python train_CCAM_DUTS.py --tag CCAM_DUTS_MOCO --batch_size 32 --pretrained mocov2 --alpha 0.25 --data_dir ../dataset/DUTS-TR/DUTS-TR-Image/  --num_workers 0

####    infer   ####
#OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python inference_CCAM_DUTS.py --tag CCAM_DUTS_MOCO --domain train --data_dir ../dataset/DUTS-TR/DUTS-TR-Image/ --ckpt ./experiments/models/CCAM_DUTS_MOCO_epoch9.pth

####    eval    #### 
# python ../Saliency-Evaluation-Toolbox/myEval.py
