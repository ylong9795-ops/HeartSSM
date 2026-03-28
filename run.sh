# rigidly align all CT whole-heart segmentations (WHS) to the atlas space 
python whs_seg_process.py --data_dir='YOUR_WHS_DIR' --save_dir='./data/whs/'

# create four-chamber mesh atlas based on whole-heart CT segmentations
python whs_mesh_template.py --data_dir='./data/whs/' --device='cuda:0'

# train label completion U-Net (LC-U-Net)
python train_label_complete.py --data_dir='./data/whs/' --n_epoch=200 --device='cuda:0' --tag='lcunet' --drop_rate=0.2 --sigma=2.0 --scaling=0.2

# perform motion correction for UK Biobank CMR images
python motion_correction.py --data_dir='YOUR_UKBB_DIR' --save_dir='./data/ukbb/' --device='cuda:0'

# perform label completion for UK Biobank CMR images
python eval_label_complete.py --data_dir='YOUR_UKBB_DIR' --save_dir='./data/ukbb/' --n_frame=50 --device='cuda:0'

# train HeartFFDNet
python train_ffd.py --data_dir='./data/ukbb/' --n_epoch=200 --n_frame=50 --n_train=600 --n_valid=100 --drop_rate=0.0 --lr=1e-4 --device='cuda:0' --tag='unet' --w_edge=0.5 --w_curv=1.0 --w_geod=0.1 --w_cycl=0.2

# trian HeartSSM
python heartssm_train.py --data_dir='./data/ukbb/' --n_frame=50 --batch_size=128 --device='cuda:0'

# use HeartSSM for mesh fitting of all UK Biobank participants
python heartssm_feat.py --data_dir='./data/ukbb/' --n_frame=50 --device='cuda:0'
