python train.py --rain_path "./data/test/rain" --norain_path "./data/test/norain" --save_path "./models/models_rain1400" --save_name "KPN_rain1400" --sample_path "./samples_rain1400" --save_by_epoch 2 --lr_g 0.001 --b1 0.5 --b2 0.999 --weight_decay 0.0 --batch_size 8 --epochs 250 --lr_decrease_epoch 2 --num_workers 2 --channel_att True --spatial_att True --angle_aug True --input_size 224