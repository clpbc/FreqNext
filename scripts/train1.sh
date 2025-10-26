# python main.py --config /home/myw/clp/FreqNext/configs/resnet_highfreq.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info imgaug_cross-C --device cuda:0;


python main.py --config /home/myw/clp/FreqNext/configs/vit_mean.yaml --train_set FaceForensics++ --val_set FaceForensics++ --set transforms_new.use_resize=True transforms_new.use_crop=False --add_info resize-intra-FLQ --device cuda:0;
python main.py --config /home/myw/clp/FreqNext/configs/vit_mean.yaml --train_set FaceForensics++ --val_set FaceForensics++ --set transforms_new.use_resize=0 transforms_new.use_crop=1 --add_info crop-intra-FLQ --device cuda:1;
python main.py --config /home/myw/clp/FreqNext/configs/vit_mean.yaml --train_set FaceForensics++ --val_set Celeb-DF-v2 --set transforms_new.use_resize=True transforms_new.use_crop=False --add_info resize-cross-C --device cuda:2;
python main.py --config /home/myw/clp/FreqNext/configs/vit_mean.yaml --train_set FaceForensics++ --val_set Celeb-DF-v2 --set transforms_new.use_resize=0 transforms_new.use_crop=1 --add_info crop-cross-C --device cuda:3;
# python main.py --config /home/myw/clp/FreqNext/configs/vit_cls.yaml --train_set FaceForensics_LQ --val_set FaceForensics_LQ --add_info intra-FLQ --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/xception_groupnorm.yaml --train_set FaceForensics_LQ --val_set CelebDF --add_info cross-C --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/xception_instancenorm.yaml --train_set FaceForensics_LQ --val_set CelebDF --add_info cross-C --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/xception_groupnorm.yaml --train_set FaceForensics_LQ --val_set FaceForensics_LQ --add_info intra-FLQ --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/xception_instancenorm.yaml --train_set FaceForensics_LQ --val_set FaceForensics_LQ --add_info intra-FLQ --device cuda:0;


# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set FaceForensics_LQ --val_set CelebDF --add_info cross-C-resize --device cuda:0;

# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set FaceForensics_LQ --val_set CelebDF --add_info cross-C --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/flip.yaml --train_set FaceForensics_LQ --val_set CelebDF --add_info cross-C --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_FSAll_ff --val_set DF40_FSAll_ff --add_info cross_forgery_fs2fs --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_FSAll_ff --val_set DF40_FRAll_ff --add_info cross_forgery_fs2fr --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_FSAll_ff --val_set DF40_EFSAll_ff --add_info cross_forgery_fs2efs --device cuda:0;

# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set FaceForensics_LQ --val_set CelebDF --add_info cross-C -device cuda:0:

# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_FSAll_ff --val_set DF40_FSAll_ff --add_info cross_forgery_fs2fs --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_FSAll_ff --val_set DF40_FRAll_ff --add_info cross_forgery_fs2fr --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_FSAll_ff --val_set DF40_EFSAll_ff --add_info cross_forgery_fs2efs --device cuda:0;

# python main.py --config /home/myw/clp/FreqNext/configs/vit_cls.yaml --train_set FaceForensics_LQ --val_set CelebDF --add_info 224-cross-C --set transforms.imgSize=224 --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/vit_cls.yaml --train_set FaceForensics_LQ --val_set WildDeepfake --add_info 224-cross-W --set transforms.imgSize=224 --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/vit_cls.yaml --train_set DF40_FSAll_ff --val_set DF40_FSAll_ff --add_info 224-cross-forgery-fs2fs --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/vit_cls.yaml --train_set DF40_FSAll_ff --val_set DF40_FRAll_ff --add_info 224-cross-forgery-fs2fr --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/vit_cls.yaml --train_set DF40_FSAll_ff --val_set DF40_EFSAll_ff --add_info 224-cross-forgery-fs2efs --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/vit_cls.yaml --train_set DF40_FSAll_ff --val_set DF40_FSAll_cdf --add_info 224-cross_domain_fs2fs --device cuda:0;


# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set FaceForensics_LQ --val_set CelebDF --add_info 448-cross-C --set transforms.imgSize=448 --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set FaceForensics_LQ --val_set WildDeepfake --add_info 448-cross-D --set transforms.imgSize=448 --device cuda:0;

# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_FSAll_ff --val_set DF40_FSAll_cdf --add_info cross_domain_fs2fs --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_FSAll_ff --val_set DF40_FRAll_cdf --add_info cross_domain_fs2fr --device cuda:0;
# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_FSAll_ff --val_set DF40_EFSAll_cdf --add_info cross_domain_fs2efs --device cuda:0;






# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set OULU MSU Replay  --val_set Casia --add_info MIO-\>C --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set OULU MSU Casia  --val_set Replay --add_info MCO-\>I --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set Casia MSU Replay  --val_set OULU --add_info MCI-\>O --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set OULU Casia Replay  --val_set MSU --add_info CIO-\>M --device cuda:0;

# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set OULU MSU Replay FaceForensics CelebDF --val_set Casia --add_info MIOFC-\>C --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set OULU MSU Casia FaceForensics CelebDF --val_set Replay --add_info CMOFC-\>I --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set Casia MSU Replay FaceForensics CelebDF --val_set OULU --add_info CMIFC-\>O --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set OULU Casia Replay FaceForensics CelebDF --val_set MSU --add_info CIOFC-\>M --device cuda:0;


# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set Casia Replay OULU CelebSpoof --val_set MSU --add_info MCIO3→1\(AD\)-M --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set MSU Replay OULU CelebSpoof --val_set Casia --add_info MCIO3→1\(AD\)-C --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set MSU Casia OULU CelebSpoof --val_set Replay --add_info MCIO3→1\(AD\)-I --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set MSU Casia Replay CelebSpoof --val_set OULU --add_info MCIO3→1\(AD\)-O --device cuda:0;

# python main.py --config /home/myw/clp/UAD/configs/basefreq3.yaml --train_set OULU Casia Replay  --val_set MSU --add_info CIO-\>M --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq3.yaml --train_set OULU Casia Replay FaceForensics CelebDF --val_set MSU --add_info CIOFC-\>M --device cuda:0;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set CeFA SURF CelebSpoof --val_set WMCA --add_info WCS2→1\(AD\)-W --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set WMCA SURF CelebSpoof --val_set CeFA --add_info WCS2→1\(AD\)-C --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set WMCA CeFA CelebSpoof --val_set SURF --add_info WCS2→1\(AD\)-S --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set CeFA SURF --val_set WMCA --add_info WCS2→1-W --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set WMCA SURF --val_set CeFA --add_info WCS2→1-C --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set WMCA CeFA --val_set SURF --add_info WCS2→1-S --device cuda:0;



# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set Casia Replay OULU CelebSpoof --val_set MSU --add_info MCIO3→1\(AD\)-M --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set MSU Replay OULU CelebSpoof --val_set Casia --add_info MCIO3→1\(AD\)-C --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set MSU Casia OULU CelebSpoof --val_set Replay --add_info MCIO3→1\(AD\)-I --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set MSU Casia Replay CelebSpoof --val_set OULU --add_info MCIO3→1\(AD\)-O --device cuda:0;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set FaceForensics --val_set FaceForensics --add_info intra-F --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set CelebDF --val_set CelebDF --add_info intra-C --device cuda:2;

# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set FaceForensics_HQ --val_set FaceForensics_HQ --add_info intra-Fhq --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set FaceForensics_LQ --val_set FaceForensics_LQ --add_info intra-Flq --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set CelebDF --val_set CelebDF --add_info intra-C --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info cross-C --device cuda:3;
# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set FaceForensics_HQ --val_set DFD --add_info cross-D --device cuda:4;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set Casia Replay OULU CelebSpoof --val_set MSU --add_info MCIO3→1\(AD\)-M --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set MSU Replay OULU CelebSpoof --val_set Casia --add_info MCIO3→1\(AD\)-C --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set MSU Casia OULU CelebSpoof --val_set Replay --add_info MCIO3→1\(AD\)-I --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set MSU Casia Replay CelebSpoof --val_set OULU --add_info MCIO3→1\(AD\)-O --device cuda:0;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set Casia Replay OULU CelebSpoof --val_set MSU --add_info MCIO3→1\(AD\)-M --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set MSU Replay OULU CelebSpoof --val_set Casia --add_info MCIO3→1\(AD\)-C --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set MSU Casia OULU CelebSpoof --val_set Replay --add_info MCIO3→1\(AD\)-I --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set MSU Casia Replay CelebSpoof --val_set OULU --add_info MCIO3→1\(AD\)-O --device cuda:0;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector_annular.yaml --train_set Casia Replay OULU CelebSpoof --val_set MSU --add_info MCIO3→1\(AD\)-M --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector_annular.yaml --train_set MSU Replay OULU CelebSpoof --val_set Casia --add_info MCIO3→1\(AD\)-C --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector_annular.yaml --train_set MSU Casia OULU CelebSpoof --val_set Replay --add_info MCIO3→1\(AD\)-I --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector_annular.yaml --train_set MSU Casia Replay CelebSpoof --val_set OULU --add_info MCIO3→1\(AD\)-O --device cuda:0;


# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set MSU --val_set Casia --add_info MCIO1→1MC --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set MSU --val_set Replay --add_info MCIO1→1MI --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set MSU --val_set OULU --add_info MCIO1→1MO --device cuda:0;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id.yaml --train_set Casia Replay OULU CelebSpoof --val_set MSU --add_info MCIO3→1\(AD\)-M --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv.yaml --train_set Casia Replay OULU CelebSpoof --val_set MSU --add_info MCIO3→1\(AD\)-M --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id.yaml --train_set Casia Replay OULU CelebSpoof --val_set MSU --add_info MCIO3→1\(AD\)-M --device cuda:0;


# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set MSU Casia Replay CelebSpoof --val_set OULU --add_info sin-pretrained-MCIO3→1\(AD\)-O --device cuda:0;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set CeFA SURF CelebSpoof --val_set WMCA --add_info sin-pretrained-WCS2→1\(AD\)-W --device cuda:0;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set MSU --val_set Casia --add_info sin-pretrained-MCIO1→1MC --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set Casia --val_set Replay --add_info sin-pretrained-MCIO1→1CI --device cuda:0;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info rings28-cross-C --set model.num_rings=28 --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info rings56-cross-C --set model.num_rings=56 --device cuda:0;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id_freqattn_orthoringloss_textcross.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info rings56-ortho0.1-cross-C --set model.num_rings=56 loss_4.weight=0.1 --device cuda:0;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ --val_set WildDeepfake --add_info cross-Wild --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ --val_set DFD --add_info cross-DFD --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ --val_set DFDC --add_info cross-DFDC --device cuda:0;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_DF --val_set FaceForensics_HQ_DF --add_info crosstype-DF-DF --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_DF --val_set FaceForensics_HQ_FS --add_info crosstype-DF-FS --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_DF --val_set FaceForensics_HQ_NT --add_info crosstype-DF-NT --device cuda:0;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_DF --val_set FaceForensics_HQ_F2F --add_info crosstype-DF-F2F --device cuda:0;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info CELoss_cross-C --set model.num_rings=28 loss_4.weight=0 --device cuda:0;
