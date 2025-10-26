# python main.py --config /home/myw/clp/FreqNext/configs/flip.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info imgaug_cross-C --device cuda:1;

# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set FaceForensics_LQ --val_set WildDeepfake --add_info cross-W --device cuda:1;
# python main.py --config /home/myw/clp/FreqNext/configs/recce.yaml --train_set FaceForensics_LQ --val_set WildDeepfake --add_info cross-W --device cuda:1;
# python main.py --config /home/myw/clp/FreqNext/configs/flip.yaml --train_set FaceForensics_HQ --val_set WildDeepfake --add_info HQ-cross-W --device cuda:1;

# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set FaceForensics_LQ --val_set FaceForensics_LQ --add_info intra-FLQ-crop --device cuda:1;
python main.py --config /home/myw/clp/FreqNext/configs/vit_mean_dr.yaml --train_set FaceForensics_LQ --val_set CelebDF --add_info cross-C --device cuda:1;
python main.py --config /home/myw/clp/FreqNext/configs/vit_mean_dr.yaml --train_set FaceForensics_LQ --val_set FaceForensics_LQ --add_info intra-FLQ --device cuda:1;

# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_FRAll_ff --val_set DF40_FSAll_ff --add_info cross_forgery_fr2fs --device cuda:1;
# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_FRAll_ff --val_set DF40_FRAll_ff --add_info cross_forgery_fr2fr --device cuda:1;
# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_FRAll_ff --val_set DF40_EFSAll_ff --add_info cross_forgery_fr2efs --device cuda:1;

# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_FRAll_ff --val_set DF40_FSAll_cdf --add_info cross_domain_fr2fs --device cuda:1;
# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_FRAll_ff --val_set DF40_FRAll_cdf --add_info cross_domain_fr2fr --device cuda:1;
# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_FRAll_ff --val_set DF40_EFSAll_cdf --add_info cross_domain_fr2efs --device cuda:1;


# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set FaceForensics --val_set CelebDF --add_info F-\>C --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set CelebDF --val_set FaceForensics --add_info C-\>F --device cuda:1;


# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set Casia MSU Replay  --val_set OULU --add_info MCI-\>O --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set Casia MSU Replay FaceForensics CelebDF --val_set OULU --add_info MCIFC-\>O --device cuda:1;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set Casia Replay OULU --val_set MSU --add_info MCIO3→1-M --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set MSU Replay OULU --val_set Casia --add_info MCIO3→1-C --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set MSU Casia OULU --val_set Replay --add_info MCIO3→1-I --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set MSU Casia Replay --val_set OULU --add_info MCIO3→1-O --device cuda:1;

# python main.py --config /home/myw/clp/UAD/configs/basefreq3.yaml --train_set Casia MSU Replay  --val_set OULU --add_info MCI-\>O --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq3.yaml --train_set Casia MSU Replay FaceForensics CelebDF --val_set OULU --add_info MCIFC-\>O --device cuda:1;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set Casia Replay OULU --val_set MSU --add_info MCIO3→1-M --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set MSU Replay OULU --val_set Casia --add_info MCIO3→1-C --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set MSU Casia OULU --val_set Replay --add_info MCIO3→1-I --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set MSU Casia Replay --val_set OULU --add_info MCIO3→1-O --device cuda:1;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set Casia Replay OULU --val_set MSU --add_info MCIO3→1-M --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set MSU Replay OULU --val_set Casia --add_info MCIO3→1-C --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set MSU Casia OULU --val_set Replay --add_info MCIO3→1-I --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set MSU Casia Replay --val_set OULU --add_info MCIO3→1-O --device cuda:1;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set CeFA SURF CelebSpoof --val_set WMCA --add_info WCS2→1\(AD\)-W --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set WMCA SURF CelebSpoof --val_set CeFA --add_info WCS2→1\(AD\)-C --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set WMCA CeFA CelebSpoof --val_set SURF --add_info WCS2→1\(AD\)-S --device cuda:1;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set CeFA SURF CelebSpoof --val_set WMCA --add_info WCS2→1\(AD\)-W --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set WMCA SURF CelebSpoof --val_set CeFA --add_info WCS2→1\(AD\)-C --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set WMCA CeFA CelebSpoof --val_set SURF --add_info WCS2→1\(AD\)-S --device cuda:1;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector_annular.yaml --train_set CeFA SURF CelebSpoof --val_set WMCA --add_info WCS2→1\(AD\)-W --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector_annular.yaml --train_set WMCA SURF CelebSpoof --val_set CeFA --add_info WCS2→1\(AD\)-C --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector_annular.yaml --train_set WMCA CeFA CelebSpoof --val_set SURF --add_info WCS2→1\(AD\)-S --device cuda:1;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set Casia --val_set MSU --add_info MCIO1→1CM --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set Casia --val_set Replay --add_info MCIO1→1CI --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set Casia --val_set OULU --add_info MCIO1→1CO --device cuda:1;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info mask0.5_cross-C --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id.yaml --train_set Casia Replay OULU CelebSpoof --val_set MSU --add_info mask0.5_MCIO3→1\(AD\)-M --device cuda:1;


# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info cross-C --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info cross-C --device cuda:1;


# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set FaceForensics_HQ --val_set DFD --add_info sin-pretrained-cross-D --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set FaceForensics_HQ --val_set FaceForensics_HQ --add_info sin-pretrained-intra-DF-DF --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set CelebDF --val_set CelebDF --add_info sin-pretrained-intra-C --device cuda:1;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id_freqattn_patchalignment.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info rings28-cross-C --set model.num_rings=28 --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id_freqattn_orthoringloss.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info rings28-cross-C --set model.num_rings=28 --device cuda:1;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id_freqattn_parallelfeatfuse_text_img_softmaxalignment.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info rings56-cross-C --set model.num_rings=56 --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id_freqattn_orthoringloss_textcross.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info rings28-ortho1-cross-C --set model.num_rings=28 loss_4.weight=1 --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id_freqattnmodify.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info rings56-cross-C --set model.num_rings=56 --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info rings56-cross-C --set model.num_rings=56 --device cuda:1;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ --val_set FaceForensics_HQ --add_info intra-FHQ --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_LQ --val_set FaceForensics_LQ --add_info intra-FLQ --device cuda:1;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info rings56-cross-C --set model.num_rings=56 --device cuda:1;





# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_FS --val_set FaceForensics_HQ_FS --add_info crosstype-FS-FS --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_FS --val_set FaceForensics_HQ_DF --add_info crosstype-FS-DF --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_FS --val_set FaceForensics_HQ_NT --add_info crosstype-FS-NT --device cuda:1;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_FS --val_set FaceForensics_HQ_F2F --add_info crosstype-FS-F2F --device cuda:1;




