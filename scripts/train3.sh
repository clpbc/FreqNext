# python main.py --config /home/myw/clp/FreqNext/configs/freqnext.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info cross-C --device cuda:4;

# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set FaceForensics_LQ --val_set DFD --add_info cross-D --device cuda:4;
# python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set FaceForensics_LQ --val_set DFDC --add_info cross-DP --device cuda:4;

# python main.py --config /home/myw/clp/FreqNext/configs/flip.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info HQ-cross-C --device cuda:4;
python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_EFSAll_ff --val_set DF40_FSAll_ff --add_info cross_forgery_efs2fs --device cuda:4;
python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_EFSAll_ff --val_set DF40_FRAll_ff --add_info cross_forgery_efs2fr --device cuda:4;
python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_EFSAll_ff --val_set DF40_EFSAll_ff --add_info cross_forgery_efs2efs --device cuda:4;

python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_EFSAll_ff --val_set DF40_FSAll_cdf --add_info cross_domain_efs2fs --device cuda:4;
python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_EFSAll_ff --val_set DF40_FRAll_cdf --add_info cross_domain_efs2fr --device cuda:4;
python main.py --config /home/myw/clp/FreqNext/configs/xception.yaml --train_set DF40_EFSAll_ff --val_set DF40_EFSAll_cdf --add_info cross_domain_efs2efs --device cuda:4;


# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set MSU OULU Casia Replay FaceForensics --val_set CelebDF --add_info MCIOF-\>C --device cuda:4;
# python main.py --config /home/myw/clp/UAD/configs/flip.yaml --train_set MSU OULU Casia Replay CelebDF --val_set FaceForensics --add_info MCIOC-\>F --device cuda:4;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set FaceForensics --val_set CelebDF --add_info F-\>C --device cuda:4;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set OULU Casia MSU Replay FaceForensics --val_set CelebDF --add_info MCIOF-\>C --device cuda:4;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set MSU Replay CelebSpoof --val_set Casia --add_info MCIO2→1\(AD\)-C --device cuda:4;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set MSU Replay CelebSpoof --val_set OULU --add_info MCIO2→1\(AD\)-O --device cuda:4;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set MSU Replay --val_set Casia --add_info MCIO2→1-C --device cuda:4;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set MSU Replay --val_set OULU --add_info MCIO2→1-O --device cuda:4;

# python main.py --config /home/myw/clp/UAD/configs/basefreq3.yaml --train_set FaceForensics --val_set CelebDF --add_info F-\>C --device cuda:4;
# python main.py --config /home/myw/clp/UAD/configs/basefreq3.yaml --train_set OULU Casia MSU Replay FaceForensics --val_set CelebDF --add_info MCIOF-\>C --device cuda:4;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set MSU Replay CelebSpoof --val_set Casia --add_info MCIO2→1\(AD\)-C --device cuda:4;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set MSU Replay CelebSpoof --val_set OULU --add_info MCIO2→1\(AD\)-O --device cuda:4;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set MSU Replay CelebSpoof --val_set Casia --add_info MCIO2→1\(AD\)-C --device cuda:4;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set MSU Replay CelebSpoof --val_set OULU --add_info MCIO2→1\(AD\)-O --device cuda:4;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set MSU Replay --val_set Casia --add_info MCIO2→1-C --device cuda:4;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set MSU Replay --val_set OULU --add_info MCIO2→1-O --device cuda:4;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set MSU Replay --val_set Casia --add_info MCIO2→1-C --device cuda:4;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set MSU Replay --val_set OULU --add_info MCIO2→1-O --device cuda:4;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info cross-C --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set FaceForensics_HQ --val_set DFD --add_info cross-D --device cuda:2;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info cross-C --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set FaceForensics_HQ --val_set DFD --add_info cross-D --device cuda:2;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector_annular.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info cross-C --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector_annular.yaml --train_set FaceForensics_HQ --val_set DFD --add_info cross-D --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector_annular.yaml --train_set FaceForensics_HQ --val_set FaceForensics_HQ --add_info intra-Fhq --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector_annular.yaml --train_set CelebDF --val_set CelebDF --add_info intra-C --device cuda:2;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set Replay --val_set Casia --add_info MCIO1→1IC --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set Replay --val_set MSU --add_info MCIO1→1IM --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set Replay --val_set OULU --add_info MCIO1→1IO --device cuda:2;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_cswin.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info cross-C --device cuda:4;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id.yaml --train_set Casia Replay OULU CelebSpoof --val_set MSU --add_info mask0.25_MCIO3→1\(AD\)-M --device cuda:2;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info mask0.5_cross-C --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id_freqattn_parallelfeatfuse.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info rings56-cross-C --set model.num_rings=56 --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id_freqattn_orthoringloss.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info rings56-cross-C --set model.num_rings=56 --device cuda:2;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id_freqattnrandompartmodify.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info rings28-cross-C --set model.num_rings=28 --device cuda:2;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set WildDeepfake --val_set WildDeepfake --add_info intra-Wild --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set CelebDF --val_set CelebDF --add_info intra-C --device cuda:2;


# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_NT --val_set FaceForensics_HQ_FS --add_info crosstype-NT-FS --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_NT --val_set FaceForensics_HQ_DF --add_info crosstype-NT-DF --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_NT --val_set FaceForensics_HQ_NT --add_info crosstype-NT-NT --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_NT --val_set FaceForensics_HQ_F2F --add_info crosstype-NT-F2F --device cuda:2;