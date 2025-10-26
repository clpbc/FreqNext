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

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set CeFA SURF CelebSpoof --val_set WMCA --add_info WCS2→1\(AD\)-W --device cuda:5;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set WMCA SURF CelebSpoof --val_set CeFA --add_info WCS2→1\(AD\)-C --device cuda:5;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set WMCA CeFA CelebSpoof --val_set SURF --add_info WCS2→1\(AD\)-S --device cuda:5;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set CeFA SURF --val_set WMCA --add_info WCS2→1-W --device cuda:5;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set WMCA SURF --val_set CeFA --add_info WCS2→1-C --device cuda:5;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set WMCA CeFA --val_set SURF --add_info WCS2→1-S --device cuda:5;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set CeFA SURF CelebSpoof --val_set WMCA --add_info WCS2→1\(AD\)-W --device cuda:5;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set WMCA SURF CelebSpoof --val_set CeFA --add_info WCS2→1\(AD\)-C --device cuda:5;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set WMCA CeFA CelebSpoof --val_set SURF --add_info WCS2→1\(AD\)-S --device cuda:5;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set CeFA SURF --val_set WMCA --add_info WCS2→1-W --device cuda:5;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set WMCA SURF --val_set CeFA --add_info WCS2→1-C --device cuda:5;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set WMCA CeFA --val_set SURF --add_info WCS2→1-S --device cuda:5;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set FaceForensics_HQ --val_set FaceForensics_HQ --add_info intra-Fhq --device cuda:3;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set FaceForensics_LQ --val_set FaceForensics_LQ --add_info intra-Flq --device cuda:3;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set CelebDF --val_set CelebDF --add_info intra-C --device cuda:3;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set FaceForensics_HQ --val_set FaceForensics_HQ --add_info intra-Fhq --device cuda:3;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set FaceForensics_LQ --val_set FaceForensics_LQ --add_info intra-Flq --device cuda:3;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set CelebDF --val_set CelebDF --add_info intra-C --device cuda:3;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set Casia Replay OULU CelebSpoof --val_set MSU --add_info MCIO3→1\(AD\)-M --device cuda:3;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set Casia Replay OULU CelebSpoof --val_set MSU --add_info MCIO3→1\(AD\)-M --device cuda:3;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set MSU Casia Replay CelebSpoof --val_set OULU --add_info MCIO3→1\(AD\)-O --device cuda:3;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set MSU Casia Replay CelebSpoof --val_set OULU --add_info MCIO3→1\(AD\)-O --device cuda:3;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set CeFA SURF CelebSpoof --val_set WMCA --add_info WCS2→1\(AD\)-W --device cuda:3;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set CeFA SURF CelebSpoof --val_set WMCA --add_info WCS2→1\(AD\)-W --device cuda:3;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set WMCA CeFA CelebSpoof --val_set SURF --add_info WCS2→1\(AD\)-S --device cuda:3;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set WMCA CeFA CelebSpoof --val_set SURF --add_info WCS2→1\(AD\)-S --device cuda:3;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set OULU --val_set Casia --add_info MCIO1→1OC --device cuda:4;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set OULU --val_set Replay --add_info MCIO1→1OI --device cuda:4;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set OULU --val_set MSU --add_info MCIO1→1OM --device cuda:4;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info mask0.25_cross-C --device cuda:3;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info mask0.25_cross-C --device cuda:3;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_conv_id_freqattnrandompartmodify.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info rings56-cross-C --set model.num_rings=56 --device cuda:3;

python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_F2F --val_set FaceForensics_HQ_FS --add_info crosstype-F2F-FS --device cuda:3;
python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_F2F --val_set FaceForensics_HQ_DF --add_info crosstype-F2F-DF --device cuda:3;
python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_F2F --val_set FaceForensics_HQ_NT --add_info crosstype-F2F-NT --device cuda:3;
python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular_convid_freqattn_fullfreqcross_textcross_freqbrunch_fusemain.yaml --train_set FaceForensics_HQ_F2F --val_set FaceForensics_HQ_F2F --add_info crosstype-F2F-F2F --device cuda:3;
