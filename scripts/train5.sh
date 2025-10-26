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

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set FaceForensics --val_set CelebDF --add_info cross-C --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set FaceForensics --val_set DFD --add_info cross-D --device cuda:2;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set FaceForensics --val_set CelebDF --add_info cross-C --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set FaceForensics --val_set DFD --add_info cross-D --device cuda:2;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set FaceForensics --val_set FaceForensics --add_info intra-F --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin.yaml --train_set CelebDF --val_set CelebDF --add_info intra-C --device cuda:2;

# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set FaceForensics --val_set FaceForensics --add_info intra-F --device cuda:2;
# python main.py --config /home/myw/clp/UAD/configs/basefreq8_sin_reverse.yaml --train_set CelebDF --val_set CelebDF --add_info intra-C --device cuda:2;

python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info cross-C --device cuda:4;
python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set FaceForensics_HQ --val_set CelebDF --add_info cross-C --device cuda:4;

python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set FaceForensics_HQ --val_set DFD --add_info cross-D --device cuda:4;
python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set FaceForensics_HQ --val_set DFD --add_info cross-D --device cuda:4;

python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set FaceForensics_HQ --val_set FaceForensics_HQ --add_info intra-Fhq --device cuda:4;
python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set FaceForensics_HQ --val_set FaceForensics_HQ --add_info intra-Fhq --device cuda:4;

python main.py --config /home/myw/clp/UAD/configs/basefreq8_sector.yaml --train_set CelebDF --val_set CelebDF --add_info intra-C --device cuda:4;
python main.py --config /home/myw/clp/UAD/configs/basefreq8_annular.yaml --train_set CelebDF --val_set CelebDF --add_info intra-C --device cuda:4;