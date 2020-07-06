set -ex
python train.py --dataroot ./datasets/font --model font_translator_gan --dataset_mode font --name train_font --no_dropout