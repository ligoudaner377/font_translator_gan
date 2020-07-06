set -ex
python test.py --dataroot ./datasets/font  --model font_translator_gan --dataset_mode font --eval --name test_font --no_dropout