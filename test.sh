set -ex
python layer_attention.py --dataroot ./datasets/font  --model font_translator_gan  --eval --name MLANH --no_dropout  --no_dropout
