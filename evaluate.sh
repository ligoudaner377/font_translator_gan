set -ex
model="font_translator_gan"
dataroot="./datasets/font"
python evaluate.py --dataroot ${dataroot} --model ${model} --name test_new_dataset --phase test_unknown_content --evaluate_mode content

python evaluate.py --dataroot ${dataroot}  --model ${model} --name test_new_dataset --phase test_unknown_content --evaluate_mode style