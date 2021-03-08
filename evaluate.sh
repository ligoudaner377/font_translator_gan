set -ex
model="font_translator_gan"
dataroot="./datasets/font"
name="test_new_dataset"
phase="test_unknown_content"
python evaluate.py --dataroot ${dataroot} --model ${model} --name ${name} --phase ${phase} --evaluate_mode content

python evaluate.py --dataroot ${dataroot}  --model ${model} --name ${name} --phase ${phase} --evaluate_mode style