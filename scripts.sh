set -ex

model="font_translator_gan"
dataroot="./datasets/font"
name="wo_gan_loss"
dataset_mode="font"

python train.py --dataroot ${dataroot} --model ${model} --dataset_mode ${dataset_mode} --name ${name} --phase train --no_dropout

python test.py --dataroot ${dataroot}  --model ${model} --dataset_mode ${dataset_mode} --name ${name} --phase test_unknown_style  --eval --no_dropout
python test.py --dataroot ${dataroot}  --model ${model} --dataset_mode ${dataset_mode} --name ${name} --phase test_unknown_content  --eval --no_dropout

python evaluate.py --dataroot ${dataroot} --model ${model} --name ${name} --phase test_unknown_content --evaluate_mode content 
python evaluate.py --dataroot ${dataroot}  --model ${model} --name ${name} --phase test_unknown_content --evaluate_mode style
python evaluate.py --dataroot ${dataroot} --model ${model} --name ${name} --phase test_unknown_style --evaluate_mode content
python evaluate.py --dataroot ${dataroot}  --model ${model} --name ${name} --phase test_unknown_style --evaluate_mode style