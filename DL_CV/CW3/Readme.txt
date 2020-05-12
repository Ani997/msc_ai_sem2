To run VGG16 with MNIST dataset
python train.py --dataset mnist --model vgg16 --reshape '(32,32)' --batch_size 128 --epoch 10 --learning_rate 0.01 --dropout_rate 0.2 --activation_ch softmax --optimizer_ch sgd



To run VGG16 with CIFAR10 dataset
python train.py --dataset cifar --model vgg16 --reshape '(32,32)' --batch_size 128 --epoch 10 --learning_rate 0.01 --dropout_rate 0.2 --activation_ch softmax --optimizer_ch sgd



To run GoogLeNet with MNIST dataset
python train.py --dataset mnist --model googlenet --reshape '(224,224)' --batch_size 128 --epoch 10 --learning_rate 0.001 --dropout_rate 0.2 --activation_ch softmax --optimizer_ch sgd



To run GoogLeNet with CIFAR dataset
python train.py --dataset cifar --model googlenet --reshape '(224,224)' --batch_size 128 --epoch 10 --learning_rate 0.001 --dropout_rate 0.2 --activation_ch softmax --optimizer_ch sgd


File Content Information:
train.py :  Contains scripts to build VGG and GoogLeNet Network, we need to pass arguments to get results, the results are stored in vgg_results.csv and inception_results.csv

processData.py :  Contain script to convert datasets into required dimensions

networks.py : Conatin models, vgg and googlenet.

evaluation.ipynb : results and graphs presented in the report.
