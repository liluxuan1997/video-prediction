We implemented causualLSTMnet and convLSTMnet. In main_gan.py, we apply convLSTMnet.
disNet.py is the code for discriminator.

The model we train on convLSTM only is stored in model_best.pth.
The latest model we train on GAN combined convLSTM and changed ResNet is stored in model_generator_new_pretrain.pth.

If you want to test the model, you should run command below
python main_gan.py --test <path to test data> --save_g <generator model to test>

If you want to train the model from the begining, just run
python main_gan.py --pretrain

--pretrain is an optional argument depending on that you choose a pretrained model or not. The
default pretrained model is model_best.pth.

Other more optional arguments can be gained by run
python main_gan.py --help