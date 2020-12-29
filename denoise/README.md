# Denoising Autoencoder

#### A Denoising Autoencoder is an autoencoder which learns to reproduce the input from a compressed representation of the output. The main difference between vanilla and denoisisng autoencoder is that a denoising autoencoder is trained on a noisy original input. This noise can be of any type(for eg: Gaussian Noise). The encoder neural network is trained to ignore the noise and create an efficient representation while the decoder tries to reconstruct the output.

#### The noise can be of any type. We have added Gaussian noise on our training datasets before training

#### The model contains:
* An encoder function g(.) parameterized by ϕ
* A decoder function f(.) parameterized by θ
* The low-dimensional code learned for input x<sub>noisy</sub> in the bottleneck layer is the output of encoder, let's call it y 
* The reconstructed input is z = g<sub>ϕ</sub>(y)

#### The parameters (θ,ϕ) are learned together to output a reconstructed data sample same as the original input:
#### <div align='center'> x' = f<sub>θ</sub>(g<sub>ϕ</sub>(x<sub>noisy</sub>)) </div>

#### Our target is to get:
#### <div align='center'> x' ≈ x

#### We have implemented the Denoising Autoencoder using PyTorch. You need to install these external libraries before running our code: 
* pytorch(for model training)
* matplotlib(for plotting graphs and images)
* tqdm(for showing progress bars)
#### Our model has already been trained on the MNIST dataset. To run our code, Open Terminal and navigate to this directory and run:
```
python denoise.py
```
#### You can train a new model from scratch or load our pre-trained model to test.
#### Our output on the MNIST Test set was:
![Output Image](output1.jpg)

#### Our loss function value graph during training:
![Graph Image](lossgraph.jpg)
