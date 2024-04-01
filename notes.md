
Session 18.11 notes:
- Usually Transformers use LR warmup
- schedulers and warmups are very used in practice
- model.train() and model.eval() tell regularizers like dropout to factor the weights(because there are differences in eval and train like the slides show)
- weight sharing allows us to detect features indepently to where ist is in the image
- same padding = same size, full padding = every pixel is convolved the same number of times
- for the ipyn param number: formula is cnn: layer size * (filtersizex * filtersizey * depth + bias)
- linear: (inunit + bias) * outunits
- plot both train loss for batch (or every 100 batches) and per epoch

Session 02.12 notes:
- degradation problem: denser shallow shift mean and covariance because of non centered activation functions e.g. that is why batchnorm is needed (train loss eval loss high)
- residual learning allows us to learn a difference of input and output instead of finding the difference itself
- dotted lines in resnet: dotted lines mean downsampling (downsampling of identity too so dimensions match)
- in joined fine-tuning: lower lr for joined fine tuning (by magnitude 10 e.g.)

Session 09.12 notes:
- gradient clip against gradient explosion (0.1 is classic value)
- Slide 23: sidmoid and tanh are fc layer then sigmoid (or tanh)
- each gate has it's own layer in LSTM
- LSTM parameter count: ((input_size + hidden_dim + bias) * hidden connected layers) * 4 for one layer
- sometimes 2 bias for fully connected layers instead concat input_size and hidden_dim
- can use pretrained models for the task
- comparison in task: parameters, learning time, accuracy e.g.

Session 16.12 notes:
- difference in MLP autoencoder param amount because biases are not the same (if they are of same dimensions in encoder and decoder -reversed)
- KL-D reinforce a structure of the Latent Space of VAE (not as important as Reconstruction loss, more used as a regularizer)

Session 13.01 notes:
- nowadays state of the art for generative models are diffusion models, GANs were state of the art in 2020
- detach used in discriminator part because we don't want to have the gradients of the generator for the discriminator part

Session 27.01:
- anchor has to be different in random sampling from idx of same labels!
- if only 1 image is given apply strong augmentation (with strong color augmentation)
- concatenate all three anchor, positive, negative in batch dimension
- margin between 0.00 < margin < 2 (2 because normalized embeddings, 0.2 is a good value)
- tensorboard allows embeddings to save as figures (maybe 3d)
- 3d plots that can be interacted with are good for visualization

Session 03.02:
- 1x1 conv are efficient to change num. of channels
- depth wise separable conv, or group convs (in DeepLab v3+, considered State of Art)
- 1. imagenet backbone pre training, 2. train segmentation model on ms coco, 3. fine tune on dataset 
- mIoU most important measurement for semantic segmentation
- some labels not in the current image that is why NaN error can be thrown
- weighting of sky and labels that are dominant
- more elegant way for 255 label is to ignore it in CrossEntropy loss with ignore_index argument
- Session08 not good training recipe, changes values
- crop and flip as augmentation are reasonable

CudaLab Project:
- RNN at the end to improve results (maybe) to be consistent in temporal differences
- for video data we can use temporal augmentation (only take every second frame to simulate speed up)
- loss calculation not necessarily need to be on last img but also maybe only on second-, third- last
- small explanation of the model, more explanation of design ideas we bring and their reasoning why

