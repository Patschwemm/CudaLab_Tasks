import torch
import random

def CutMix(input, target):
    """
    CutMix Implementation. Paper https://arxiv.org/pdf/1905.04899.pdf

    Augmentation: Crop a part of another image out and combine the losses in a weighted way by lambda.
    Use Augmentation in Training Loop.
    
    Args:
    - input: input of (training) data: batch x dims
    - target: 

    """
    # get batch size to shuffle index of minibatch, and set a shuffled list
    B = list(range(input.shape[0]))
    shuffled_batch_idx = torch.tensor(random.sample(B, len(B)))

    # alg according to paper
    W = input[0].shape[1]
    H = input[0].shape[2]
    Lambda = torch.rand(1)
    r_x = torch.rand(1) * H
    r_y = torch.rand(1) * W
    r_w = torch.sqrt(1 - Lambda) * H
    r_h = torch.sqrt(1 - Lambda) * W

    x1 = int(torch.clamp((r_x - r_w / 2), min=0, max=W))
    x2 = int(torch.clamp((r_x + r_w / 2), min=0, max=W))
    y1 = int(torch.clamp((r_y - r_h / 2), min=0, max=H))
    y2 = int(torch.clamp((r_y + r_h / 2), min=0, max=H))
    
    # target = Lambda * target + (1 - Lambda) * target[shuffled_batch_idx]
    Lambda = 1 - ((x2-x1) * (y2 - y1) / (W*H))
    input[:, :, y1:y2, x1:x2] = input[shuffled_batch_idx, :, y1:y2, x1:x2]

    return input, target, shuffled_batch_idx, Lambda

def apply_cutmix(model, criterion, prob_cutmix=0.25):
    """
    Applies upper cutmix in the training loop for the whole batch:
    Implementation for classifying use case.

    Args:
    - model: model 
        to predict output -> in this case predicts the labels
    - criterion: 
        loss function
    - probability: 
        probability with which we apply the cutmix augmentation

    """

    random_prob = torch.rand(1)
    if random_prob <= prob_cutmix:
        # used for cutmix agumentation
        images, labels, shuffled_idx, Lambda = CutMix(images, labels)
        
        # Forward pass to get output/logits
        outputs = model(images)
        
        # Calculate Loss: softmax --> cross entropy loss
        # split loss values according to cutmix paper
        loss = criterion(outputs, labels) * Lambda  + criterion(outputs, labels[shuffled_idx]) * (1 - Lambda)
    else: 
        #compute output as usual
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    return loss
