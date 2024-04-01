import torch
import torch.nn.functional as F
import torch.nn as nn

class TripletLoss(nn.Module):
    """ Implementation of the triplet loss function """
    def __init__(self, margin=0.2, reduce="mean"):
        """ Module initializer """
        assert reduce in ["mean", "sum"]
        super().__init__()
        self.margin = margin
        self.reduce = reduce
        return
        
    def forward(self, anchor, positive, negative):
        """ Computing pairwise distances and loss functions """
        # L2 distances
        d_ap = (anchor - positive).pow(2).sum(dim=-1)
        d_an = (anchor - negative).pow(2).sum(dim=-1)
        
        # triplet loss function
        loss = (d_ap - d_an + self.margin)
        loss = torch.maximum(loss, torch.zeros_like(loss))
        
        # averaging or summing      
        loss = torch.mean(loss) if(self.reduce == "mean") else torch.sum(loss)
      
        return loss

class TripletSemiHardLoss(torch.nn.Module):
    def __init__(self, margin=0.2):
        super(TripletSemiHardLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute the L2 distances
        pos_dist = F.pairwise_distance(anchor, positive, 2)
        neg_dist = F.pairwise_distance(anchor, negative, 2)
        
        # Compute the distance difference between the anchor-positive and anchor-negative pairs
        margin_dist = self.margin + pos_dist - neg_dist
        
        # Compute the loss as the maximum of the margin and the distance difference, averaged over all the triplets in the batch
        loss = torch.mean(torch.max(margin_dist, torch.zeros_like(margin_dist)))
        
        return loss
    

class n_pair_loss(nn.Module):
    """
    Compute the N-pair loss.
    
    Args:
        anchor: Tensor of shape (batch_size, embedding_size).
        positive: Tensor of shape (batch_size, embedding_size) containing positive samples.
        negative: Tensor of shape (batch_size, embedding_size) containing negative samples.
        l2_reg: Float. L2 regularization strength.
    
    Returns:
        Scalar tensor representing the mean N-pair loss.
    """

    def __init__(self, l2_reg=0.02) -> None:
        super().__init__()
        self.l2_reg = l2_reg

    def forward(self, anchor, positive, negative):
        batch_size, embedding_size = anchor.shape
        
        # reshape inputs
        anchor = anchor.view(batch_size, 1, embedding_size)
        positive = positive.view(batch_size, 1, embedding_size)
        negative = negative.view(batch_size, 1, embedding_size)
        
        # Compute similarity matrix
        pos_sim = torch.matmul(anchor, positive.transpose(2, 1))
        neg_sim = torch.matmul(anchor, negative.transpose(2, 1))
        
        # compute loss
        numerator = torch.exp(pos_sim[:, :, 0])  # shape (batch_size,)
        denominator = numerator + torch.exp(neg_sim[:, :, 0]).sum(dim=1)
        loss = -torch.log(numerator / denominator).mean()
        
        # add L2 regularization to the loss
        l2_loss = (anchor ** 2).sum() / (2 * batch_size)
        loss += self.l2_reg * l2_loss
        
        return loss
    

class angular_loss(nn.Module):
    """
    Compute the Angular Loss given anchor, positive, and negative samples.
    
    Args:
        anchor: Tensor of shape (batch_size, embedding_size) representing the anchor samples.
        positive: Tensor of shape (batch_size, embedding_size) representing the positive samples.
        negative: Tensor of shape (batch_size, embedding_size) representing the negative samples.
        m: Float. Margin hyperparameter that controls the distance between the positive and negative samples in the angular space.
    
    Returns:
        Scalar tensor representing the mean Angular Loss.
    """

    def __init__(self, margin=0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor = torch.nn.functional.normalize(anchor, p=2, dim=1)
        positive = torch.nn.functional.normalize(positive, p=2, dim=1)
        negative = torch.nn.functional.normalize(negative, p=2, dim=1)

        # Concatenate anchor, positive, and negative samples into a single tensor.
        inputs = torch.cat([anchor, positive, negative], dim=0)
        
        # Compute cosine similarity matrix.
        sim_matrix = torch.matmul(inputs, inputs.transpose(1, 0))
        
        # Clip cosine similarity values to avoid NaN values.
        sim_matrix = torch.clamp(sim_matrix, -1 + 1e-7, 1 - 1e-7)
        
        # Compute pairwise angles between embeddings and weights.
        theta = torch.acos(sim_matrix)
        
        # Split the angles into anchor-positive and anchor-negative pairs.
        pos_theta = theta[:anchor.size(0), anchor.size(0):]
        neg_theta = theta[:anchor.size(0), -negative.size(0):]
        
        # Compute loss.
        pos_margin = torch.ones_like(pos_theta) * self.margin
        neg_margin = torch.zeros_like(neg_theta)
        pos_loss = torch.max(pos_margin + pos_theta, torch.zeros_like(pos_theta)).mean()
        neg_loss = torch.max(neg_margin - neg_theta, torch.zeros_like(neg_theta)).mean()
        loss = pos_loss + neg_loss
        
        return loss
        
        return loss
    