# Reproduction of vision transformer in pytorch
# Author    : https://github.com/ForeverHaibara 
# Reference : https://github.com/google-research/vision_transformer 


import torch   
import numpy as np

# class Encoder
# class MultiheadAttention
# class MLP
# class VisionTransformer


class Encoder(torch.nn.Module):
    def __init__(self, channels, hidden_size, num_heads = 3, linear_dropout = .1, attention_dropout = .1):
        # num_heads == patch_num
        super().__init__()
        
        self.layer_norm1 = torch.nn.LayerNorm(channels)

        # self.attention   = torch.nn.MultiHeadAttention(
        #                                 embed_dim = channels, 
        #                                 num_heads = num_heads, 
        #                                 dropout = attention_dropout)

        self.attention   =  MultiheadAttention(
                                        embed_dim = channels, 
                                        num_heads = num_heads, 
                                        dropout = attention_dropout)

        self.layer_norm2 = torch.nn.LayerNorm(channels)

        self.mlp         = MLPLayer(channels, hidden_size, dropout = linear_dropout)


    def forward(self, x):
        y = self.layer_norm1(x)
        y = self.attention(y)

        x = x + y 

        y = self.layer_norm2(x) 
        y = self.mlp(x) 

        return x + y



class MultiheadAttention(torch.nn.Module):
    def __init__(self, num_heads = 3, embed_dim = 192, dropout = 0.):
        super().__init__() 
        
        assert embed_dim % num_heads == 0, 'Embedding dimension must be divisible by number of heads.'

        self.num_heads = num_heads
        
        # map the input to qkv
        self.qkv = torch.nn.Linear(embed_dim, embed_dim * 3)

        self.dropout = torch.nn.Dropout(dropout)

        self.attention = None
    
    def forward(self, x):
        n, length, embed_dim = x.shape

        # map the input to qkv
        # [n, length, embed_dim] -> [n, length, embed_dim * 3]
        x = self.qkv(x) 

        # partition x into qkv and distribute embeddings to heads
        # [n, length, embed_dim * 3] -> [3, n, heads, length, embed_dim / heads]
        x = x.reshape((n, length, 3, self.num_heads, embed_dim // self.num_heads))
        x = x.permute((2, 0, 3, 1, 4))
            
        q, k, v = x[0], x[1], x[2]

        # first matmul (remember to transpose k)
        attention = torch.matmul(q, k.permute((0,1,3,2)))

        # scale the attention
        attention = attention * (q.shape[-1] ** -0.5)

        # softmax 
        # we store it because it is useful when visualizing
        self.attention = torch.nn.Softmax(dim = -1)(attention)

        # dropout
        attention = self.dropout(self.attention)

        # second matmul, shape = [n, heads, length, embed_dim / heads]
        x = torch.matmul(attention, v)

        # restore shape
        # [n, heads, length, embed_dim / heads] -> [n, length, embed_dim]
        x = x.permute((0, 2, 1, 3)).reshape((n, length, embed_dim))

        return x



class MLPLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size = None, dropout = .1):
        """An MLP layer consists of two linear layers followed with dropout."""
        super().__init__()

        output_size   = output_size or input_size 
        
        self.dense1   = torch.nn.Linear(input_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.dense1.weight)

        self.act1     = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(dropout)


        self.dense2   = torch.nn.Linear(hidden_size, output_size)
        torch.nn.init.xavier_uniform_(self.dense2.weight)
            
        #self.act2     = torch.nn.LeakyReLU()
        self.dropout2 = torch.nn.Dropout(dropout)


    def forward(self, x):
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        # x = self.act2(x)
        x = self.dropout2(x)

        return x



class VisionTransformer(torch.nn.Module):
    def __init__(self, input_size = 224, output_size = 100,
                        channels = 192, hidden_size = 768,
                        patch_size = 16, 
                        num_heads = 3, encode_layers = 12, 
                        pos_encoding = True,
                        res_head = None,
                        linear_dropout = 0., attention_dropout = 0., encoding_dropout = 0.):
        """
        Vision Transformer

        Parameters
        --------
        input_size   : int, the input image should be of size input_size * input_size
        output_size  : int, the number of classes to classify
        channels     : int, the embedding dimension
        hidden_size  : int, the size of hidden layer in MLP
        patch_size   : int, the size of a patch
        num_heads    : int, number of heads
        encode_layers: int, number of encoders
        pos_encoding : bool, whether or not use the positional encoding
        res_head     : list, the resnet blocks in each stage for resnet-VIT, defaults to None
        linear_dropout   : float, dropout rate in MLP
        attention_dropout: float, dropout rate in multihead attention
        encoding_dropout : float, dropout rate after positional encoding


        Reference: 
        https://arxiv.org/abs/2010.11929 
        """
        super().__init__()

        self.res_head = res_head
        if self.res_head is not None:
            from resnet_torch import ResHead
            # the input size corresponds to the size after ResNet
            input_size = input_size // (2 ** (1 + len(self.res_head)))
            self.res_head = ResHead(self.res_head)

        assert input_size % patch_size == 0, 'Input size must be divisible by patch size.'
        self.input_size = input_size
        self.patch_size = patch_size

        # convolution on each patch is the projection
        self.proj = torch.nn.Conv2d(
            in_channels = 3 if self.res_head is None else 2**(5+len(res_head)),
                                    out_channels = channels,
                                    kernel_size = patch_size, stride = patch_size,
                                    padding = 'valid')

        # num of patches
        patch_num = (input_size // patch_size) ** 2 

        # class tokens and (learnable) positional encodings
        self.cls_token = torch.nn.parameter.Parameter(torch.randn((1, 1, channels)))
        self.pos_encoding = torch.nn.parameter.Parameter(
                            torch.randn((1, patch_num + 1, channels))) if pos_encoding is True else None

        self.dropout = torch.nn.Dropout(encoding_dropout)

        self.encoders = torch.nn.ModuleList(
                                [Encoder(channels  = channels,
                                 hidden_size       = hidden_size,
                                 num_heads         = num_heads,
                                 linear_dropout    = linear_dropout,
                                 attention_dropout = attention_dropout) for _ in range(encode_layers)]
                                            )
        
        self.layer_norm = torch.nn.LayerNorm(channels)

        self.head = torch.nn.Linear(channels, output_size)


    def forward(self, x):
        """Input: torch tensor x with shape [batch_size, 3, height, width]"""
    
        # if there is a ResNet head
        if self.res_head is not None: 
            x = self.res_head(x) 

        # [batch_size, 3, height, width] -> [batch_size, channels, patch_x, patch_y] by conv
        x = self.proj(x)  

        # [batch_size, channels, patch_x, patch_y] -> [batch_size, channels, patch_num]
        x = torch.flatten(x, 2, 3)

        # [batch_size, channels, patch_num] -> [batch_size, patch_num, channels]
        x = x.permute((0,2,1))
        
        # [batch_size, patch_num, channels] -> [batch_size, patch_num + 1, channels]
        cls = torch.tile(self.cls_token, (x.shape[0], 1, 1))
        x = torch.cat([cls, x], dim = 1)

        # positional encoding
        if self.pos_encoding is not None:
            x = x + self.pos_encoding
        x = self.dropout(x)
        
        # encoders
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
        
        # layernorm
        x = self.layer_norm(x)

        # reserve the first "patch"
        x = x[:, 0]

        # linear classifier
        x = self.head(x)

        return x


    def visualize(self, x, layer = 0, dtype = 'int'):
        """
        Visualize the attention-guided image of some x with some of the encoders.

        Parameters
        --------
        x    : 4darray, input images with [N,C,H,W] format (C=3) and entries in [0,1]
        layer: int, the index of attention map for visualization (starting from 1, defaults to the last)
        dtype: str, 'int' or 'float' for the dtype of returned images

        Returns
        --------
        imgs     : 4darray, [N,C,H,W] formatted rgba images (C=4)
        attention: 3darray, [N,H,W] formatted, with entries in [0,1]

        Reference:
        https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb 
        """
        
        if not layer: layer = len(self.encoders)

        # preprocessing, see details in <forward>
        with torch.no_grad():
            _ = self.forward(x)

            device = self.cls_token.device

            # initialization
            attention = torch.eye((self.input_size // self.patch_size)**2 + 1,
                                device = device).unsqueeze(0)
            for i in range(layer):
                # weight size = 65 * 65
                weight = torch.mean(self.encoders[i].attention.attention, dim = 1)

                # bias
                weight += torch.eye(weight.shape[-1], device = device)
                
                # normalize
                weight /= torch.tile(torch.sum(weight, dim = -1).unsqueeze(-1), (1, 1, weight.shape[-1]))

                # accumalate along the layers
                attention = torch.matmul(weight, attention)
            
            # drop the class token in the front
            attention = attention[:, 0, 1:]

            # normalize
            attention = attention  / torch.max(attention, dim = -1)[0].unsqueeze(-1)

            # reshape
            attention = attention.reshape((x.shape[0], 1, 
                            self.input_size // self.patch_size, self.input_size // self.patch_size))

            # tile up to restore the original size of the image by transpose2d
            attention = torch.nn.functional.conv_transpose2d(
                            attention,
                            torch.zeros((1,3, self.patch_size, self.patch_size), 
                                        dtype = torch.float32, device = device) + 1,
                            stride = self.patch_size)
            
            # place the attention map in the alpha channel (rgba imgs)
            imgs = torch.cat((x, attention[:,:1]), dim = 1)
            imgs = imgs.permute((0,2,3,1)).cpu().numpy()

        if 'int' in dtype:
            imgs = (imgs * 255.).clip(0, 255).astype('uint8')
        return imgs, attention


if __name__ == '__main__':
    pass
