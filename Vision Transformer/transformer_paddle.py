# Reproduce of vision transformer in paddlepaddle
# Author    : https://github.com/ForeverHaibara 
# Reference : https://github.com/google-research/vision_transformer 


import paddle   
import numpy as np

# class Encoder
# class MultiheadAttention
# class MLP
# class VisionTransformer


class Encoder(paddle.nn.Layer):
    def __init__(self, channels, hidden_size, num_heads = 3, linear_dropout = .1, attention_dropout = .1):
        # num_heads == patch_num
        super().__init__()
        
        self.layer_norm1 = paddle.nn.LayerNorm(channels)

        # self.attention   = paddle.nn.MultiHeadAttention(
        #                                 embed_dim = channels, 
        #                                 num_heads = num_heads, 
        #                                 dropout = attention_dropout)

        self.attention   =  MultiheadAttention(
                                        embed_dim = channels, 
                                        num_heads = num_heads, 
                                        dropout = attention_dropout)

        self.layer_norm2 = paddle.nn.LayerNorm(channels)

        self.mlp         = MLPLayer(channels, hidden_size, dropout = linear_dropout)


    def forward(self, x):
        y = self.layer_norm1(x)
        y = self.attention(y)

        x = x + y 

        y = self.layer_norm2(x) 
        y = self.mlp(x) 

        return x + y



class MultiheadAttention(paddle.nn.Layer):
    def __init__(self, num_heads = 3, embed_dim = 192, dropout = 0.):
        super().__init__() 
        
        assert embed_dim % num_heads == 0, 'Embedding dimension must be divisible by number of heads.'

        self.num_heads = num_heads
        
        # map the input to qkv
        self.qkv = paddle.nn.Linear(embed_dim, embed_dim * 3)

        self.dropout = paddle.nn.Dropout(dropout)

        self.attention = None
    
    def forward(self, x):
        n, length, embed_dim = x.shape

        # map the input to qkv
        # [n, length, embed_dim] -> [n, length, embed_dim * 3]
        x = self.qkv(x) 

        # partition x into qkv and distribute embeddings to heads
        # [n, length, embed_dim * 3] -> [3, n, heads, length, embed_dim / heads]
        x = x.reshape((n, length, 3, self.num_heads, embed_dim // self.num_heads))
        x = x.transpose((2, 0, 3, 1, 4))
            
        q, k, v = x[0], x[1], x[2]

        # first matmul (remember to transpose k)
        attention = paddle.matmul(q, k.transpose((0,1,3,2)))

        # scale the attention
        attention = attention * (q.shape[-1] ** -0.5)

        # softmax 
        # we store it because it is useful when visualizing
        self.attention = paddle.nn.Softmax(axis = -1)(attention)

        # dropout
        attention = self.dropout(self.attention)

        # second matmul, shape = [n, heads, length, embed_dim / heads]
        x = paddle.matmul(attention, v)

        # restore shape
        # [n, heads, length, embed_dim / heads] -> [n, length, embed_dim]
        x = x.transpose((0, 2, 1, 3)).reshape((n, length, embed_dim))

        return x



class MLPLayer(paddle.nn.Layer):
    def __init__(self, input_size, hidden_size, output_size = None, dropout = .1):
        """An MLP layer consists of two linear layers followed with dropout."""
        super().__init__()

        output_size   = output_size or input_size 
        
        self.dense1   = paddle.nn.Linear(input_size, hidden_size, 
                                        weight_attr = paddle.nn.initializer.XavierUniform())
        self.act1     = paddle.nn.LeakyReLU()
        self.dropout1 = paddle.nn.Dropout(dropout)


        self.dense2   = paddle.nn.Linear(hidden_size, output_size, 
                                        weight_attr = paddle.nn.initializer.XavierUniform()) 
        #self.act2     = paddle.nn.LeakyReLU()
        self.dropout2 = paddle.nn.Dropout(dropout)


    def forward(self, x):
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        # x = self.act2(x)
        x = self.dropout2(x)

        return x



class VisionTransformer(paddle.nn.Layer):
    def __init__(self, input_size = 224, output_size = 100,
                        channels = 192, hidden_size = 768,
                        patch_size = 16, 
                        num_heads = 3, encode_layers = 12, 
                        pos_encoding = True,
                        linear_dropout = 0., attention_dropout = 0., encoding_dropout = 0.):
        super().__init__()

        assert input_size % patch_size == 0, 'Input size must be divisible by patch size.'
        self.input_size = input_size
        self.patch_size = patch_size

        # convolution on each patch is the projection
        self.proj = paddle.nn.Conv2D(in_channels = 3, out_channels = channels,
                                     kernel_size = patch_size, stride = patch_size,
                                     padding = 'VALID')

        # num of patches
        patch_num = (input_size // patch_size) ** 2 

        # class tokens and (learnable) positional encodings
        self.params = paddle.nn.ParameterList(
                        [paddle.create_parameter((1, 1, channels),dtype = 'float32'),
                         paddle.create_parameter((1, patch_num + 1, channels),dtype = 'float32')]
                                                )
        self.cls_token = self.params[0]
        self.pos_encoding = pos_encoding and self.params[1]

        self.dropout = paddle.nn.Dropout(encoding_dropout)

        self.encoders = paddle.nn.LayerList(
                                [Encoder(channels  = channels,
                                 hidden_size       = hidden_size,
                                 num_heads         = num_heads,
                                 linear_dropout    = linear_dropout,
                                 attention_dropout = attention_dropout) for _ in range(encode_layers)]
                                            )
        
        self.layer_norm = paddle.nn.LayerNorm(channels)

        self.head = paddle.nn.Linear(channels, output_size)


    def forward(self, x):
        """Input: paddle tensor x with shape [batch_size, 3, height, width]"""
    
        # [batch_size, 3, height, width] -> [batch_size, channels, patch_x, patch_y] by conv
        x = self.proj(x)  

        # [batch_size, channels, patch_x, patch_y] -> [batch_size, channels, patch_num]
        x = paddle.flatten(x, 2, 3)

        # [batch_size, channels, patch_num] -> [batch_size, patch_num, channels]
        x = x.transpose((0,2,1))
        
        # [batch_size, patch_num, channels] -> [batch_size, patch_num + 1, channels]
        cls = paddle.tile(self.cls_token, (x.shape[0], 1, 1))
        x = paddle.concat([cls, x], axis = 1)

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
        with paddle.no_grad():
            _ = self.forward(x)

        # initialization
        attention = paddle.eye((self.input_size // self.patch_size)**2 + 1).unsqueeze(0)
        for i in range(layer):
            # weight size = 65 * 65
            weight = paddle.mean(self.encoders[i].attention.attention, axis = 1)

            # bias
            weight += paddle.eye(weight.shape[-1])
            
            # normalize
            weight /= paddle.tile(paddle.sum(weight, axis = -1).unsqueeze(-1), (1, 1, weight.shape[-1]))

            # accumalate along the layers
            attention = paddle.matmul(weight, attention)
        
        # drop the class token in the front
        attention = attention[:, 0, 1:]

        # normalize
        attention = attention  / paddle.max(attention, axis = -1).unsqueeze(-1)

        # reshape
        attention = attention.reshape((x.shape[0], 1, 
                        self.input_size // self.patch_size, self.input_size // self.patch_size))

        # tile up to restore the original size of the image by transpose2d
        attention = paddle.nn.functional.conv2d_transpose(
                        attention,
                        paddle.zeros((1,3, self.patch_size, self.patch_size), dtype = 'float32') + 1,
                        stride = self.patch_size)
        
        # place the attention map in the alpha channel (rgba imgs)
        imgs = paddle.concat((x, attention[:,:1]), axis = 1)
        imgs = imgs.transpose((0,2,3,1)).numpy()

        if 'int' in dtype:
            imgs = (imgs * 255.).clip(0, 255).astype('uint8')
        return imgs, attention