import sched
from dataloader_torch import * 
from configs import CONFIG
from transformer_torch import VisionTransformer
import torch

cuda = CONFIG.torch_device
#torch.cuda.set_device(cuda)

resize = (CONFIG.resize, CONFIG.resize)

# Data Preparation
(train_x, train_y) , (test_x, test_y) = preprocessor(CONFIG.torch_data_file, resize = resize)

# Network Definition
net = VisionTransformer(input_size        = resize[0], 
                        output_size       = 100, 
                        channels          = CONFIG.channels,
                        hidden_size       = CONFIG.hidden_size,
                        patch_size        = CONFIG.patch_size,
                        num_heads         = CONFIG.num_heads,
                        encode_layers     = CONFIG.encode_layers,
                        pos_encoding      = CONFIG.pos_encoding,
                        linear_dropout    = CONFIG.linear_dropout,
                        attention_dropout = CONFIG.attention_dropout,
                        encoding_dropout  = CONFIG.encoding_dropout)
net = net.to(cuda)

optim = torch.optim.Adam(net.parameters(), lr = CONFIG.learning_rate)

scheduler = torch.optim.lr_scheduler.LinearLR(optim,
            start_factor = .33333,
            end_factor   = 1.,
            total_iters  = CONFIG.warmup_iters,
            verbose      = False)

losses = []
accs = []

# Logwriter Initialization
from torch.utils.tensorboard import SummaryWriter
import os 
log_file = os.path.join(CONFIG.log_file, 
                        os.path.split(CONFIG.torch_save_path)[-1].split('.')[0])
try:
    os.makedirs(log_file)
except FileExistsError:
    pass
writer = SummaryWriter(log_file)


######################################
#              TRAINING
######################################
n = 50000
epochs = CONFIG.epochs
for epoch in range(len(accs) + 1, epochs + len(accs) + 1):
    for x , labels in dataloader(train_x, train_y, CONFIG.train_size,
                                 cut = CONFIG.cut, mix = CONFIG.mix, onehot = True,
                                 jitter = CONFIG.jitter):
        y = net(x.to(cuda)) 
        labels = torch.tensor(labels, dtype = torch.float32, device = cuda) 
        
        # torch.nn.BCEWithLogitsLoss = Sigmoid + BCE
        # seemingly not good in practice
        #loss = torch.nn.BCEWithLogitsLoss()(y, labels)
        
        # we shall use Softmax + BCE
        y = torch.nn.Softmax(dim=-1)(y)
        loss = torch.nn.functional.binary_cross_entropy(y, labels)
        
        losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()

        if scheduler is not None:
            scheduler.step()
        
    # compute the accuracy on the testing data
    accs.append(test(net, test_x, test_y, CONFIG.test_size, cuda = cuda))
    
    print('Epochs =', epoch, ' Top1/5 Acc =', accs[-1])

    for i in range(len(losses) - n // CONFIG.train_size, len(losses)):
        writer.add_scalar(tag="train/loss", global_step = i, scalar_value = losses[i])  
    writer.add_scalar(tag="valid/top1 acc", global_step = len(accs), scalar_value = accs[-1][0]) 
    writer.add_scalar(tag="valid/top5 acc", global_step = len(accs), scalar_value = accs[-1][1]) 

    # save the model if better
    if accs[-1][0] >= max([i[0] for i in accs]):
        torch.save(net.state_dict(), CONFIG.torch_save_path)


writer.close()
