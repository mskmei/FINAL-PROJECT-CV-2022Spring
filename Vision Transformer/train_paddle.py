from dataloader_paddle import * 
from configs import CONFIG
from transformer_paddle import VisionTransformer
import paddle 

paddle.set_device(CONFIG.paddle_device)

resize = (CONFIG.resize, CONFIG.resize)

# Data Preparation
(train_x, train_y) , (test_x, test_y) = preprocessor(CONFIG.paddle_data_file, resize = resize)

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

scheduler = paddle.optimizer.lr.LinearWarmup(
        learning_rate = CONFIG.learning_rate,
        warmup_steps = CONFIG.warmup_iters,
        start_lr = 0., 
        end_lr = CONFIG.learning_rate, verbose=False)
#scheduler = None
optim = paddle.optimizer.Adam(parameters = net.parameters(), 
                                learning_rate = scheduler or CONFIG.learning_rate)
losses = []
accs = []


# Logwriter Initialization
from visualdl import LogWriter
import os 
log_file = os.path.join(CONFIG.log_file, 
                        os.path.split(CONFIG.paddle_save_path)[-1].split('.')[0])
try:
    os.makedirs(log_file)
except FileExistsError:
    pass
writer = LogWriter(log_file)


######################################
#              TRAINING
######################################
n = 50000
epochs = CONFIG.epochs
for epoch in range(len(accs) + 1, epochs + len(accs) + 1):
    for x , labels in dataloader(train_x, train_y, CONFIG.train_size,
                                 cut = CONFIG.cut, mix = CONFIG.mix, onehot = True):
        y = net(x) 
        labels = paddle.to_tensor(labels, dtype = 'float32')
        loss = paddle.nn.CrossEntropyLoss(soft_label = True)(y, labels)
        losses.append(loss.item())
        optim.clear_grad()
        loss.backward()
        optim.step()
        if scheduler is not None:
            scheduler.step()
 
    # compute the accuracy on the testing data
    accs.append(test(net, test_x, test_y, CONFIG.test_size))

    print('Epochs =', epoch, ' Top1/5 Acc =', accs[-1])

    for i in range(len(losses) - n // CONFIG.train_size, len(losses)):
        writer.add_scalar(tag="train/loss", step = i, value = losses[i])  
    writer.add_scalar(tag="valid/top1 acc", step = len(accs), value = accs[-1][0]) 
    writer.add_scalar(tag="valid/top5 acc", step = len(accs), value = accs[-1][1]) 

    # save the model if better
    if accs[-1][0] >= max([i[0] for i in accs]):
        paddle.save(net.state_dict(), CONFIG.paddle_save_path)

        
writer.close()
