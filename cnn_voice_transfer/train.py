import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable
from utils import *
from model import *
import time
import math
import torch
from tensorboardX import SummaryWriter
# Writer will output to ./runs/ directory by default
writer = SummaryWriter()
cuda = True if torch.cuda.is_available() else False

#loading model function
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

basepath = "input/"

CONTENT_FILENAME = basepath + "Physics_WFE.mp3"
STYLE_FILENAME = basepath + "AB1_E.mp3"

a_content, sr = wav2spectrum(CONTENT_FILENAME)
a_style, sr = wav2spectrum(STYLE_FILENAME)

a_content_torch = torch.from_numpy(a_content)[None, None, :, :]
if cuda:
    a_content_torch = a_content_torch.cuda()
print(a_content_torch.shape)
a_style_torch = torch.from_numpy(a_style)[None, None, :, :]
if cuda:
    a_style_torch = a_style_torch.cuda()
print(a_style_torch.shape)
#uncommment to load and save.
#model = load_checkpoint('Checkpoint/checkpoint.pth')
model = RandomCNN()

#data parallelisation
if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model, device_ids=[0,1])
model.eval()
#Creating device handles for storing device handles
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

a_C_var = Variable(a_content_torch, requires_grad=False).float()
a_S_var = Variable(a_style_torch, requires_grad=False).float()
if cuda:
    model = model.cuda()
    a_C_var = a_C_var.cuda()
    a_S_var = a_S_var.cuda()

a_C = model(a_C_var)
a_C = a_C.to(device)
a_S = model(a_S_var)
a_S = a_S.to(device)


# Optimizer
learning_rate = 0.002
a_G_var = Variable(torch.randn(a_content_torch.shape).cuda() * 1e-3, requires_grad=True)
optimizer = torch.optim.Adam([a_G_var])
checkpoint = {'model' : model,
              'state_dict' : model.state_dict (),
              'optimizer' : optimizer.state_dict ()}
# coefficient of content and style
style_param = 0.8
content_param = 1e2

num_epochs = 50000
print_every = 1
plot_every = 100

# Keep track of losses for plotting
current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()
# Train the Model
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad()
    a_G = model(a_G_var)
    a_G = a_G.to(device)

    content_loss = content_param * compute_content_loss(a_C, a_G)
    style_loss = style_param * compute_layer_style_loss(a_S, a_G)
    loss = content_loss + style_loss
    loss.backward()
    optimizer.step()

    # print
    if epoch % print_every == 0:
        print("{} {}% {} content_loss:{:4f} style_loss:{:4f} total_loss:{:4f}".format(epoch,
                                                                                      epoch / num_epochs * 100,
                                                                                      timeSince(start),
                                                                                      content_loss.item(),
                                                                                      style_loss.item(), loss.item()))
        current_loss += loss.item()
        writer.add_scalar ('Training Loss', current_loss, epoch)
        #writer.flush()
        #writer.flush()
        #writer.flush ()
    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
        torch.save (checkpoint, 'Checkpoint/checkpoint.pth')





    if epoch % 1000==0:
        gen_spectrum = a_G_var.cpu().data.numpy().squeeze()
        gen_audio_C = "AmitabhBachchanCloned.wav"
        spectrum2wav(gen_spectrum, sr, gen_audio_C)

"""Loading a model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

model = load_checkpoint('checkpoint.pth')
"""

plt.figure()
plt.plot(all_losses)
plt.savefig('loss_curve.png')

plt.figure(figsize=(5, 5))
# we then use the 2nd column.
plt.subplot(1, 1, 1)
plt.title("Content Spectrum")
plt.imsave('Content_Spectrum.png', a_content[:400, :])

plt.figure(figsize=(5, 5))
# we then use the 2nd column.
plt.subplot(1, 1, 1)
plt.title("Style Spectrum")
plt.imsave('Style_Spectrum.png', a_style[:400, :])

plt.figure(figsize=(5, 5))
# we then use the 2nd column.
plt.subplot(1, 1, 1)
plt.title("CNN Voice Transfer Result")
plt.imsave('Gen_Spectrum.png', gen_spectrum[:400, :])

writer.close()
