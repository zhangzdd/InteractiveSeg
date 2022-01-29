import sys
import os
import torch 
import torchvision
from skimage.io import imsave

labelled_img_list, assimilated_results_filename  = sys.argv[1:]
with open("logs/{}".format(labelled_img_list)) as f:
    label_path = f.readlines() 
    print(label_path)
    tensor_batch = []
    for pth in label_path:
        img_tensor = torchvision.io.read_image("data/initial_label/{}".format(pth[:-1]))
        tensor_batch.append(img_tensor)

_batch = torch.stack(tensor_batch)
#print(_batch.shape)
var_tensor = torch.var(_batch/_batch.max(),dim=0)
print(var_tensor.shape)
#print(var_tensor)
cumulative_var_tensor = torch.sum(var_tensor,dim = 0)
print(cumulative_var_tensor.shape)
torchvision.utils.save_image(cumulative_var_tensor,"data/variance/uncertainty.png")
#imsave(cumulative_var_tensor,"data/variance/uncertainty.png")