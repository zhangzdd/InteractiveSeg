import sys
from model import *
from torchvision.utils import save_image
import torchvision
import json
from u_net import *
raw_data, model_name,round = sys.argv[1:]
#submodel = SuggestiveAnnotationModel(n_input_channels=3, n_classes=3)
submodel = UNet(n_channels=3,n_classes=3)
#input = read from raw data
data_input = []
with open(raw_data) as f:
    jstr = json.load(f)
    #print(jstr)
    for original in jstr.keys():
        print(original)
        data_input = torchvision.io.read_image(original)
        print(data_input.shape)

#result = submodel(torch.rand((1, 3, 512, 512)))
#print(data_input.float())
result = submodel(data_input.float().unsqueeze(0))
save_image(result[0],"data/initial_label/model_{}_result.png".format(model_name))
torch.save(submodel.state_dict(),"models/round_{}/{}.ckpt".format(round,model_name))
