import sys
from model import *
from torchvision.utils import save_image

raw_data, model_name,round = sys.argv[1:]
submodel = SuggestiveAnnotationModel(n_input_channels=3, n_classes=3)
#input = read from raw data
result = submodel(torch.rand((1, 3, 512, 512)))
save_image(result[0],"data/initial_label/{}_result.png".format(model_name))
torch.save(submodel.state_dict(),"models/round_{}/{}.ckpt".format(round,model_name))
