import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
from model import init_model_tuned
import torch.nn as nn

model_name = "resnet_v0"

model = init_model_tuned(train=False, model_name=model_name)
#model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.load_state_dict(torch.load('./models/model_{}.pth'.format(model_name)))

model.eval()
#torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("../ClassifierApp/app/src/main/assets/{}.pt".format(model_name))
