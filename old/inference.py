import torch
from train_model_supervised import SUPERVISED_ADD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SUPERVISED_ADD(device=device, load_model=True, mode='eval')
model.load_model_cpt(cpt=<checkpoint_number>, device=device)
file_path = 'inference_input.wav'
result = model.infer(file_path)
print(result)

