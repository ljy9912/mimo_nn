import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from utils import reproduc
from cone_projection import soc_2dim_with_angle

parser = argparse.ArgumentParser(description='Dataset selection')
parser.add_argument('--work_dir', type=str, default='./work_dirs/', help='Directory to save model weights and logs')
parser.add_argument('--fun', type=str, default='ReLU', help='Function to be approximated')
parser.add_argument('--seed', type=int, default=0, help='Seed for the code')
args = parser.parse_args()
work_dir = args.work_dir
if not os.path.isdir(work_dir):
    os.mkdir(work_dir)
if not os.path.isdir(f'{work_dir}/data'):
    os.mkdir(f'{work_dir}/data')
reproduc(args.seed)

# Initialize arrays to store data
angle = np.pi / 3
data_num = 50000
input_data = np.zeros((data_num, 2))
output_data = np.zeros((data_num, 1))

# Generate random input data
input_data[:, 0] = np.random.uniform(-10, 10, data_num)
input_data[:, 1] = np.random.uniform(-10, 10, data_num)

if args.fun == 'ReLU':
    fun = nn.ReLU()
elif args.fun == 'LeakyReLU':
    fun = nn.LeakyReLU()
elif args.fun == 'soc_2dim':
    fun = soc_2dim_with_angle(angle)
else:
    raise ValueError("Function chosen not implemented! Please choose within ReLU, LeakyReLU and soc_2dim.")

A1 = torch.from_numpy(np.array([[0.9016, 0.1787], [0.0437, 0.5363]]))
A1 /= torch.linalg.norm(A1, 2)
A2 = torch.from_numpy(np.array([[0.5426, 0.4668], [0.4043, 0.9253]]))
A2 /= torch.linalg.norm(A2, 2)

input_data = torch.from_numpy(input_data)
output_data = torch.bmm(A1.unsqueeze(0).repeat(data_num, 1, 1), input_data.unsqueeze(2)).squeeze(2)
output_data = fun(output_data)
output_data = torch.bmm(A2.unsqueeze(0).repeat(data_num, 1, 1), output_data.unsqueeze(2)).squeeze(2).float()
input_data = input_data.float()

# Save as .pt files
if args.fun == 'ReLU':
    torch.save(input_data, f"{work_dir}/data/input_data_relu.pt")
    torch.save(output_data, f"{work_dir}/data/output_data_relu.pt")
elif args.fun == 'LeakyReLU':
    torch.save(input_data, f"{work_dir}/data/input_data_leaky.pt")
    torch.save(output_data, f"{work_dir}/data/output_data_leaky.pt")
elif args.fun == 'soc_2dim':
    torch.save(input_data, f"{work_dir}/data/input_data_soc.pt")
    torch.save(output_data, f"{work_dir}/data/output_data_soc.pt")

