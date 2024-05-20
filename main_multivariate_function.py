import os
import csv
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import reproduc, mymodel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset, random_split

parser = argparse.ArgumentParser(description='Training Configuration')
parser.add_argument('--work_dir', type=str, default='./work_dirs', help='Directory to save model weights and logs')
parser.add_argument('--act_fun', type=str, default='ReLU', help='Activation function')
parser.add_argument('--Dataset', type=str, default='LeakyReLU', help='Activation function')
parser.add_argument('--seed', type=int, default=0, help='Seed for the code')
args = parser.parse_args()
work_dir = args.work_dir
reproduc(args.seed)

if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)

num_data = 50000
batch_size = 128
shuffle = True

# Load data from saved .pt files
if args.Dataset == 'ReLU':
    try:
        input_tensor = torch.load(f"{work_dir}/data/input_data_relu.pt")
        output_tensor = torch.load(f"{work_dir}/data/output_data_relu.pt")
    except:
        raise ValueError("Data for ReLU not generated. Please generate data first.")
    lr = 0.001
elif args.Dataset == 'LeakyReLU':
    try:
        input_tensor = torch.load(f"{work_dir}/data/input_data_leaky.pt")
        output_tensor = torch.load(f"{work_dir}/data/output_data_leaky.pt")
    except:
        raise ValueError("Data for Leaky ReLU not generated. Please generate data first.")
    lr = 0.001
elif args.Dataset == 'soc_2dim':
    try:
        input_tensor = torch.load(f"{work_dir}/data/input_data_soc.pt")
        output_tensor = torch.load(f"{work_dir}/data/output_data_soc.pt")
    except:
        raise ValueError("Data for Leaky ReLU not generated. Please generate data first.")
    lr = 0.0005
else:
    raise ValueError("Function chosen not implemented! Please choose within ReLU, LeakyReLU and soc_2dim.")

# output_tensor = output_tensor.unsqueeze(1)

# Transform into PyTorch Dataset
dataset = TensorDataset(input_tensor, output_tensor)

# Define DataLoader with the dataset, specifying batch size and shuffle option
train_len = int(0.8 * len(dataset))  # 80% for training
val_len = len(dataset) - train_len  # 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

# Create DataLoaders for both sets
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

hidden_list = np.arange(2, 40)
acc_list = np.array([])

for hidden_dimension in hidden_list:
    # Initialize the model
    model = mymodel(hidden_dimension, act_fun=args.act_fun).cuda()

    # Initialize the optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Initialize the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    criterion = nn.MSELoss()

    # Initialize variable to track best validation loss
    best_val_loss = float('inf')

    for epoch in range(50):  # Assuming 50 epochs
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (input_batch, output_batch) in enumerate(train_dataloader):
            optimizer.zero_grad()
            predictions = model(input_batch.cuda())
            output_batch = output_batch
            loss = criterion(predictions, output_batch.cuda())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
    
        # Evaluation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_batch, output_batch in val_dataloader:
                predictions = model(input_batch.cuda())
                output_batch = output_batch
                loss = criterion(predictions, output_batch.cuda())
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Step the scheduler
        scheduler.step()
    
        # Print or log the losses for the epoch
        print(f"Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

    acc_list = np.append(acc_list, best_val_loss)

# Check if the file exists
if not os.path.exists(f'{work_dir}/acc__act_fun{args.act_fun}_dataset_{args.Dataset}.csv'):
    # Create the file if it doesn't exist
    with open(f'{work_dir}/acc_act_fun_{args.act_fun}_dataset_{args.Dataset}.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(acc_list.tolist())
