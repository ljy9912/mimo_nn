
# Code for MIMO Activation Function

## Requirements
The code is written in pure python, and users can install the required package via the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Running Experiments
### Multivariate Function Fitting via FNN
The main code is in the file `main_multivariate_function.py`.
For the experiments in the paper, you can run
```bash
./train_multivariate_function.sh
```

For a single training process, first generate dataset by running:
```bash
python data_generation.py --work_dir "./work_dirs" --fun "soc_2dim" --seed 0
```
The parameters and the corresponding options are:
- work_dir: the directory that the generated dataset lies in. For example, if we specify the work_dir to be "./work_dirs", then the dataset lies under the path "./work_dirs/data/xxx.pt".
- fun: The function to be approximated. Options: ReLU, LeakyReLU, soc_2dim (projection to the 2-dimensional cone).
- seed: The seed for the experiment.

Then, you can run the main file. For example,
```bash
python main.py --work_dir "./work_dirs" --act_fun "ReLU" --Dataset "soc_2dim"
```
The parameters and the corresponding options are:
- work_dir: The directory to read dataset and store results. Make sure that the dataset lies under the directory "{work_dir}/data/xxx.pt", otherwise the code will go into trouble.
- act_fun: The activation function that the FNN utilize. Options: ReLU, LeakyReLU, PReLU, soc_2dim (projection to the 2-dimensional cone).

### ResNet on CIFAR10
The main code is in the file `main_resnet.py`.
For the experiments in the paper, you can run
```bash
./train_resnet.sh
```

For a single training process, you can run the main file:
```bash
# For cone projection
python data_generation.py --work_dir "./work_dirs" --act_fun "soc_2dim" --angle_tan "0.84" --seed 0
# For other activation functions
python data_generation.py --work_dir "./work_dirs" --act_fun "ReLU" --seed 0
```
The parameters and the corresponding options are:
- work_dir: The directory to store results.
- act_fun: The activation function that the FNN utilize. Options: ReLU, LeakyReLU, PReLU, soc_2dim (projection to the 2-dimensional cone), soc_2dim_leaky (leaky version of projection to the 2-dimensional cone), soc (projection to the 3-dimensional cone).
- angle_tan: Tangent value of the cone's half-apex angle. Only valid for soc_2dim, soc_2dim_leaky and soc.