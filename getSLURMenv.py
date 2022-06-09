# Load and print the contents of the env variable SBATCH_CPU_IDS, SBATCH_GPU_IDS and SBATCH_CPU_BIND_LIST
#
import os


# get the cpu ids and gpu ids from the environment variables
# def get_cpu_gpu_ids(env_var_name):
#     cpu_gpu_ids = os.environ.get(env_var_name)
#     if cpu_gpu_ids is None:
#         print(f"Error: environment variable {env_var_name} is not set")
#         sys.exit(1)
#     # split the string and get the cpu ids
#     cpu_gpu_ids = cpu_gpu_ids.split(",")
#     cpu_gpu_ids = [i.split("-") for i in cpu_gpu_ids]
#     cpu_gpu_ids = [
#         [int(i) for i in cpu_gpu_ids[0]] + [int(i) for i in cpu_gpu_ids[1]]
#     ]
#     return cpu_gpu_ids

print('cpu ids:', os.environ.get('SBATCH_CPU_IDS'))
print('gpu ids:', os.environ.get('SBATCH_GPU_IDS'))
print('cpu bind list:', os.environ.get('SBATCH_CPU_BIND_LIST'))
