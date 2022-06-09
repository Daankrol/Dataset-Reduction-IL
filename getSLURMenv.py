import logging

def get_cpu_ids():
    return os.environ.get('SBATCH_CPU_IDS')


def get_gpu_ids():
    return os.environ.get('SBATCH_GPU_IDS')


def get_cpu_bind_list():
    return os.environ.get('SBATCH_CPU_BIND_LIST')


def get_gpu_bind_list():
    return os.environ.get('SBATCH_GPU_BIND_LIST')


# now call all functions and log it in a file 
logging.basicConfig(filename='slurm_envs.log', level=logging.INFO)
logging.info(f'cpu ids: {get_cpu_ids()}')
logging.info(f'gpu ids: {get_gpu_ids()}')
logging.info(f'cpu bind list: {get_cpu_bind_list()}')
logging.info(f'gpu bind list: {get_gpu_bind_list()}')

