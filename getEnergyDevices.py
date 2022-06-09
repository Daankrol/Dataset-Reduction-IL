from pyJoules.energy_meter import measure_energy, EnergyMeter
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.device_factory import DeviceFactory

# domains = [RaplPackageDomain(0), NvidiaGPUDomain(0)]
devices = DeviceFactory.create_devices()
# meter = EnergyMeter(devices)
print(devices)


# to get ids of cpus assigned: scontrol -dd show job $SLURM_JOB_ID | grep "CPU_IDs" 
# to get ids of gpus assigned: scontrol -dd show job $SLURM_JOB_ID | grep "GPU_IDs"


# We get the CPU ids and GPU ids within the job with running the following command:
# scontrol -dd show job $SLURM_JOB_ID | grep "CPU_IDs". This returns a string like:
# Nodes=research1 CPU_IDS=22-27,42-43 Mem=102400 GRES=gpu(IDX:5)
# We need to extract the CPU ids and GPU ids from this string. 