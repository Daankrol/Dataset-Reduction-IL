from pyJoules.energy_meter import measure_energy, EnergyMeter
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler
from pyJoules.device.device_factory import DeviceFactory

# domains = [RaplPackageDomain(0), NvidiaGPUDomain(0)]
devices = DeviceFactory.create_devices()
# meter = EnergyMeter(devices)
print(devices)
