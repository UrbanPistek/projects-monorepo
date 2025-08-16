#!/usr/bin/env python3
"""
Complete System Information Script
Collects comprehensive system information including GPU details
"""

import platform
import psutil
import socket
import subprocess
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

def get_size(bytes_value: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

def get_basic_system_info() -> Dict[str, Any]:
    """Get basic system information"""
    uname = platform.uname()
    boot_time = datetime.fromtimestamp(psutil.boot_time())
    
    return {
        "System": uname.system,
        "Node Name": uname.node,
        "Release": uname.release,
        "Version": uname.version,
        "Machine": uname.machine,
        "Processor": uname.processor,
        "Architecture": platform.architecture()[0],
        "Boot Time": boot_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Python Version": platform.python_version(),
        "Platform": platform.platform()
    }

def get_cpu_info() -> Dict[str, Any]:
    """Get detailed CPU information"""
    cpu_freq = psutil.cpu_freq()
    cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
    
    info = {
        "Physical Cores": psutil.cpu_count(logical=False),
        "Total Cores": psutil.cpu_count(logical=True),
        "Max Frequency": f"{cpu_freq.max:.2f} MHz" if cpu_freq else "N/A",
        "Min Frequency": f"{cpu_freq.min:.2f} MHz" if cpu_freq else "N/A",
        "Current Frequency": f"{cpu_freq.current:.2f} MHz" if cpu_freq else "N/A",
        "Total CPU Usage": f"{psutil.cpu_percent()}%",
        "Per Core Usage": [f"Core {i}: {usage}%" for i, usage in enumerate(cpu_usage)]
    }
    
    return info

def get_memory_info() -> Dict[str, Any]:
    """Get memory information"""
    svmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    return {
        "Total": get_size(svmem.total),
        "Available": get_size(svmem.available),
        "Used": get_size(svmem.used),
        "Percentage Used": f"{svmem.percent}%",
        "Swap Total": get_size(swap.total),
        "Swap Used": get_size(swap.used),
        "Swap Free": get_size(swap.free),
        "Swap Percentage": f"{swap.percent}%"
    }

def get_disk_info() -> List[Dict[str, Any]]:
    """Get disk information"""
    partitions = psutil.disk_partitions()
    disk_info = []
    
    for partition in partitions:
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
            disk_info.append({
                "Device": partition.device,
                "Mountpoint": partition.mountpoint,
                "File System": partition.fstype,
                "Total Size": get_size(partition_usage.total),
                "Used": get_size(partition_usage.used),
                "Free": get_size(partition_usage.free),
                "Percentage Used": f"{(partition_usage.used / partition_usage.total) * 100:.1f}%"
            })
        except PermissionError:
            # This can happen on Windows
            continue
    
    return disk_info


def get_gpu_info_nvidia() -> List[Dict[str, Any]]:
    """Get NVIDIA GPU information using nvidia-smi"""
    gpus = []
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,power.draw', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        for line in result.stdout.strip().split('\n'):
            if line:
                values = [v.strip() for v in line.split(',')]
                if len(values) >= 9:
                    gpus.append({
                        "Index": values[0],
                        "Name": values[1],
                        "Driver Version": values[2],
                        "Memory Total": f"{values[3]} MB",
                        "Memory Used": f"{values[4]} MB",
                        "Memory Free": f"{values[5]} MB",
                        "Temperature": f"{values[6]}°C" if values[6] != '[Not Supported]' else 'N/A',
                        "GPU Utilization": f"{values[7]}%" if values[7] != '[Not Supported]' else 'N/A',
                        "Power Draw": f"{values[8]}W" if values[8] != '[Not Supported]' else 'N/A'
                    })
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return gpus


def get_gpu_info_pytorch() -> Dict[str, Any]:
    """Get GPU information using PyTorch"""
    gpu_info = {"PyTorch Available": False}
    
    try:
        import torch
        gpu_info["PyTorch Available"] = True
        gpu_info["PyTorch Version"] = torch.__version__
        gpu_info["CUDA Available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            gpu_info["CUDA Version"] = torch.version.cuda
            gpu_info["CUDNN Version"] = torch.backends.cudnn.version()
            gpu_info["GPU Count"] = torch.cuda.device_count()
            
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "Device": i,
                    "Name": props.name,
                    "Total Memory": get_size(props.total_memory),
                    "Multi Processor Count": props.multi_processor_count,
                    "CUDA Capability": f"{props.major}.{props.minor}"
                })
            gpu_info["Devices"] = devices
    except ImportError:
        pass
    
    return gpu_info

def print_section(title: str, data: Any, indent: int = 0) -> None:
    """Print a section with proper formatting"""
    prefix = "  " * indent
    print(f"{prefix}{'='*20} {title} {'='*20}")
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                print(f"{prefix}{key}:")
                print_section("", value, indent + 1)
            else:
                print(f"{prefix}{key}: {value}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                print(f"{prefix}Item {i + 1}:")
                print_section("", item, indent + 1)
            else:
                print(f"{prefix}- {item}")
    else:
        print(f"{prefix}{data}")
    print()

def main():
    """Main function to collect and display all system information"""
    print("SYSTEM INFORMATION REPORT")
    print("=" * 60)
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    # Basic System Info
    print_section("BASIC SYSTEM INFO", get_basic_system_info())
    
    # CPU Info
    print_section("CPU INFO", get_cpu_info())
    
    # Memory Info
    print_section("MEMORY INFO", get_memory_info())
    
    # Disk Info
    print_section("DISK INFO", get_disk_info())
    
    # GPU Info - NVIDIA
    nvidia_gpus = get_gpu_info_nvidia()
    if nvidia_gpus:
        print_section("NVIDIA GPU INFO", nvidia_gpus)
    else:
        print_section("NVIDIA GPU INFO", "No NVIDIA GPUs found or nvidia-smi not available")
    
    # GPU Info - PyTorch
    print_section("PYTORCH GPU INFO", get_gpu_info_pytorch())
    
    print("=" * 60)
    print("System information collection complete!")

if __name__ == "__main__":
    main()