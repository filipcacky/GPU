#!/usr/bin/env python3

import os
import sys
import subprocess
import pandas as pd
from dataclasses import dataclass

if len(sys.argv) != 3:
    print("Invalid arguments")
    exit(1)

files = os.listdir(sys.argv[1])

task_dict = dict()


@dataclass
class Task:
    a: str
    b: str
    size: int


for file in files:
    if file.startswith('.'):
        continue
    parts = file.split('_')
    size = int(parts[0])
    body = str.join("_", parts[1:-1])
    ending = parts[-1].split('.')[0]

    if body not in task_dict.keys():
        task_dict[body] = Task("", "", size)

    if ending == 'A':
        task_dict[body].a = os.path.join(sys.argv[1], file)
        pass
    else:
        task_dict[body].b = os.path.join(sys.argv[1], file)
        pass

tasks = sorted(task_dict.values(), key=lambda x: x.size)

result = []

for task in tasks:
    cpu_params = [sys.argv[2],
                  '--lhs', task.a,
                  '-c', '0']
    gpu_params = [sys.argv[2],
                  '--lhs', task.a,
                  '-c', '1']

    if len(task.b) > 0:
        cpu_params.append('--rhs')
        cpu_params.append(task.b)
        gpu_params.append('--rhs')
        gpu_params.append(task.b)

    proc = subprocess.Popen(cpu_params, stdout=subprocess.PIPE)
    proc.wait()
    line = bytes.decode(proc.stdout.readline())
    try:
        iters_cpu, ns_cpu, mean_nonzero = line.strip().split(' ')
    except:
        print(cpu_params)
        print(line)
        exit(1)

    proc = subprocess.Popen(gpu_params, stdout=subprocess.PIPE)
    proc.wait()
    line = bytes.decode(proc.stdout.readline())
    try:
        iters_gpu, ns_gpu, mean_nonzero = line.strip().split(' ')
    except:
        print(gpu_params)
        print(line)
        exit(1)

    print(f"{task.a} cpu: {ns_cpu} gpu: {ns_gpu} speedup: {int(ns_cpu) / int(ns_gpu)} mean_nonzero: {mean_nonzero}")

    result.append({"A": task.a, "b": task.b, "size": task.size, "mean_nonzero": mean_nonzero,
                   "iters_cpu": iters_cpu, "ns_cpu": ns_cpu,
                   "iters_gpu": iters_gpu, "ns_gpu": ns_gpu,
                   "speedup": int(ns_cpu)/int(ns_gpu)})

result = pd.DataFrame(result)

print(result)

result.to_csv("./bench.txt")
