import csv
import numpy as np
import os  

def append_result(args, result):
    directory = args.method
    filename = f"logs/{directory}.csv"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_exists = os.path.exists(filename)
    if not file_exists:
        print(f"{filename}not exist")

    with open(filename, "a", newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Dataset","Seed","r","Experiment", "Result"])

        for i, res in enumerate(result, start=1):
            writer.writerow([args.dataset, args.seed,args.reduction_rate,i, res])
