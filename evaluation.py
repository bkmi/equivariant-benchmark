import os
import csv
import subprocess
import schnetpack
import datetime

import torch

import e3nn


def evaluate(
    model_dir,
    model,
    loaders: dict,
    device,
    metrics,
    csv_file="evaluation.csv",
    results_file="results.pkl",
    custom_header=None,
):
    header = []
    aggregated = []
    results = []

    for key, loader in loaders.items():
        header += [f"{key} {metric.name}" for metric in metrics]
        agg, result = evaluate_dataset(metrics, model, loader, device)
        aggregated.append(agg)
        results.extend(result)

    if custom_header:
        header = custom_header

    csv_file = os.path.join(model_dir, csv_file)
    with open(csv_file, "w") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerow(aggregated)

    out_results = {}
    for item in results:
        for k, v in item.items():
            try:
                out_results[k] = torch.cat([out_results[k], v])
            except KeyError:
                out_results[k] = v

    results_file = os.path.join(model_dir, results_file)
    torch.save(out_results, results_file)


def evaluate_dataset(metrics, model, loader, device):
    for metric in metrics:
        metric.reset()

    model.requires_grad_(False)
    model = model.to(device)
    results = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        result = model(batch)
        results.append(result)

        for metric in metrics:
            metric.add_batch(batch, result)

    aggregated = [metric.aggregate() for metric in metrics]
    return aggregated, results


def record_versions(record):
    if os.path.exists(record):
        raise FileExistsError(f"{record} already exists.")

    with open(record, 'w', newline='\n') as f:
        f.write(str(datetime.datetime.now()) + "\n")
        f.write(str(os.uname()[1]))

        for file in [__file__, e3nn.__file__, schnetpack.__file__]:
            directory = os.path.dirname(file)
            f.write(str(directory) + '\n')
            commands = [
                f"git -C {directory} rev-parse HEAD",
                f"git -C {directory} branch -vv",
                f"git -C {directory} remote -v"
            ]
            for command in commands:
                f.write(subprocess.run(command.split(), capture_output=True).stdout.decode() + '\n')


def main():
    record_versions("test.txt")


if __name__ == '__main__':
    main()
