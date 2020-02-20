import os
import csv
import subprocess
import schnetpack
import datetime

import e3nn


def evaluate(
    model_dir,
    model,
    loaders: dict,
    device,
    metrics,
    custom_header=None,
):
    header = []
    results = []

    for key, loader in loaders.items():
        header += [f"{key} {metric.name}" for metric in metrics]
        results += evaluate_dataset(metrics, model, loader, device)

    if custom_header:
        header = custom_header

    eval_file = os.path.join(model_dir, "evaluation.csv")
    with open(eval_file, "w") as file:
        wr = csv.writer(file)
        wr.writerow(header)
        wr.writerow(results)


def evaluate_dataset(metrics, model, loader, device):
    for metric in metrics:
        metric.reset()

    model = model.to(device)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        result = model(batch)

        for metric in metrics:
            metric.add_batch(batch, result)

    results = [metric.aggregate() for metric in metrics]
    return results


def record_versions(record):
    if os.path.exists(record):
        raise FileExistsError(f"{record} already exists.")

    with open(record, 'w', newline='\n') as f:
        f.write(str(datetime.datetime.now()) + "\n")

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
