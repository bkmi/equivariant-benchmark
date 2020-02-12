import os
import csv


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


def main():
    pass


if __name__ == '__main__':
    main()
