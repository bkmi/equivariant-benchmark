import os
import pandas as pd


def extract_evaluation(file):
    df = pd.read_csv(file)
    extracted_data = {}
    for column in df.columns:
        mode = column.split()[0]
        error_type = column.split()[1].split("_")[0]
        target = "_".join(column.split()[1].split("_")[1:])
        if mode == 'test':
            assert len(df[column]) == 1
            try:
                extracted_data[target][error_type] = df[column][0]
            except KeyError:
                extracted_data[target] = {error_type: df[column][0]}
    return extracted_data


def extract_evaluations(parent):
    def all_evaluations(p):
        sub_directories = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
        return [os.path.join(p, d, 'evaluation.csv') for d in sub_directories]

    return {
        target: result
        for evaluation in all_evaluations(parent)
        for target, result in extract_evaluation(evaluation).items()
    }


def main():
    parent_directories = [
        # 'big',
        'big_l1',
        'big3',
        'big3_l1'
    ]
    names = [
        'l0',
        'l0 & l1',
        'l0 shallow',
        'l0 & l1 shallow',
    ]
    data = {}
    for name, parent in zip(names, parent_directories):
        data[name] = extract_evaluations(parent)

    df = pd.DataFrame.from_dict({(i, j): data[i][j]
                                 for i in data.keys()
                                 for j in data[i].keys()},
                                orient='index').swaplevel().sort_index()
    df.round(3).to_csv('results.csv')
    print('done')


if __name__ == '__main__':
    main()
