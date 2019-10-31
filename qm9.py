import os
import numpy as np
from ase.db import connect

atomic_to_sym = {0: '-', 1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
sym_to_onehot = {'H': [1, 0, 0, 0, 0, 0],
                 'C': [0, 1, 0, 0, 0, 0],
                 'N': [0, 0, 1, 0, 0, 0],
                 'O': [0, 0, 0, 1, 0, 0],
                 'F': [0, 0, 0, 0, 1, 0],
                 '-': [0, 0, 0, 0, 0, 1]}
arg_to_atomic = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9, 5: 0}


def atomic_number_to_onehot(atomic_numbers):
    return [sym_to_onehot[atomic_to_sym[x]] for x in atomic_numbers]


def load_data(train_data: str, test_data: str, qm9db: str, ntrain: int, ntest: int, save_data: bool):
    if os.path.exists(train_data):
        print('Loading training data.')
        qm9 = {k: v for k, v in np.load(train_data, allow_pickle=True).items()}
    else:
        print('Querying training data.')
        qm9 = {'mol_id': [], 'numbers': [], 'positions': []}
        with connect(qm9db) as conn:
            for atoms in conn.select('4<natoms<=18', limit=ntrain):
                qm9['mol_id'].append(atoms.mol_id)
                qm9['positions'].append(atoms.positions)
                qm9['numbers'].append(atoms.numbers)
                qm9['dmats'].append(atoms.data['dmats'])

        if save_data:
            print('Saving training data.')
            np.savez(train_data, **qm9)

    if os.path.exists(test_data):
        print('Loading test data.')
        qm9_test = {k: v for k, v in np.load(test_data, allow_pickle=True).items()}
    else:
        print('Querying test data.')
        qm9_test = {'mol_id': [], 'numbers': [], 'positions': []}
        with connect(qm9db) as conn:
            for atoms in conn.select('natoms=19', limit=ntest):
                qm9_test['mol_id'].append(atoms.mol_id)
                qm9_test['positions'].append(atoms.positions)
                qm9_test['numbers'].append(atoms.numbers)
        if save_data:
            print('Saving test data.')
            np.savez(test_data, **qm9_test)
    return qm9, qm9_test
