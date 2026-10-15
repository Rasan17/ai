"""
Exact forward-pass check of the app's neural network.

The app trains its real TensorFlow.js network (CPU backend, in Node via nn_forward_runner.js); this script
then verifies, independently in NumPy:
  1. Architecture: the layers TF.js actually built match the configuration requested (units, ReLU, dropout
     rate, L2 on hidden kernels, softmax/linear output, weight shapes).
  2. Forward pass: predictions recomputed from the exported weights (dense -> ReLU -> ... -> softmax/linear;
     dropout is inactive at inference) equal the network's raw outputs to float32 precision.
  3. App outputs: the probabilities/values the app displays equal the recomputed ones after its 4-decimal
     rounding, and the predicted class is the arg-max.

Usage (from the project folder; needs Node.js and Python with numpy + scikit-learn):
    python3 tests/nn_forward_check.py
"""
import json
import os
import subprocess
import sys

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'bench_work')
RAW_TOL = 1e-5      # float32 arithmetic (relative to max(1, |value|): float32 has ~7 significant digits)
APP_TOL = 5e-5      # the app rounds displayed probabilities / values to 4 decimals (+ float32 slack, relative)

CONFIG = {'hiddenLayers': [64, 32, 16], 'epochs': 60, 'learningRate': 0.005, 'batchSize': 16,
          'dropoutRate': 0.2, 'l2Reg': 1e-4, 'activation': 'relu', 'optimizer': 'adam', 'seed': 7}


def build_cases():
    cases = []
    for name, loader, task in [('breast_cancer', load_breast_cancer, 'binary'),
                               ('wine', load_wine, 'multiclass'),
                               ('diabetes', load_diabetes, 'regression')]:
        d = loader()
        X, y = d.data.astype(float), d.target
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=3,
                                  stratify=y if task != 'regression' else None)
        sc = StandardScaler().fit(X[tr])
        k = int(len(np.unique(y))) if task != 'regression' else 1
        cases.append({'name': name, 'task': task, 'numClasses': k, 'config': CONFIG,
                      'trainX': sc.transform(X[tr]).tolist(), 'testX': sc.transform(X[te]).tolist(),
                      'trainY': (y[tr].astype(int) if task != 'regression' else y[tr].astype(float)).tolist()})
    return cases


def forward(layers, X, task):
    """Independent NumPy forward pass using the exported weights. Dropout is inactive at inference."""
    h = np.asarray(X, dtype=np.float64)
    for L in layers:
        if L['className'] == 'Dropout':
            continue
        W = np.array(L['weights'][0]['values']).reshape(L['weights'][0]['shape'])
        b = np.array(L['weights'][1]['values']) if L['useBias'] else 0.0
        h = h @ W + b
        if L['activation'] == 'relu':
            h = np.maximum(h, 0)
        elif L['activation'] == 'softmax':
            e = np.exp(h - h.max(1, keepdims=True))
            h = e / e.sum(1, keepdims=True)
        elif L['activation'] != 'linear':
            raise ValueError(f"unexpected activation {L['activation']}")
    return h


def main():
    os.makedirs(WORK, exist_ok=True)
    cases = build_cases()
    inp, outp = os.path.join(WORK, 'nn_input.json'), os.path.join(WORK, 'nn_output.json')
    with open(inp, 'w') as f:
        json.dump({'cases': cases}, f)
    subprocess.run(['node', os.path.join(HERE, 'nn_forward_runner.js'), inp, outp], check=True)
    res = json.load(open(outp))

    rows = []
    for case, r in zip(cases, res['cases']):
        task, k = r['task'], r['numClasses']
        layers = r['layers']
        dense = [L for L in layers if L['className'] == 'Dense']
        drops = [L for L in layers if L['className'] == 'Dropout']
        hidden, output = dense[:-1], dense[-1]
        p = len(case['trainX'][0])

        # 1. Architecture
        exp_units = CONFIG['hiddenLayers'] + [k if task != 'regression' else 1]
        arch_ok = ([L['units'] for L in dense] == exp_units
                   and all(L['activation'] == 'relu' for L in hidden)
                   and output['activation'] == ('softmax' if task != 'regression' else 'linear')
                   and len(drops) == len(hidden) and all(abs(L['rate'] - CONFIG['dropoutRate']) < 1e-12 for L in drops)
                   and all(L['l2'] is not None and abs(L['l2'] - CONFIG['l2Reg']) < 1e-12 for L in hidden)
                   and output['l2'] is None
                   and dense[0]['weights'][0]['shape'] == [p, CONFIG['hiddenLayers'][0]])
        rows.append((r['name'], 'Architecture as built by TF.js matches the configuration',
                     ' → '.join(f"Dense({L['units']}, {L['activation']}{', L2' if L['l2'] else ''})" for L in dense)
                     + f"; {len(drops)} × Dropout({CONFIG['dropoutRate']})", 'PASS' if arch_ok else 'FAIL'))

        # 2. Forward pass vs raw network output
        mine = forward(layers, case['testX'], task)
        raw = np.array(r['raw']).reshape(mine.shape)
        scale = np.maximum(1.0, np.abs(mine))
        d_raw = float((np.abs(mine - raw) / scale).max())
        rows.append((r['name'], f'NumPy forward pass vs TF.js raw output ({len(raw)} test cases)',
                     f'max |diff|/max(1,|value|) = {d_raw:.1e} (tol {RAW_TOL:.0e})', 'PASS' if d_raw <= RAW_TOL else 'FAIL'))

        # 3. What the app shows
        if task != 'regression':
            app_p = np.array([[o['probabilities'][str(c)] for c in range(k)] for o in r['appOutputs']])
            app_pred = np.array([o['pred'] for o in r['appOutputs']])
            d_app = float(np.abs(app_p - mine).max())
            argmax_ok = bool(np.all(app_pred == np.argmax(mine, 1)))
            sums = np.abs(mine.sum(1) - 1).max()
            ok = d_app <= APP_TOL and argmax_ok and sums < 1e-9
            rows.append((r['name'], 'App-displayed probabilities and predicted class',
                         f'max |diff| = {d_app:.1e} (tol {APP_TOL:.0e}); class = arg-max: {"yes" if argmax_ok else "NO"}; '
                         f'probabilities sum to 1', 'PASS' if ok else 'FAIL'))
        else:
            app_v = np.array([o['pred'] for o in r['appOutputs']])
            d_app = float((np.abs(app_v - mine[:, 0]) / np.maximum(1.0, np.abs(mine[:, 0]))).max())
            rows.append((r['name'], 'App-displayed predicted values', f'max |diff|/max(1,|value|) = {d_app:.1e} (tol {APP_TOL:.0e})',
                         'PASS' if d_app <= APP_TOL else 'FAIL'))
        t = r['training']
        rows.append((r['name'], 'Training provenance (informational)',
                     f"{t['fitCases']} fit / {t['validationCases']} validation cases; "
                     + (f"early stop at epoch {t['stoppedEpoch']}, best epoch {t['bestEpoch']}" if t['earlyStopped']
                        else f"all {t['maxEpochs']} epochs, best epoch {t['bestEpoch']}"), 'INFO'))

    w = max(len(x[1]) for x in rows)
    print(f"TensorFlow.js {res['tfjs']} ({res['backend']} backend)")
    for name, check, result, verdict in rows:
        print(f'{verdict:<5} {name:<14} {check:<{w}}  {result}')
    n_fail = sum(1 for x in rows if x[3] == 'FAIL')
    n_pass = sum(1 for x in rows if x[3] == 'PASS')
    print(f'\n{n_pass}/{n_pass + n_fail} forward-pass checks passed')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
