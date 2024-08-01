import sys
import numpy as np

if len(sys.argv) < 2:
    print("read_logs: error: no input files")
    sys.exit()
if len(sys.argv) > 2:
    print("read_logs: error: too many input files")
    sys.exit()
if '.npy' not in sys.argv[1]:
    print('read_logs: error: logs is not a ".npy" file')

logs = np.load(sys.argv[1], allow_pickle=True).item()
epochs = len(logs['t'])
for i in range(0, epochs):
    print(f'Epoch {i + 1}/{epochs}')
    print(
        f'batch_size: {logs["batch_size"][i]: d}',
        f'- resolution: {logs["resolution"][i]: d}',
        f'- dropout_rate: {logs["dropout_rate"][i]: .1f}',
        '\n',
        end=''
    )
    print(
        f'{logs["t"][i]: .0f}s',
        f'- loss: {logs["loss"][i]: .4f}',
        f'- accuracy: {logs["accuracy"][i]: .4f}',
        f'- val_loss: {logs["val_loss"][i]: .4f}',
        f'- val_accuracy: {logs["val_accuracy"][i]: .4f}',
        f'- lr: {logs["lr"][i]: g}',
        '\n',
        end=''
    )
    print('----')
print(f'total training time: {np.cumsum(logs["t"])[-1]}')
