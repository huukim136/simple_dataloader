from pathlib import Path
from typing import Union
import pdb
import random
## why '\n' is not necessary?

def get_files(path: Union[str, Path], extension='.npy'):
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))

wav1 = get_files('../TIMIT',extension='.npy')
wav2 = get_files('../gen_data_full',extension='.npy')
wav = wav1 + wav2
# pdb.set_trace()
random.shuffle(wav)
length = len(wav)
bound = int(0.95*length)
# bound = length

with open('timit_gen_train_full.txt',"w") as f:
    for i in range(bound-100):
        # pdb.set_trace()
        f.write(str(wav[i]))
        f.write('\n')   ##
        # txt_f.close()

with open('timit_gen_val_full.txt',"w") as f:
    for i in range(bound-100, bound):
        # pdb.set_trace()
        f.write(str(wav[i]))
        f.write('\n')   ##

with open('timit_gen_test_full.txt',"w") as f:
    for i in range(bound, length):
        # pdb.set_trace()
        f.write(str(wav[i]))
        f.write('\n')   ##

# wav = get_files('../TIMIT/test',extension='.npy')
# length = len(wav)
# with open('timit_test.txt',"w") as f:
#     for i in range(length):
#         # pdb.set_trace()
#         f.write(str(wav[i]))
#         f.write('\n')   ##
