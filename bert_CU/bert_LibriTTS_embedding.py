import os
import numpy as np
from bert_CU.bert.main import ImportBert
from rich.progress import track
import multiprocessing


def bert_embedding(root, saved_path, L):
    script = open(root, 'r')
    lines = sorted([line for line in script.readlines()])

    for i in range(L):
        lines.insert(0, lines[0])
        lines.append(lines[-1])
    # 
    BERT = ImportBert()
    print('running start......')
    embedding(saved_path, lines, L, BERT)

def embedding(out_path, lines, L, BERT):
    os.makedirs(out_path, exist_ok=True)
    for i in range(L,len(lines)-L):
        if i%500==0:
            print(i)
        tmp_CU = []
        name = lines[i].split('\t')[0]
        if name != 'none':
            for k in range(i-L,i+L):
                tmp_CU += [BERT.infer(lines[k].split('|')[-1],lines[k+1].split('|')[-1])[0]]

            speaker = name.split('_')[0]
            np.save(os.path.join(out_path, speaker + '-embeds-' + name + '.npy'), tmp_CU)


def pan_list(A):
    max_len = max([l.shape[0] for l in A])
    return np.stack(np.pad(l, [(max_len-l.shape[0], 0),(0, 0)], mode='constant', constant_values=0) for l in A)