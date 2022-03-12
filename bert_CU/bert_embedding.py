import os
import numpy as np
from bert_CU.bert.main import ImportBert
from rich.progress import track
from multiprocessing import  Process


def bert_embedding(root, saved_path, L):
    script = open(root, 'r')
    lines = sorted([line for line in script.readlines()])


    # 
    BERT = ImportBert()
    print('running start......')

    os.makedirs(saved_path, exist_ok=True)
    embedding(saved_path, lines, L, BERT)
    # process_list = []
    # step = int(len(lines)/10)
    # for i in range(10):
    #     if i==9:
    #         print(i*step)
    #         p = Process(target=embedding, args=(saved_path, lines[i*step:i*step+step], L, BERT))
    #     else:
    #         print(i*step,i*step+step)
    #         p = Process(target=embedding, args=(saved_path, lines[i*step:], L, BERT))
    #     process_list.append(p)
    #
    # for p in process_list:
    #     p.start()
    # for p in process_list:
    #     p.join()

def embedding(out_path, lines, L, BERT):
    print('pid {} is runing.'.format(os.getpid()))
    for i in range(L):
        lines.insert(0, lines[0])
        lines.append(lines[-1])
    for i in range(L,len(lines)-L):
        if i%500==0:
            print(i)
        tmp_CU = []
        for k in range(i-L,i+L):
            tmp_CU += [BERT.infer(lines[k].split('|')[-1],lines[k+1].split('|')[-1])[0]]

        name = lines[i].split('|')[0]
        speaker = lines[i].split('|')[1]
        np.save(os.path.join(out_path, speaker + '-embeds-' + name + '.npy'), tmp_CU)


def pan_list(A):
    max_len = max([l.shape[0] for l in A])
    return np.stack(np.pad(l, [(max_len-l.shape[0], 0),(0, 0)], mode='constant', constant_values=0) for l in A)