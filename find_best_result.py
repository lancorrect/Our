import os
import shutil

best_acc = 0
best_f1 = 0
best_agrs = None
best_model_path =None
best_path = None
base_dir = './results_bert/'
for path in os.listdir(base_dir):
    flag = True
    with open(os.path.join(base_dir + path), 'r') as f:
        results = eval(f.readline())
        tmp_acc = results['acc']
        tmp_f1 = results['f1']
        if best_acc < tmp_acc:
            best_acc = tmp_acc
            best_f1 = tmp_f1
            best_model_path = results['model_path']
            best_path = str(path)
            best_agrs = results
        f.close()

print('best_acc:%f,' % (best_acc))
print('best_f1:%f,' % (best_f1))
print('model_path:%s' % (best_model_path))
print('best_dirpath:%s' % (best_path))
for key in best_agrs.keys():
    if key in ['acc', 'f1', 'model_path']:
        continue
    else:
        print('%s:%s' % (key, str(best_agrs[key])), end=' ')

print('\n'+'-'*60)
print('start copy')
destpath = './best_result_log'
if not os.path.exists(destpath):
    os.mkdir(destpath, mode=0o777)
shutil.copy(os.path.join(base_dir, best_path), os.path.join(destpath, best_path))
print('copy complete')
print('-'*60)
