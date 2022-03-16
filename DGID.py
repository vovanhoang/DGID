from audioop import avg
import cmath
from fileinput import filename
from gc import enable
import sys
from matplotlib.pyplot import axis
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, auc 
import warnings
import pickle
import tensorflow as tf
from fastprogress import fastprogress
warnings.filterwarnings("ignore")
from fastai.tabular.all import *


torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

#setting GPU Mem
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# #############################################

if torch.cuda.is_available():
    print("GPU enabling...")
    torch.cuda.device('cuda')
else:
	print("No GPU")

#########################################################
dataPath = ('/Volumes/D/NCS_2021-2023/DGID/10_2/dataset/')
modelPath = '/Volumes/D/NCS_2021-2023/DGID/10_2/'
fileName = 'train.csv'
df_train = pd.read_csv('/Volumes/D/NCS_2021-2023/DGID/10_2/dataset/train.csv')
df_test = pd.read_csv('/Volumes/D/NCS_2021-2023/DGID/10_2/dataset/test.csv')

#########################################################
labels = ['SQL Injection', 'Infilteration', 'DoS attacks-SlowHTTPTest','DoS attacks-GoldenEye', 'Bot', 'DoS attacks-Slowloris','Brute Force -Web', 'DDOS attack-LOIC-UDP', 'Benign','Brute Force -XSS']

cat_names = ['Dst Port', 'Protocol']
y_names = 'Label'
cont_names = ['Flow Duration', 'Tot Fwd Pkts',
              'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
              'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
              'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
              'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
              'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
              'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
              'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
              'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
              'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
              'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
              'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
              'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
              'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
              'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
              'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
              'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
              'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
              'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
              'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
              'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']
# labels = ['Dos', 'U2R', 'R2L', 'Probe', 'normal']


# cat_names = ['protocol_type', 'service']
# y_names = 'Label'

# cont_names = ['flag', 'src_bytes', 'dst_bytes',
#               'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
#               'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
#               'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
#               'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
#               'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
#               'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
#               'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
#               'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
#               'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
#               'dst_host_srv_rerror_rate']

procs = [Categorify, FillMissing, Normalize]
y_block = CategoryBlock()
#########################################################
verbose = 0
dep_var = 'Label'
params = {'gpu_id':0,
    'n_estimators':300,
    'tree_method':'exact',    
    #'tree_method':'gpu_hist',
    'max_depth':6,
    # 'enable_categorical':True,
    'objective':"multi:softmax", 
    'booster':"gbtree",     
    'learning_rate':0.01,     
    # 'silent':0, 
    # 'single_precision_histogram': True,
    'eval_metric':"mlogloss"    
}
#########################################################
print('XGBoost Training model...')
model = xgb.XGBClassifier(**params)
xgb_model = model.fit(df_train[df_train.columns[:-1]], df_train[y_names])
print('XGBoost Predicting...')
start1 = time.time()
xgb_preds = xgb_model.predict_proba(df_test[df_test.columns[:-1]])
elapsed_xgb = time.time() - start1
print('XGBoost save model...')
file_name = "xgb_model.pkl"
pickle.dump(xgb_model, open(file_name, "wb"))
# print(classification_report(xgb_preds,df_test[y_names]))
# xgboostacc = accuracy_score(df_test[y_names], xgb_preds)
# cm = confusion_matrix(df_test[y_names], xgb_preds)
# print(cm)
# precision1 = precision_score(df_test[y_names], xgb_preds, average='weighted')
# f11 = f1_score(df_test[y_names], xgb_preds, average='weighted')
# recall1 = recall_score(df_test[y_names], xgb_preds, average='weighted')
# print('Accuracy: {:.2f}%; precision: {:.2f}%; F1: {:.2f}%; Recall: {:.2f}%; elapsed: {:.2f} s'.format(xgboostacc*100, precision1*100, f11*100, recall1*100,  elapsed_xgb ))
#########################################################
print('DNN Training model...')
# acc1 = 0.8
# acc3 = 0.9
# step = 0
# while acc3 < 0.9999:
    # step = step + 1
print('Training model...')
print('Setting model...' )
        # create model
dls = TabularDataLoaders.from_df(df_train, path=dataPath, cat_names=cat_names, cont_names=cont_names, procs=procs, y_names=y_names, bs=64 ) #, valid_idx=list(range(1,test.shape[0])))
roc_auc = RocAuc(average='weighted')
learn = tabular_learner(dls, layers=[400,200], metrics=[accuracy, Precision(average='weighted'), F1Score(average='weighted'), Recall(average='weighted'), roc_auc])
learn.fit(5, 1e-2)
dl = learn.dls.test_dl(df_test, with_labels=True, drop_last=False)
print("DNN Predicting...")
start2 = time.time()
nn_preds, tests, clas_idx = learn.get_preds(dl=dl, with_loss=False, with_decoded=True)
loss, acc, precision, f1, recall, roc = learn.validate(dl=dl)
elapsed = time.time() - start2

acc1 = accuracy(tensor(xgb_preds), tensor(tests))
print('Accuracy of XGBoost: {:.2f}%' .format(acc1*100,))
acc2 = accuracy(tensor(nn_preds), tensor(tests))
print('Accuracy of DNN: {:.2f}%' .format(acc2*100,))
start4 = time.time() 
avgs = (nn_preds + xgb_preds) / 2
elapsed_agvs = time.time() - start4
print('start4: ', elapsed_agvs)
    #argmax = avgs.argmax(dim=1)
start3 = time.time()
acc3 = accuracy(tensor(avgs), tensor(tests))
elapsed_ensemble = time.time() - start3
print('Accuracy of Ensemble: {:.2f}%' .format(acc3*100,))
# print("Obtained good model! Current step = ", step)
modelName = "DGID"
modelFile = os.path.join(modelPath, modelName)
print('Saving model to file %s...' % modelFile)
learn.save(modelFile)
learn.export(modelFile)

#########################################################
print("-----FINAL RESULT------")
print("XGBboost_Elapsed: ", elapsed_xgb)
print('DNN_Elapsed: ', elapsed)
print('Ensemble_Elapsed: ', elapsed_ensemble)

print('Result of DNN')
print('Accuracy: {:.2f}%; precision: {:.2f}%; F1: {:.2f}%; Recall: {:.2f}%; roc-auc: {:.2f}%; elapsed: {:.2f} s'.format(acc*100, precision*100, f1*100, recall*100, roc*100,  elapsed ))
interp = ClassificationInterpretation.from_learner(learn,dl=dl)
print("Confusion Matrix:\n", interp.confusion_matrix())

print('XGBoost result')
precision1 = precision_score(tests, xgb_preds.argmax(axis=1), average='weighted')
f11 = f1_score(tests, xgb_preds.argmax(axis=1), average='weighted')
recall1 = recall_score(tests, xgb_preds.argmax(axis=1), average='weighted')
print('Accuracy: {:.2f}%; precision: {:.2f}%; F1: {:.2f}%; Recall: {:.2f}%; elapsed: {:.2f} s'.format(acc1*100, precision1*100, f11*100, recall1*100,  elapsed_xgb ))
print(classification_report(xgb_preds.argmax(axis=1),tests))
cm = confusion_matrix(tests,np.argmax(xgb_preds, axis=1))
print(cm)

print('Ensemble learning')
accuracy3 = accuracy_score(tests, avgs.argmax(axis=1))
precision3 = precision_score(tests, avgs.argmax(axis=1), average='weighted')
f13 = f1_score(tests, avgs.argmax(axis=1), average='weighted')
recall3 = recall_score(tests, avgs.argmax(axis=1), average='weighted')
print('Accuracy: {:.2f}%; precision: {:.2f}%; F1: {:.2f}%; Recall: {:.2f}%; elapsed: {:.2f} s'.format(accuracy3*100, precision3*100, f13*100, recall3*100,  elapsed_ensemble ))
print("Elapsed_Ensemble: ", elapsed_ensemble)
print(classification_report(avgs.argmax(axis=1), tests))
cm = confusion_matrix(tests,avgs.argmax(axis=1))
print(cm)