import torch
import pandas as pd
import numpy as np
import common_utils

data = common_utils.read_data('NOTEEVENTS-2.csv')
label = common_utils.read_label('annotation.csv')
w2v_model = read_word2vec_model('mimiciii_word2vec.wordvectors')
merged_data = common_utils.join_data_with_label(data, label)

num_discharge_sumarry = len(data)
num_unique_patients = len(np.unique(data.subject_id.values))

num_annotation = len(label)
label['unique_pat_hospital_id'] = label.apply(lambda x: str(x['subject.id'] + x['Hospital.Admission.ID']), axis=1)
num_unique_patient_in_annotation = len(np.unique(label['subject.id'].values))
num_unique_patient_hospital_pair_in_annotation = len(np.unique(label['unique_pat_hospital_id'].values ))
num_labels = 13

print('Num discharge summary : ' + str(num_discharge_sumarry))
print('Num unique patient: ' + str(num_unique_patients))

print('Number of labelled data : ' + str(num_annotation))
print('Number of unique patient in annotation data : ' + str(num_unique_patient_in_annotation))
print('Number of unique patient + hospital in annotation data : ' + str(num_unique_patient_hospital_pair_in_annotation ))


print(w2v_model.wv.most_similar('drug', topn=10))