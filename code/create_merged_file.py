import argparse
import pathlib
import numpy as np
import pandas as pd
from scipy import stats


def read_survey_data(version='v2'):
    # BASELINE group
    if version=='v1':
        prediction_csv = 'data/user-study/v1/Prediction_only_July_22_2024.csv'
        participant_rows = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        slide_name_row = 23
        model_prediction_row = 21
    elif version=='v2':
        prediction_csv = 'data/user-study/v2/Prediction_only_July_22_2024_final.csv'
        participant_rows = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        slide_name_row = 27
        model_prediction_row = 25

    group_pred = pd.read_csv(prediction_csv)
    group_pred = pd.concat([group_pred.iloc[:, 31:121], 
                            group_pred[['ResponseId','Q3', 'Q6', 'Q200','Q370', 'Q376', 'Q373']]
                        ], axis=1)
    items_pred, slide_names, predictions = [], [], []
    for index in [index * 5 for index in range(0, 18)]:
        slide_names.append(group_pred.iloc[slide_name_row, index])
        predictions.append(group_pred.iloc[model_prediction_row, index])
        df = group_pred.iloc[participant_rows, np.arange(index, index+5).tolist() ].reset_index(drop=True)
        df.columns = ['Q_pred_1', 'Q_confidence_1', 'Q_pred_2', 'Q_confidence_2','Q_pred']
        df.insert(loc = 0, column = 'ResponseId', value  = group_pred.loc[participant_rows, ['ResponseId']].to_numpy().squeeze().tolist())
        df.insert(loc = 0, column = 'SlideName', value  = ['SlideName'] + ((df.shape[0]-1) * [slide_names[-1]]) )
        df.insert(loc = 0, column = 'Group', value  = ['Study group'] + ((df.shape[0]-1) * ['baseline']) )
        df.insert(loc = df.shape[1]-1, column = 'Q_saliency', value  = ['N/A'] + ((df.shape[0]-1) * [np.nan]) )
        df.insert(loc = df.shape[1]-1, column = 'Q_region', value  = ['N/A'] + ((df.shape[0]-1) * [np.nan]) )
        df.insert(loc = df.shape[1], column = 'background_1', value  = group_pred.loc[participant_rows, ['Q3']].to_numpy().squeeze().tolist())
        df.insert(loc = df.shape[1], column = 'background_2', value  = group_pred.loc[participant_rows, ['Q6']].to_numpy().squeeze().tolist())
        df.insert(loc = df.shape[1], column = 'feedback_1', value  = group_pred.loc[participant_rows, ['Q200']].to_numpy().squeeze().tolist())
        df.insert(loc = df.shape[1], column = 'feedback_2', value  = group_pred.loc[participant_rows, ['Q370']].to_numpy().squeeze().tolist())
        df.insert(loc = df.shape[1], column = 'feedback_3', value  = group_pred.loc[participant_rows, ['Q376']].to_numpy().squeeze().tolist())
        df.insert(loc = df.shape[1], column = 'feedback_4', value  = group_pred.loc[participant_rows, ['Q373']].to_numpy().squeeze().tolist())
        items_pred.append(df)

    df_baseline = pd.DataFrame(columns=list(items_pred[0].keys()))
    for df in items_pred:
        df_baseline = pd.concat([df_baseline, df.loc[1:, :].reset_index(drop=True)], axis=0)
    df_baseline = df_baseline.reset_index(drop=True)

    # XAI group
    if version=='v1':
        xai_csv = 'data/user-study/v1/All_Outputs_July_24_2024.csv'
        participant_rows = [0,2,3,4,6,7,8,9,10,15,16,17,18,19,20,21,22,23]
        slide_name_row = 27
        model_prediction_row = 25
    elif version=='v2':
        xai_csv = 'data/user-study/v2/All Outputs_July_24_2024_final.csv'
        participant_rows = [0,2,3,4,6,7,8,9,10,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
        slide_name_row = 34
        model_prediction_row = 32

    group_xai = pd.read_csv(xai_csv)
    group_xai = pd.concat([group_xai.iloc[:, 31:157], 
                        group_xai[['ResponseId','Q3', 'Q6', 'Q200','Q370', 'Q376', 'Q373']]
                        ], axis=1)

    items_xai, slide_names, predictions = [], [], []
    for index in [index * 7 for index in range(0, 18)]:
        slide_names.append(group_xai.iloc[slide_name_row, index])
        predictions.append(group_xai.iloc[model_prediction_row, index])
        df = group_xai.iloc[participant_rows, np.arange(index, index+7).tolist() ].reset_index(drop=True)
        df.columns = ['Q_pred_1', 'Q_confidence_1', 'Q_pred_2', 'Q_confidence_2', 'Q_saliency', 'Q_region','Q_pred']
        df.insert(loc = 0, column = 'ResponseId', value  = group_xai.loc[participant_rows, ['ResponseId']].to_numpy().squeeze().tolist())
        df.insert(loc = 0, column = 'SlideName', value  = ['SlideName'] + ((df.shape[0]-1) * [slide_names[-1]]) )
        df.insert(loc = 0, column = 'Group', value  = ['Study group'] + ((df.shape[0]-1) * ['xai']) )  
        df.insert(loc = df.shape[1], column = 'background_1', value  = group_xai.loc[participant_rows, ['Q3']].to_numpy().squeeze().tolist())
        df.insert(loc = df.shape[1], column = 'background_2', value  = group_xai.loc[participant_rows, ['Q6']].to_numpy().squeeze().tolist())
        df.insert(loc = df.shape[1], column = 'feedback_1', value  = group_xai.loc[participant_rows, ['Q200']].to_numpy().squeeze().tolist())
        df.insert(loc = df.shape[1], column = 'feedback_2', value  = group_xai.loc[participant_rows, ['Q370']].to_numpy().squeeze().tolist())
        df.insert(loc = df.shape[1], column = 'feedback_3', value  = group_xai.loc[participant_rows, ['Q376']].to_numpy().squeeze().tolist())
        df.insert(loc = df.shape[1], column = 'feedback_4', value  = group_xai.loc[participant_rows, ['Q373']].to_numpy().squeeze().tolist())
        items_xai.append(df)
        
    df_xai = pd.DataFrame(columns=list(items_xai[0].keys()))
    for df in items_xai:
        df_xai = pd.concat([df_xai, df.loc[1:, :].reset_index(drop=True)], axis=0)
    df_xai = df_xai.reset_index(drop=True)

    # correct a small typo error in the original slide name
    df = pd.concat([df_baseline, df_xai], axis=0).reset_index(drop=True)
    for index in range(0, df.shape[0]):
        if df.iloc[index]['SlideName'] == 'WSSlide373-01':
            df.loc[index, 'SlideName'] = 'WSSlide373'

    slide2predclass ={'22q11DSSlide52':'22q11.2 deletion syndrome',
                    '22q11DSSlide56':'Kabuki syndrome',
                    '22q11DSSlide210':'22q11.2 deletion syndrome',
                    'AngelmanSlide13':'Angelman syndrome',
                    'AngelmanSlide92':'Angelman syndrome',
                    'AngelmanSlide273':'Kabuki syndrome',
                    'KSSlide5':'Kabuki syndrome',
                    'KSSlide133':'Kabuki syndrome',
                    'KSSlide134':'22q11.2 deletion syndrome',
                    'NSSlide103':'Noonan syndrome',
                    'NSSlide198':'Noonan syndrome',
                    'NSSlide300':'Williams syndrome',
                    'UnaffectedSlide204':'Unaffected',
                    'UnaffectedSlide229':'Unaffected',
                    'UnaffectedSlide237':'Angelman syndrome',
                    'WSSlide11':'Angelman syndrome',
                    'WSSlide228':'Williams syndrome',
                    'WSSlide373':'Williams syndrome'}

    categories = {'22q11DS':'22q11.2 deletion syndrome',
                'Angelman':'Angelman syndrome',
                'KS':'Kabuki syndrome',
                'NS':'Noonan syndrome',
                'WS':'Williams syndrome',
                'Unaffected':'Unaffected'
                }

    ground_truth, predictions =[], []
    for index in range(0, df.shape[0]):
        predictions.append(slide2predclass[df.iloc[index]['SlideName']])
        ground_truth.append(categories[df.iloc[index]['SlideName'].split('Slide')[0]])

    df.insert(loc = 3, column = 'ModelPrediction', value  =  predictions)
    df.insert(loc = 3, column = 'GroundTruth', value  =  ground_truth)

    # Remove invalid responses
    indices2remove = [idx for idx, s in enumerate(list(df['Q_pred_1'])) if s not in list(categories.values())] + \
                    [idx for idx, s in enumerate(list(df['Q_pred_2'])) if s not in list(categories.values())]
    indices2remove = list(set(indices2remove))
    df = df.drop(indices2remove, axis=0).reset_index(drop=True)

    indices2remove_ = list([np.argwhere(df['ResponseId']==participant_id).squeeze().tolist() for participant_id in list(df['ResponseId'].unique()) if np.sum(df['ResponseId'] == participant_id) < 9 ])
    indices2remove = []
    for indices in indices2remove_:
        if isinstance(indices, list):
            indices2remove += indices
        else:
            indices2remove += [indices]
    indices2remove = list(set(indices2remove))
    df = df.drop(indices2remove, axis=0).reset_index(drop=True)

    # Add calculated columns
    df.insert(loc = df.shape[1], column = 'Q1_isTruePred', value  = (df['Q_pred_1'] == df['GroundTruth']).astype(np.int8))
    df.insert(loc = df.shape[1], column = 'Q2_isTruePred', value  = (df['Q_pred_2'] == df['GroundTruth']).astype(np.int8))
    df.insert(loc = df.shape[1], column = 'AI_isTruePred', value  = (df['ModelPrediction'] == df['GroundTruth']).astype(np.int8))
    df.insert(loc = df.shape[1], column = 'prediction_improvement', value  = df['Q2_isTruePred'] - df['Q1_isTruePred'])
    df.insert(loc = df.shape[1], column = 'confidence_improvement', value  = df['Q_confidence_2'].astype(np.int8) - df['Q_confidence_1'].astype(np.int8))

    output_csv = pathlib.Path('output/merged_table_%s.csv' % version)
    output_dir = pathlib.Path(output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    read_survey_data(version='v2')
