import pathlib
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate
from matplotlib import pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from create_merged_file import read_survey_data

plt.rcParams['text.usetex'] = False
plt.style.use('ggplot')

def plot_fig2(df, output_dir):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='tight')

    # Prediction improvement
    baseline_AImodelFALSE = pd.DataFrame([float(df[(df['ResponseId']==ResponseId) & (df['AI_isTruePred']==0)]['prediction_improvement'].mean()) for ResponseId in list(df[df['Group']=='baseline']['ResponseId'].unique())]).squeeze().to_numpy()
    XAI_AImodelFALSE = pd.DataFrame([float(df[(df['ResponseId']==ResponseId) & (df['AI_isTruePred']==0)]['prediction_improvement'].mean()) for ResponseId in list(df[df['Group']=='xai']['ResponseId'].unique())]).squeeze().to_numpy()
    baseline_AImodelTRUE = pd.DataFrame([float(df[(df['ResponseId']==ResponseId) & (df['AI_isTruePred']==1)]['prediction_improvement'].mean()) for ResponseId in list(df[df['Group']=='baseline']['ResponseId'].unique())]).squeeze().to_numpy()
    XAI_AImodelTRUE = pd.DataFrame([float(df[(df['ResponseId']==ResponseId) & (df['AI_isTruePred']==1)]['prediction_improvement'].mean()) for ResponseId in list(df[df['Group']=='xai']['ResponseId'].unique())]).squeeze().to_numpy()
    df_ = pd.DataFrame(columns=['prediction improvement', 'AI model prediction', 'Group'])
    n = 0
    for item in list(baseline_AImodelTRUE):
        df_.loc[n,:] = [item, True, 'Baseline group']
        n += 1
    for item in list(XAI_AImodelTRUE):
        df_.loc[n,:] = [item, True, 'XAI group']
        n += 1
    for item in list(baseline_AImodelFALSE):
        df_.loc[n,:] = [item, False, 'Baseline group']
        n += 1
    for item in list(XAI_AImodelFALSE):
        df_.loc[n,:] = [item, False, 'XAI group']
        n += 1

    box_plot1 = sns.boxplot(x="AI model prediction", y="prediction improvement",
                hue="Group", palette=['#E24A33', '#348ABD'],
                data=df_, notch=False, 
                            showmeans=True, 
                            meanprops={'marker':'P', 'markeredgecolor':'black', 'markersize':'6'},
                            width=0.4, ax=ax[0])
    ax[0].grid(axis='x')
    means = df_.groupby(['AI model prediction', 'Group'])['prediction improvement'].mean()
    box_plot1.text(0-0.3, round(means.iloc[0], 2), round(means.iloc[0], 3), horizontalalignment='center',size='11',color='k')
    box_plot1.text(0+0.3, round(means.iloc[1], 2), round(means.iloc[1], 3), horizontalalignment='center',size='11',color='k')
    box_plot1.text(1-0.3, round(means.iloc[2], 2), round(means.iloc[2], 3), horizontalalignment='center',size='11',color='k')
    box_plot1.text(1+0.3, round(means.iloc[3], 2), round(means.iloc[3], 3), horizontalalignment='center',size='11',color='k')

    for tick in ax[0].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[0].get_yticklabels():
        tick.set_fontname("Arial")

    ax[0].legend(loc='upper left', ncols=1, fontsize=16) 
    ax[0].set_ylabel('Prediction improvement', fontname='Arial', fontsize=16)
    ax[0].set_xlabel('AI model prediction', fontname='Arial', fontsize=16)

    # Confidence improvement
    baseline_AImodelFALSE = pd.DataFrame([float(df[(df['ResponseId']==ResponseId) & (df['AI_isTruePred']==0)]['confidence_improvement'].mean()) for ResponseId in list(df[df['Group']=='baseline']['ResponseId'].unique())]).squeeze().to_numpy()
    XAI_AImodelFALSE = pd.DataFrame([float(df[(df['ResponseId']==ResponseId) & (df['AI_isTruePred']==0)]['confidence_improvement'].mean()) for ResponseId in list(df[df['Group']=='xai']['ResponseId'].unique())]).squeeze().to_numpy()
    baseline_AImodelTRUE = pd.DataFrame([float(df[(df['ResponseId']==ResponseId) & (df['AI_isTruePred']==1)]['confidence_improvement'].mean()) for ResponseId in list(df[df['Group']=='baseline']['ResponseId'].unique())]).squeeze().to_numpy()
    XAI_AImodelTRUE = pd.DataFrame([float(df[(df['ResponseId']==ResponseId) & (df['AI_isTruePred']==1)]['confidence_improvement'].mean()) for ResponseId in list(df[df['Group']=='xai']['ResponseId'].unique())]).squeeze().to_numpy()
    df_ = pd.DataFrame(columns=['confidence improvement', 'AI model prediction', 'Group'])
    n = 0
    for item in list(baseline_AImodelTRUE):
        df_.loc[n,:] = [item, True, 'Baseline group']
        n += 1
    for item in list(XAI_AImodelTRUE):
        df_.loc[n,:] = [item, True, 'XAI group']
        n += 1
    for item in list(baseline_AImodelFALSE):
        df_.loc[n,:] = [item, False, 'Baseline group']
        n += 1
    for item in list(XAI_AImodelFALSE):
        df_.loc[n,:] = [item, False, 'XAI group']
        n += 1

    box_plot2 = sns.boxplot(x="AI model prediction", y="confidence improvement",
                        hue="Group", palette=['#E24A33', '#348ABD'],
                        data=df_, notch=False, 
                            showmeans=True, 
                            meanprops={'marker':'P', 'markeredgecolor':'black', 'markersize':'6'},
                            width=0.4, ax=ax[1])
    ax[1].grid(axis='x')
    means = df_.groupby(['AI model prediction', 'Group'])['confidence improvement'].mean()
    box_plot2.text(0-0.3, round(means.iloc[0], 2), round(means.iloc[0], 3), horizontalalignment='center',size='11',color='k')
    box_plot2.text(0+0.3, round(means.iloc[1], 2), round(means.iloc[1], 3), horizontalalignment='center',size='11',color='k')
    box_plot2.text(1-0.3, round(means.iloc[2], 2), round(means.iloc[2], 3), horizontalalignment='center',size='11',color='k')
    box_plot2.text(1+0.3, round(means.iloc[3], 2), round(means.iloc[3], 3), horizontalalignment='center',size='11',color='k')

    for tick in ax[1].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[1].get_yticklabels():
        tick.set_fontname("Arial")

    ax[1].get_legend().remove() 
    ax[1].set_ylabel('Confidence improvement', fontname='Arial', fontsize=16)
    ax[1].set_xlabel('AI model prediction', fontname='Arial', fontsize=16)

    plt.savefig(pathlib.Path(output_dir, 
                             'fig-02-prediction-and-confidence-AI-model-true-false.svg'))
    plt.close(fig)

def plot_fig3(df, output_dir):
    easy_samples = [
        'KSSlide133', 'NSSlide103', 'KSSlide5', 'WSSlide373',
        'UnaffectedSlide204', '22q11DSSlide210', 'UnaffectedSlide229',
        'AngelmanSlide13', 'NSSlide198'
    ]

    i_baseline_easy = [i for i in range(0, df.shape[0]) if (df.iloc[i]['SlideName'] in easy_samples) and (df.iloc[i]['Group']=='baseline')]
    i_baseline_hard = [i for i in range(0, df.shape[0]) if (df.iloc[i]['SlideName'] not in easy_samples) and (df.iloc[i]['Group']=='baseline')]
    i_xai_easy = [i for i in range(0, df.shape[0]) if (df.iloc[i]['SlideName'] in easy_samples) and (df.iloc[i]['Group']=='xai')]
    i_xai_hard = [i for i in range(0, df.shape[0]) if (df.iloc[i]['SlideName'] not in easy_samples) and (df.iloc[i]['Group']=='xai')]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='tight')

    # Prediction improvement
    df_group = df.iloc[i_baseline_easy].reset_index(drop=True)
    baseline_EASY = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['prediction_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()
    df_group = df.iloc[i_baseline_hard].reset_index(drop=True)
    baseline_HARD = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['prediction_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()
    df_group = df.iloc[i_xai_easy].reset_index(drop=True)
    xai_EASY = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['prediction_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()
    df_group = df.iloc[i_xai_hard].reset_index(drop=True)
    xai_HARD = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['prediction_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()

    df_ = pd.DataFrame(columns=['prediction improvement', 'Sample difficulty', 'Group'])
    n = 0
    for item in list(baseline_HARD):
        df_.loc[n,:] = [item, 'difficult samples', 'Baseline group']
        n += 1
    for item in list(baseline_EASY):
        df_.loc[n,:] = [item, 'more obvious clinical features', 'Baseline group']
        n += 1
    for item in list(xai_HARD):
        df_.loc[n,:] = [item, 'difficult samples', 'XAI group']
        n += 1
    for item in list(xai_EASY):
        df_.loc[n,:] = [item, 'more obvious clinical features', 'XAI group']
        n += 1

    box_plot1 = sns.boxplot(x="Sample difficulty", y="prediction improvement",
                hue="Group", palette=['#E24A33', '#348ABD'],
                data=df_, notch=False, 
                            showmeans=True, 
                            meanprops={'marker':'P', 'markeredgecolor':'black', 'markersize':'6'},
                            width=0.4, ax=ax[0])
    ax[0].grid(axis='x')
    means = df_.groupby(['Sample difficulty', 'Group'])['prediction improvement'].mean()
    box_plot1.text(0-0.3, round(means.iloc[0], 2), round(means.iloc[0], 3), horizontalalignment='center',size='11',color='k')
    box_plot1.text(0+0.3, round(means.iloc[1], 2), round(means.iloc[1], 3), horizontalalignment='center',size='11',color='k')
    box_plot1.text(1-0.3, round(means.iloc[2], 2), round(means.iloc[2], 3), horizontalalignment='center',size='11',color='k')
    box_plot1.text(1+0.3, round(means.iloc[3], 2), round(means.iloc[3], 3), horizontalalignment='center',size='11',color='k')

    for tick in ax[0].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[0].get_yticklabels():
        tick.set_fontname("Arial")

    ax[0].legend(loc='upper left', ncols=1, fontsize=16) 
    ax[0].set_ylabel('Prediction improvement', fontname='Arial', fontsize=16)
    ax[0].set_xlabel('AI model prediction', fontname='Arial', fontsize=16)

    # Confidence improvement
    df_group = df.iloc[i_baseline_easy].reset_index(drop=True)
    baseline_EASY = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['confidence_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()
    df_group = df.iloc[i_baseline_hard].reset_index(drop=True)
    baseline_HARD = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['confidence_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()
    df_group = df.iloc[i_xai_easy].reset_index(drop=True)
    xai_EASY = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['confidence_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()
    df_group = df.iloc[i_xai_hard].reset_index(drop=True)
    xai_HARD = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['confidence_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()

    df_ = pd.DataFrame(columns=['confidence improvement', 'Sample difficulty', 'Group'])
    n = 0
    for item in list(baseline_HARD):
        df_.loc[n,:] = [item, 'difficult samples', 'Baseline group']
        n += 1
    for item in list(baseline_EASY):
        df_.loc[n,:] = [item, 'more obvious clinical features', 'Baseline group']
        n += 1
    for item in list(xai_HARD):
        df_.loc[n,:] = [item, 'difficult samples', 'XAI group']
        n += 1
    for item in list(xai_EASY):
        df_.loc[n,:] = [item, 'more obvious clinical features', 'XAI group']
        n += 1

    box_plot2 = sns.boxplot(x="Sample difficulty", y="confidence improvement",
                hue="Group", palette=['#E24A33', '#348ABD'],
                data=df_, notch=False, 
                            showmeans=True, 
                            meanprops={'marker':'P', 'markeredgecolor':'black', 'markersize':'6'},
                            width=0.4, ax=ax[1])
    ax[1].grid(axis='x')
    means = df_.groupby(['Sample difficulty', 'Group'])['confidence improvement'].mean()
    box_plot2.text(0-0.3, round(means.iloc[0], 2), round(means.iloc[0], 3), horizontalalignment='center',size='11',color='k')
    box_plot2.text(0+0.3, round(means.iloc[1], 2), round(means.iloc[1], 3), horizontalalignment='center',size='11',color='k')
    box_plot2.text(1-0.3, round(means.iloc[2], 2), round(means.iloc[2], 3), horizontalalignment='center',size='11',color='k')
    box_plot2.text(1+0.3, round(means.iloc[3], 2), round(means.iloc[3], 3), horizontalalignment='center',size='11',color='k')

    for tick in ax[1].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[1].get_yticklabels():
        tick.set_fontname("Arial")

    ax[1].get_legend().remove()
    ax[1].set_ylabel('Confidence improvement', fontname='Arial', fontsize=16)
    ax[1].set_xlabel('AI model prediction', fontname='Arial', fontsize=16)

    plt.savefig(pathlib.Path(output_dir, 
                             'fig-03-prediction-and-confidence-sample-difficulty.svg'))
    plt.close(fig)


def plot_fig3_NEW(df, output_dir):
    # Criteria: 
    # 'We did decide it is better to split easy and hard images by participant’s initial accuracy instead of Ben’s grouping. 
    # 'Figure 3 will need to be recreated with this new split. Below are the images for each group.'
    # Easy (>70% average accuracy)
    easy_samples = [
        '22q11DSSlide210', 'KSSlide5', 'KSSlide133', 'NSSlide103',
         'NSSlide198', 'WSSlide11', 'WSSlide228', 'WSSlide373', 'UnaffectedSlide229'
    ]


    i_baseline_easy = [i for i in range(0, df.shape[0]) if (df.iloc[i]['SlideName'] in easy_samples) and (df.iloc[i]['Group']=='baseline')]
    i_baseline_hard = [i for i in range(0, df.shape[0]) if (df.iloc[i]['SlideName'] not in easy_samples) and (df.iloc[i]['Group']=='baseline')]
    i_xai_easy = [i for i in range(0, df.shape[0]) if (df.iloc[i]['SlideName'] in easy_samples) and (df.iloc[i]['Group']=='xai')]
    i_xai_hard = [i for i in range(0, df.shape[0]) if (df.iloc[i]['SlideName'] not in easy_samples) and (df.iloc[i]['Group']=='xai')]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='tight')

    # Prediction improvement
    df_group = df.iloc[i_baseline_easy].reset_index(drop=True)
    baseline_EASY = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['prediction_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()
    df_group = df.iloc[i_baseline_hard].reset_index(drop=True)
    baseline_HARD = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['prediction_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()
    df_group = df.iloc[i_xai_easy].reset_index(drop=True)
    xai_EASY = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['prediction_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()
    df_group = df.iloc[i_xai_hard].reset_index(drop=True)
    xai_HARD = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['prediction_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()

    df_ = pd.DataFrame(columns=['prediction improvement', 'Sample difficulty', 'Group'])
    n = 0
    for item in list(baseline_HARD):
        df_.loc[n,:] = [item, 'difficult samples', 'Baseline group']
        n += 1
    for item in list(baseline_EASY):
        df_.loc[n,:] = [item, 'more obvious clinical features', 'Baseline group']
        n += 1
    for item in list(xai_HARD):
        df_.loc[n,:] = [item, 'difficult samples', 'XAI group']
        n += 1
    for item in list(xai_EASY):
        df_.loc[n,:] = [item, 'more obvious clinical features', 'XAI group']
        n += 1

    box_plot1 = sns.boxplot(x="Sample difficulty", y="prediction improvement",
                hue="Group", palette=['#E24A33', '#348ABD'],
                data=df_, notch=False, 
                            showmeans=True, 
                            meanprops={'marker':'P', 'markeredgecolor':'black', 'markersize':'6'},
                            width=0.4, ax=ax[0])
    ax[0].grid(axis='x')
    means = df_.groupby(['Sample difficulty', 'Group'])['prediction improvement'].mean()
    box_plot1.text(0-0.3, round(means.iloc[0], 2), round(means.iloc[0], 3), horizontalalignment='center',size='11',color='k')
    box_plot1.text(0+0.3, round(means.iloc[1], 2), round(means.iloc[1], 3), horizontalalignment='center',size='11',color='k')
    box_plot1.text(1-0.3, round(means.iloc[2], 2), round(means.iloc[2], 3), horizontalalignment='center',size='11',color='k')
    box_plot1.text(1+0.3, round(means.iloc[3], 2), round(means.iloc[3], 3), horizontalalignment='center',size='11',color='k')

    for tick in ax[0].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[0].get_yticklabels():
        tick.set_fontname("Arial")

    ax[0].legend(loc='upper left', ncols=1, fontsize=16) 
    ax[0].set_ylabel('Prediction improvement', fontname='Arial', fontsize=16)
    ax[0].set_xlabel('AI model prediction', fontname='Arial', fontsize=16)

    # Confidence improvement
    df_group = df.iloc[i_baseline_easy].reset_index(drop=True)
    baseline_EASY = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['confidence_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()
    df_group = df.iloc[i_baseline_hard].reset_index(drop=True)
    baseline_HARD = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['confidence_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()
    df_group = df.iloc[i_xai_easy].reset_index(drop=True)
    xai_EASY = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['confidence_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()
    df_group = df.iloc[i_xai_hard].reset_index(drop=True)
    xai_HARD = pd.DataFrame([df_group[df_group['ResponseId']==ResponseId]['confidence_improvement'].mean() for ResponseId in sorted(set(df_group['ResponseId'].tolist()))]).squeeze().to_numpy()

    df_ = pd.DataFrame(columns=['confidence improvement', 'Sample difficulty', 'Group'])
    n = 0
    for item in list(baseline_HARD):
        df_.loc[n,:] = [item, 'difficult samples', 'Baseline group']
        n += 1
    for item in list(baseline_EASY):
        df_.loc[n,:] = [item, 'more obvious clinical features', 'Baseline group']
        n += 1
    for item in list(xai_HARD):
        df_.loc[n,:] = [item, 'difficult samples', 'XAI group']
        n += 1
    for item in list(xai_EASY):
        df_.loc[n,:] = [item, 'more obvious clinical features', 'XAI group']
        n += 1

    box_plot2 = sns.boxplot(x="Sample difficulty", y="confidence improvement",
                hue="Group", palette=['#E24A33', '#348ABD'],
                data=df_, notch=False, 
                            showmeans=True, 
                            meanprops={'marker':'P', 'markeredgecolor':'black', 'markersize':'6'},
                            width=0.4, ax=ax[1])
    ax[1].grid(axis='x')
    means = df_.groupby(['Sample difficulty', 'Group'])['confidence improvement'].mean()
    box_plot2.text(0-0.3, round(means.iloc[0], 2), round(means.iloc[0], 3), horizontalalignment='center',size='11',color='k')
    box_plot2.text(0+0.3, round(means.iloc[1], 2), round(means.iloc[1], 3), horizontalalignment='center',size='11',color='k')
    box_plot2.text(1-0.3, round(means.iloc[2], 2), round(means.iloc[2], 3), horizontalalignment='center',size='11',color='k')
    box_plot2.text(1+0.3, round(means.iloc[3], 2), round(means.iloc[3], 3), horizontalalignment='center',size='11',color='k')

    for tick in ax[1].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[1].get_yticklabels():
        tick.set_fontname("Arial")

    ax[1].get_legend().remove()
    ax[1].set_ylabel('Confidence improvement', fontname='Arial', fontsize=16)
    ax[1].set_xlabel('AI model prediction', fontname='Arial', fontsize=16)

    plt.savefig(pathlib.Path(output_dir, 
                             'fig-03-prediction-and-confidence-sample-difficulty_NEW.svg'))
    plt.close(fig)

    
def plot_fig4(df, output_dir):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='tight')

    # (1) AI predictions
    likert_values = ('-2', '-1', '0', '+1', '+2')
    group_means = {
        'Baseline group': (round((df[df['Group']=='baseline']['Q_pred']==-2).sum() / df[df['Group']=='baseline']['Q_pred'].shape[0], 2),
                    round((df[df['Group']=='baseline']['Q_pred']==-1).sum() / df[df['Group']=='baseline']['Q_pred'].shape[0], 2),
                    round((df[df['Group']=='baseline']['Q_pred']==0).sum()  / df[df['Group']=='baseline']['Q_pred'].shape[0], 2), 
                    round((df[df['Group']=='baseline']['Q_pred']==1).sum()  / df[df['Group']=='baseline']['Q_pred'].shape[0], 2),
                    round((df[df['Group']=='baseline']['Q_pred']==2).sum()  / df[df['Group']=='baseline']['Q_pred'].shape[0], 2)),
        'XAI group': (round((df[df['Group']=='xai']['Q_pred']==-2).sum() / df[df['Group']=='xai']['Q_pred'].shape[0], 2),
                round((df[df['Group']=='xai']['Q_pred']==-1).sum() / df[df['Group']=='xai']['Q_pred'].shape[0], 2),
                round((df[df['Group']=='xai']['Q_pred']==0).sum()  / df[df['Group']=='xai']['Q_pred'].shape[0], 2), 
                round((df[df['Group']=='xai']['Q_pred']==1).sum()  / df[df['Group']=='xai']['Q_pred'].shape[0], 2),
                round((df[df['Group']=='xai']['Q_pred']==2).sum()  / df[df['Group']=='xai']['Q_pred'].shape[0], 2)),
    }

    x = np.arange(len(likert_values))  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0

    for attribute, measurement in group_means.items():
        offset = width * multiplier
        rects = ax[0].bar(x + offset, measurement, width, label=attribute)
        ax[0].bar_label(rects, padding=2, rotation=0)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[0].set_ylabel('Frequency', fontname='Arial', fontsize=16)
    ax[0].set_xlabel('Perceived usefulness', fontname='Arial', fontsize=16)
    ax[0].set_title('AI predictions', fontname='Arial', fontsize=18)
    ax[0].set_xticks(x + width, likert_values)
    ax[0].legend(loc='upper left', ncols=1, fontsize=16)
    ax[0].set_ylim(0, 0.55)
    for tick in ax[0].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[0].get_yticklabels():
        tick.set_fontname("Arial")


    likert_values = ('-2', '-1', '0', '+1', '+2')
    group_means = {
        'Saliency map': (round((df[df['Group']=='xai']['Q_saliency']==-2).sum() / df[df['Group']=='xai']['Q_saliency'].shape[0], 2),
                        round((df[df['Group']=='xai']['Q_saliency']==-1).sum() / df[df['Group']=='xai']['Q_saliency'].shape[0], 2),
                        round((df[df['Group']=='xai']['Q_saliency']==0).sum()  / df[df['Group']=='xai']['Q_saliency'].shape[0], 2), 
                        round((df[df['Group']=='xai']['Q_saliency']==1).sum()  / df[df['Group']=='xai']['Q_saliency'].shape[0], 2),
                        round((df[df['Group']=='xai']['Q_saliency']==2).sum()  / df[df['Group']=='xai']['Q_saliency'].shape[0], 2)),
        'Region relevance': (round((df[df['Group']=='xai']['Q_region']==-2).sum() / df[df['Group']=='xai']['Q_region'].shape[0], 2),
                            round((df[df['Group']=='xai']['Q_region']==-1).sum() / df[df['Group']=='xai']['Q_region'].shape[0], 2),
                            round((df[df['Group']=='xai']['Q_region']==0).sum()  / df[df['Group']=='xai']['Q_region'].shape[0], 2), 
                            round((df[df['Group']=='xai']['Q_region']==1).sum()  / df[df['Group']=='xai']['Q_region'].shape[0], 2),
                            round((df[df['Group']=='xai']['Q_region']==2).sum()  / df[df['Group']=='xai']['Q_region'].shape[0], 2)),
    }

    x = np.arange(len(likert_values))  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0

    for attribute, measurement in group_means.items():
        offset = width * multiplier
        color = 'cadetblue' if attribute=='Region relevance' else 'mediumpurple'
        rects = ax[1].bar(x + offset, measurement, width, label=attribute, color=color)
        ax[1].bar_label(rects, padding=2, rotation=0)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[1].set_ylabel('Frequency', fontname='Arial', fontsize=16)
    ax[1].set_xlabel('Perceived usefulness', fontname='Arial', fontsize=16)
    ax[1].set_title('Explanations (XAI group)', fontname='Arial', fontsize=18)
    ax[1].set_xticks(x + width, likert_values)
    ax[1].legend(loc='upper left', ncols=1, fontsize=16)
    ax[1].set_ylim(0, 0.55)
    for tick in ax[1].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[1].get_yticklabels():
        tick.set_fontname("Arial")

    plt.savefig(pathlib.Path(output_dir, 
                             'fig-04-usefulness-of-ai-prediction-saliency-region-relevance.svg'))
    plt.close(fig)


def plot_supp_fig1(df, output_dir):
    experience_years = []
    for participant_id in df[df['Group']=='baseline']['ResponseId'].unique():
        experience_years.append(['baseline', participant_id, df[(df['Group']=='baseline') & (df['ResponseId']==participant_id)]['background_1'].unique()[0]])

    for participant_id in df[df['Group']=='xai']['ResponseId'].unique():
        experience_years.append(['xai', participant_id, df[(df['Group']=='xai') & (df['ResponseId']==participant_id)]['background_1'].unique()[0]])

    df_exp = pd.DataFrame(experience_years, columns=['Group', 'ResponseId', 'Experience'])

    counts = {
        'Baseline group': (
            df_exp[(df_exp['Group']=='baseline') & (df_exp['Experience']=='< 1 year')].shape[0],
            df_exp[(df_exp['Group']=='baseline') & (df_exp['Experience']=='1 to 5 years')].shape[0],
            df_exp[(df_exp['Group']=='baseline') & (df_exp['Experience']=='5 to 10 years')].shape[0],
            df_exp[(df_exp['Group']=='baseline') & (df_exp['Experience']=='> 10 years')].shape[0]
        ),
        'XAI group': (
            df_exp[(df_exp['Group']=='xai') & (df_exp['Experience']=='< 1 year')].shape[0],
            df_exp[(df_exp['Group']=='xai') & (df_exp['Experience']=='1 to 5 years')].shape[0],
            df_exp[(df_exp['Group']=='xai') & (df_exp['Experience']=='5 to 10 years')].shape[0],
            df_exp[(df_exp['Group']=='xai') & (df_exp['Experience']=='> 10 years')].shape[0]
        ),
    }
    counts = pd.DataFrame(counts)

    fig, ax = plt.subplots(figsize=(8, 5), layout='tight')
    x = np.arange(4)
    width = 0.2
    multiplier = 0
    for attribute, measurement in counts.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width=width, capsize=5, label=attribute)
        ax.bar_label(rects, padding=2, fontname='Arial', fontsize=16)
        multiplier += 1
    ax.legend(loc='upper left', ncols=1, fontsize=16)
    ax.set_xlabel('Experience', fontname='Arial', fontsize=16)
    ax.set_ylabel('Count', fontname='Arial', fontsize=16)
    ax.set_xticks(x + 0.5*width, ['$<$ 1 year', '1 to 5 years', '5 to 10 years', '$>$ 10 years'])
    for tick in ax.get_xticklabels():
        tick.set_fontname("Arial")
        tick.set_fontsize(16)
    for tick in ax.get_yticklabels():
        tick.set_fontname("Arial")
        tick.set_fontsize(16)
    plt.savefig(pathlib.Path(output_dir, 
                             'supp-fig-01-participant-demographics-experience-years.svg'))
    plt.close(fig)


import matplotlib.patches as mpatches

def plot_supp_fig2(df, output_dir):
    # Prediction and confidence improvement per slide (baseline)
    ai_model_results = []
    for slide_name in sorted(list(df['SlideName'].unique())):
        ii = np.argwhere((df['SlideName'] == slide_name) & (df['Group'] == 'baseline')).squeeze()
        group_mean = df.iloc[ii][['prediction_improvement', 'confidence_improvement']].mean().tolist()
        group_ci = (1.96 * df.iloc[ii][['prediction_improvement', 'confidence_improvement']].sem()).tolist()
        ai_model_results.append(
            df.iloc[ii[0]][['SlideName', 'GroundTruth', 'ModelPrediction']].tolist() +
            [group_mean[0], group_ci[0], group_mean[1], group_ci[1]]
        )

    ai_model_results = pd.DataFrame(
        ai_model_results,
        columns=[
            'SlideName', 'GroundTruth', 'ModelPrediction',
            'prediction_improvement_mean', 'prediction_improvement_ci',
            'confidence_improvement_mean', 'confidence_improvement_ci'
        ]
    )

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), layout='tight')

    # Prediction improvement bars
    for i in range(ai_model_results.shape[0]):
        color = '#348ABD' if ai_model_results.iloc[i]['GroundTruth'] == ai_model_results.iloc[i]['ModelPrediction'] else '#E24A33'
        ax[0].bar(
            x=ai_model_results.iloc[i]['SlideName'],
            height=ai_model_results.iloc[i]['prediction_improvement_mean'],
            yerr=ai_model_results.iloc[i]['prediction_improvement_ci'],
            capsize=5,
            width=0.2,
            color=color
        )

    ax[0].grid(True)
    ax[0].set_xticklabels([])
    ax[0].set_ylabel('Prediction improvement', fontname='Arial', fontsize=22)
    ax[0].set_ylim(-1.3, 1.3)
    legends = [
        mpatches.Patch(color='#E24A33', label='AI model FALSE predicted'),
        mpatches.Patch(color='#348ABD', label='AI model TRUE predicted')
    ]
    ax[0].legend(handles=legends, loc='upper right', ncols=2, fontsize=22)

    # Confidence improvement bars
    for i in range(ai_model_results.shape[0]):
        color = '#348ABD' if ai_model_results.iloc[i]['GroundTruth'] == ai_model_results.iloc[i]['ModelPrediction'] else '#E24A33'
        ax[1].bar(
            x=ai_model_results.iloc[i]['SlideName'],
            height=ai_model_results.iloc[i]['confidence_improvement_mean'],
            yerr=ai_model_results.iloc[i]['confidence_improvement_ci'],
            capsize=5,
            width=0.2,
            color=color
        )
    ax[1].set_xticks(ax[1].get_xticks())
    ax[1].set_xticklabels(ai_model_results['SlideName'], rotation=60, ha='right', fontname='Arial', fontsize=20)
    ax[1].set_ylabel('Confidence improvement', fontname='Arial', fontsize=22)
    ax[1].set_xlabel('Slide names', fontname='Arial', fontsize=22)
    ax[1].set_ylim(-1.3, 1.3)

    for tick in ax[0].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[0].get_yticklabels():
        tick.set_fontname("Arial")
    for tick in ax[1].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[1].get_yticklabels():
        tick.set_fontname("Arial")

    plt.savefig(pathlib.Path(output_dir, 
                             'supp-fig-02-prediction-confidence-improvement-by-slide.svg'))
    plt.close(fig)


def plot_supp_fig3(df, output_dir):
    # Prediction and confidence improvement per slide (XAI)
    ai_model_results = []
    for slide_name in sorted(list(df['SlideName'].unique())):
        ii = np.argwhere((df['SlideName'] == slide_name) & (df['Group'] == 'xai')).squeeze()
        group_mean = df.iloc[ii][['prediction_improvement', 'confidence_improvement']].mean().tolist()
        group_ci = (1.96 * df.iloc[ii][['prediction_improvement', 'confidence_improvement']].sem()).tolist()
        ai_model_results.append(
            df.iloc[ii[0]][['SlideName', 'GroundTruth', 'ModelPrediction']].tolist() +
            [group_mean[0], group_ci[0], group_mean[1], group_ci[1]]
        )

    ai_model_results = pd.DataFrame(
        ai_model_results,
        columns=[
            'SlideName', 'GroundTruth', 'ModelPrediction',
            'prediction_improvement_mean', 'prediction_improvement_ci',
            'confidence_improvement_mean', 'confidence_improvement_ci'
        ]
    )

    fig, ax = plt.subplots(2, 1, figsize=(15, 10), layout='tight')

    for i in range(ai_model_results.shape[0]):
        color = '#348ABD' if ai_model_results.iloc[i]['GroundTruth'] == ai_model_results.iloc[i]['ModelPrediction'] else '#E24A33'
        ax[0].bar(
            x=ai_model_results.iloc[i]['SlideName'],
            height=ai_model_results.iloc[i]['prediction_improvement_mean'],
            yerr=ai_model_results.iloc[i]['prediction_improvement_ci'],
            capsize=5,
            width=0.2,
            color=color
        )

    ax[0].grid(True)
    ax[0].set_xticklabels([])
    ax[0].set_ylabel('Prediction improvement', fontname='Arial', fontsize=22)
    ax[0].set_ylim(-1.0, 1.0)
    legends = [
        mpatches.Patch(color='#E24A33', label='AI model FALSE predicted'),
        mpatches.Patch(color='#348ABD', label='AI model TRUE predicted')
    ]
    ax[0].legend(handles=legends, loc='upper right', ncols=2, fontsize=22)

    for i in range(ai_model_results.shape[0]):
        color = '#348ABD' if ai_model_results.iloc[i]['GroundTruth'] == ai_model_results.iloc[i]['ModelPrediction'] else '#E24A33'
        ax[1].bar(
            x=ai_model_results.iloc[i]['SlideName'],
            height=ai_model_results.iloc[i]['confidence_improvement_mean'],
            yerr=ai_model_results.iloc[i]['confidence_improvement_ci'],
            capsize=5,
            width=0.2,
            color=color
        )
    ax[1].set_xticks(ax[1].get_xticks())
    ax[1].set_xticklabels(ai_model_results['SlideName'], rotation=60, ha='right', fontname='Arial', fontsize=20)
    ax[1].set_ylabel('Confidence improvement', fontname='Arial', fontsize=22)
    ax[1].set_xlabel('Slide names', fontname='Arial', fontsize=22)
    ax[1].set_ylim(-2, 2)

    for tick in ax[0].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[0].get_yticklabels():
        tick.set_fontname("Arial")
    for tick in ax[1].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[1].get_yticklabels():
        tick.set_fontname("Arial")

    plt.savefig(pathlib.Path(output_dir, 'supp-fig-03-prediction-confidence-improvement-by-slide-XAI.svg'))
    plt.close(fig)


def plot_supp_fig4(df, output_dir):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10), layout='tight')

    group_mean = {
        '-1': df[(df['prediction_improvement']==-1) & (df['Group']=='baseline')]['Q_pred'].mean(),
        '0' : df[(df['prediction_improvement']==0) & (df['Group']=='baseline')]['Q_pred'].mean(),
        '+1': df[(df['prediction_improvement']==1) & (df['Group']=='baseline')]['Q_pred'].mean(),
    }
    group_mean = pd.Series(group_mean)
    group_sem = {
        '-1': df[(df['prediction_improvement']==-1) & (df['Group']=='baseline')]['Q_pred'].sem(),
        '0' : df[(df['prediction_improvement']==0) & (df['Group']=='baseline')]['Q_pred'].sem(),
        '+1': df[(df['prediction_improvement']==1) & (df['Group']=='baseline')]['Q_pred'].sem(),
    }
    group_sem = pd.Series(group_sem)
    ci = 1.96 * group_sem

    ax[0,0].bar(x=group_mean.index, height=group_mean, width=0.2, yerr=ci, capsize=5) 
    vals =  group_mean.tolist()
    ax[0,0].text(group_mean.index.tolist()[0],  vals[0]+0.02, '%2.3f' % float(vals[0]))
    ax[0,0].text(group_mean.index.tolist()[1],  vals[1]+0.02, '%2.3f' % float(vals[1]))
    ax[0,0].text(group_mean.index.tolist()[2],  vals[2]+0.02, '%2.3f' % float(vals[2]))
    ax[0,0].set_xlabel('Prediction improvement', fontname='Arial', fontsize=20)
    ax[0,0].set_ylabel('Perceived usefulness', fontname='Arial', fontsize=20)
    ax[0,0].set_title('AI predictions (Baseline group)', fontname='Arial', fontsize=20)
    ax[0,0].set_xticks(ax[0,0].get_xticks())
    ax[0,0].set_ylim(-1.6, 1.6)
    for tick in ax[0,0].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[0,0].get_yticklabels():
        tick.set_fontname("Arial")

    group_mean = {
        '-1': df[(df['prediction_improvement']==-1) & (df['Group']=='xai')]['Q_pred'].mean(),
        '0' : df[(df['prediction_improvement']==0) & (df['Group']=='xai')]['Q_pred'].mean(),
        '+1': df[(df['prediction_improvement']==1) & (df['Group']=='xai')]['Q_pred'].mean(),
    }
    group_mean = pd.Series(group_mean)
    group_sem = {
        '-1': df[(df['prediction_improvement']==-1) & (df['Group']=='xai')]['Q_pred'].sem(),
        '0' : df[(df['prediction_improvement']==0) & (df['Group']=='xai')]['Q_pred'].sem(),
        '+1': df[(df['prediction_improvement']==1) & (df['Group']=='xai')]['Q_pred'].sem(),
    }
    group_sem = pd.Series(group_sem)
    ci = 1.96 * group_sem

    ax[0,1].bar(x=group_mean.index, height=group_mean, width=0.2, yerr=ci, capsize=5, color='#348ABD') 
    vals =  group_mean.tolist()
    ax[0,1].text(group_mean.index.tolist()[0],  vals[0]+0.02, '%2.3f' % float(vals[0]))
    ax[0,1].text(group_mean.index.tolist()[1],  vals[1]+0.02, '%2.3f' % float(vals[1]))
    ax[0,1].text(group_mean.index.tolist()[2],  vals[2]+0.02, '%2.3f' % float(vals[2]))
    ax[0,1].set_xlabel('Prediction improvement', fontname='Arial', fontsize=20)
    ax[0,1].set_ylabel('Perceived usefulness', fontname='Arial', fontsize=20)
    ax[0,1].set_title('AI predictions (XAI group)', fontname='Arial', fontsize=20)
    ax[0,1].set_xticks(ax[0,1].get_xticks())
    ax[0,1].set_ylim(-1.6, 1.6)
    for tick in ax[0,1].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[0,1].get_yticklabels():
        tick.set_fontname("Arial")

    group_mean = {
        '-1': df[(df['prediction_improvement']==-1) & (df['Group']=='xai')]['Q_saliency'].mean(),
        '0' : df[(df['prediction_improvement']==0) & (df['Group']=='xai')]['Q_saliency'].mean(),
        '+1': df[(df['prediction_improvement']==1) & (df['Group']=='xai')]['Q_saliency'].mean(),
    }
    group_mean = pd.Series(group_mean)
    group_sem = {
        '-1': df[(df['prediction_improvement']==-1) & (df['Group']=='xai')]['Q_saliency'].sem(),
        '0' : df[(df['prediction_improvement']==0) & (df['Group']=='xai')]['Q_saliency'].sem(),
        '+1': df[(df['prediction_improvement']==1) & (df['Group']=='xai')]['Q_saliency'].sem(),
    }
    group_sem = pd.Series(group_sem)
    ci = 1.96 * group_sem

    ax[1,0].bar(x=group_mean.index, height=group_mean, width=0.2, yerr=ci, capsize=5, color='mediumpurple') 
    vals =  group_mean.tolist()
    ax[1,0].text(group_mean.index.tolist()[0],  vals[0]-0.07, '%2.3f' % float(vals[0]))
    ax[1,0].text(group_mean.index.tolist()[1],  vals[1]-0.07, '%2.3f' % float(vals[1]))
    ax[1,0].text(group_mean.index.tolist()[2],  vals[2]-0.07, '%2.3f' % float(vals[2]))
    ax[1,0].set_xlabel('Prediction improvement', fontname='Arial', fontsize=20)
    ax[1,0].set_ylabel('Perceived usefulness', fontname='Arial', fontsize=20)
    ax[1,0].set_title('Explanations (saliency map)', fontname='Arial', fontsize=20)
    ax[1,0].set_xticks(ax[1,0].get_xticks())
    ax[1,0].set_ylim(-1.8, 1.6)
    for tick in ax[1,0].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[1,0].get_yticklabels():
        tick.set_fontname("Arial")
        
    group_mean = {
        '-1': df[(df['prediction_improvement']==-1) & (df['Group']=='xai')]['Q_region'].mean(),
        '0' : df[(df['prediction_improvement']==0) & (df['Group']=='xai')]['Q_region'].mean(),
        '+1': df[(df['prediction_improvement']==1) & (df['Group']=='xai')]['Q_region'].mean(),
    }
    group_mean = pd.Series(group_mean)
    group_sem = {
        '-1': df[(df['prediction_improvement']==-1) & (df['Group']=='xai')]['Q_region'].sem(),
        '0' : df[(df['prediction_improvement']==0) & (df['Group']=='xai')]['Q_region'].sem(),
        '+1': df[(df['prediction_improvement']==1) & (df['Group']=='xai')]['Q_region'].sem(),
    }
    group_sem = pd.Series(group_sem)
    ci = 1.96 * group_sem

    ax[1,1].bar(x=group_mean.index, height=group_mean, width=0.2, yerr=ci, capsize=5, color='cadetblue') 
    vals =  group_mean.tolist()
    ax[1,1].text(group_mean.index.tolist()[0],  vals[0]-0.07, '%2.3f' % float(vals[0]))
    ax[1,1].text(group_mean.index.tolist()[1],  vals[1]-0.07, '%2.3f' % float(vals[1]))
    ax[1,1].text(group_mean.index.tolist()[2],  vals[2]-0.07, '%2.3f' % float(vals[2]))
    ax[1,1].set_xlabel('Prediction improvement', fontname='Arial', fontsize=20)
    ax[1,1].set_ylabel('Perceived usefulness', fontname='Arial', fontsize=20)
    ax[1,1].set_title('Explanations (region relevance)', fontname='Arial', fontsize=20)
    ax[1,1].set_xticks(ax[1,1].get_xticks())
    ax[1,1].set_ylim(-1.8, 1.6)
    for tick in ax[1,1].get_xticklabels():
        tick.set_fontname("Arial")
    for tick in ax[1,1].get_yticklabels():
        tick.set_fontname("Arial")

    plt.savefig(pathlib.Path(output_dir, 'supp-fig-04-correlation-of-usefulness-and-pred-improvement.svg'))
    plt.close(fig)



def plot_supp_fig5(df, output_dir):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout='tight')

    w = WordCloud()
    stop_words = list(w.stopwords)
    custom_stop_words = ['more', 'much', 'image', 'answer', 'take', 'rather', 'think', 'mostly', 'main']
    stop_words = set(stop_words + custom_stop_words)

    text = str(df[df['Group']=='baseline']['feedback_1'].unique())
    wordcloud = WordCloud(height=500, width=700, max_font_size=60, stopwords=stop_words, colormap='seismic', background_color='white').generate(text)
    ax[0].imshow(wordcloud, interpolation="bilinear")
    ax[0].axis("off")
    ax[0].set_title('Baseline group', fontname='Arial', fontsize=20)

    text = str(df[df['Group']=='xai']['feedback_1'].unique())
    wordcloud = WordCloud(height=500, width=700, max_font_size=60, stopwords=stop_words, colormap='seismic', background_color='white').generate(text)
    ax[1].imshow(wordcloud, interpolation="bilinear")
    ax[1].axis("off")
    ax[1].set_title('XAI group', fontname='Arial', fontsize=20)
    plt.tight_layout()
    plt.savefig(pathlib.Path(output_dir, 'supp-fig-05-wordcloud.svg'))
    plt.close(fig)

def correlation_analysis(df):
    print('Correlation analysis between prediction improvement and usefullness ratings')
    print('AI group (Baseline)')
    print(tabulate(df[df['Group']=='baseline'].loc[:, ['Q_pred', 'prediction_improvement']].corr(method='spearman'), 
               headers='keys', tablefmt='psql'))
    print('\n')
    print('XAI group')
    print(tabulate(df[df['Group']=='xai'].loc[:, ['Q_pred', 'Q_saliency', 'Q_region', 'prediction_improvement']].corr(method='spearman'), 
               headers='keys', tablefmt='psql'))

    print('\n')
    print('Comparisons / Spearmann correlation coefficients with p-values')

    res = stats.spearmanr(df[df['Group']=='baseline'].loc[:, ['prediction_improvement']],
                          df[df['Group']=='baseline'].loc[:, ['Q_pred']] )
    print('[Baseline] Prediction improvement vs. Q_pred:', 
          '%3.5f' % res.statistic, 
          '%d'    % len(df[df['Group']=='baseline'].loc[:, ['Q_pred']]), 
          '%3.5f' % res.pvalue)


    res = stats.spearmanr(df[df['Group']=='xai'].loc[:, ['Q_pred']],
                          df[df['Group']=='xai'].loc[:, ['prediction_improvement']])
    print('[XAI] prediction improvement vs Q_pred:', res.statistic, len(df[df['Group']=='xai'].loc[:, ['Q_pred']]), res.pvalue)


    res = stats.spearmanr(df[df['Group']=='xai'].loc[:, ['Q_saliency']],
                          df[df['Group']=='xai'].loc[:, ['prediction_improvement']])
    print('[XAI] prediction improvement vs Q_saliency:', res.statistic, len(df[df['Group']=='xai'].loc[:, ['Q_saliency']]), res.pvalue)

    res = stats.spearmanr(df[df['Group']=='xai'].loc[:, ['Q_region']],
                          df[df['Group']=='xai'].loc[:, ['prediction_improvement']] )
    print('[XAI] prediction improvement vs Q_region:', 
          '%3.5f' % res.statistic, 
          '%d'    % len(df[df['Group']=='xai'].loc[:, ['Q_region']]), 
          '%3.5f' % res.pvalue)

    res = stats.spearmanr(df[df['Group']=='xai'].loc[:, ['Q_saliency']],
                          df[df['Group']=='xai'].loc[:, ['Q_region']] )
    print('[XAI] region vs saliency:', 
          '%3.5f' % res.statistic, 
          '%d'    % len(df[df['Group']=='xai'].loc[:, ['Q_saliency']]), 
          '%3.5f' % res.pvalue)
    
    res = stats.spearmanr(df[df['Group']=='xai'].loc[:, ['Q_pred']],
                          df[df['Group']=='xai'].loc[:, ['Q_saliency']] )
    print('[XAI] Q_pred vs Q_saliency:', 
          '%3.5f' % res.statistic, 
          '%d'    % len(df[df['Group']=='xai'].loc[:, ['Q_saliency']]), 
          '%3.5f' % res.pvalue)
    
def statistical_tests(df):
    """
    Perform statistical tests on prediction and confidence improvements between groups.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the experimental data
    """
    
    # Helper function to extract participant-level means for a specific condition
    def get_condition_data(group, ai_true_pred, metric):
        return pd.DataFrame([
            float(df[(df['ResponseId'] == response_id) & (df['AI_isTruePred'] == ai_true_pred)][metric].mean())
            for response_id in df[df['Group'] == group]['ResponseId'].unique()
        ]).squeeze().to_numpy()
    
    # Helper function to print descriptive statistics
    def print_descriptive_stats(name1, data1, name2, data2):
        print(f"{name1} (n={len(data1)}) vs. {name2} (n={len(data2)})")
        print("-" * 50)
        print(f"{name1} (M={np.mean(data1):.3f}, Mdn={np.median(data1):.3f}, SD={np.std(data1):.3f})")
        print(f"{name2} (M={np.mean(data2):.3f}, Mdn={np.median(data2):.3f}, SD={np.std(data2):.3f})")
        print()
    
    # Helper function to run and display normality and variance tests
    def print_assumption_tests(name1, data1, name2, data2):
        # Shapiro-Wilk test for normality
        sw1 = stats.shapiro(data1)
        sw2 = stats.shapiro(data2)
        print(f"Shapiro-Wilk ({name1}):".ljust(35) + 
              f"Statistic={sw1.statistic:9.3f}, p={np.format_float_scientific(sw1.pvalue, precision=3)}, " + 
              f"{'Normal' if sw1.pvalue > 0.05 else 'Non-normal'}")
        
        print(f"Shapiro-Wilk ({name2}):".ljust(35) + 
              f"Statistic={sw2.statistic:9.3f}, p={np.format_float_scientific(sw2.pvalue, precision=3)}, " + 
              f"{'Normal' if sw2.pvalue > 0.05 else 'Non-normal'}")
        
        # Levene's test for homogeneity of variance
        lev = stats.levene(data1, data2)
        print(f"Levene's test:".ljust(35) + 
              f"Statistic={lev.statistic:9.3f}, p={lev.pvalue:.5f}, " + 
              f"{'Equal variances' if lev.pvalue > 0.05 else 'Unequal variances'}")
        print()
    
    # Helper function to run appropriate statistical test based on normality
    def run_statistical_test(name1, data1, name2, data2):
        # Check normality assumptions
        sw1 = stats.shapiro(data1)
        sw2 = stats.shapiro(data2)
        
        # If both distributions are normal, use t-test
        if sw1.pvalue > 0.05 and sw2.pvalue > 0.05:
            t_result = stats.ttest_ind(data1, data2, equal_var=(stats.levene(data1, data2).pvalue > 0.05))
            print(f"Independent t-test:".ljust(35) + 
                  f"Statistic={t_result.statistic:9.3f}, p={t_result.pvalue:.5f}, df={int(t_result.df)}")
            
            # Calculate effect size (Cohen's d)
            r = np.sqrt(t_result.statistic**2 / (t_result.statistic**2 + t_result.df))
            print(f"Effect size (r):".ljust(35) + f"{r:.3f}")
        else:
            # Use non-parametric test (Mann-Whitney U) if data not normal
            mw_result = stats.mannwhitneyu(data1, data2)
            print(f"Mann–Whitney U test:".ljust(35) + 
                  f"Statistic={mw_result.statistic:9.3f}, p={mw_result.pvalue:.5f}")
        print()
    
    # =========================================================================
    # SECTION 1: PREDICTION IMPROVEMENT - AI TRUE/FALSE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 1: PREDICTION IMPROVEMENT - AI TRUE/FALSE ANALYSIS")
    print("=" * 80)
    
    # Extract data for all conditions
    baseline_false = get_condition_data('baseline', 0, 'prediction_improvement')
    xai_false = get_condition_data('xai', 0, 'prediction_improvement')
    baseline_true = get_condition_data('baseline', 1, 'prediction_improvement')
    xai_true = get_condition_data('xai', 1, 'prediction_improvement')
    
    # Comparison 1: Baseline vs. XAI when AI prediction is FALSE
    print("\nComparison 1.1: Baseline vs. XAI (When AI Model Prediction is FALSE)")
    print_descriptive_stats("Baseline (AI=FALSE)", baseline_false, "XAI (AI=FALSE)", xai_false)
    print_assumption_tests("Baseline (AI=FALSE)", baseline_false, "XAI (AI=FALSE)", xai_false)
    run_statistical_test("Baseline (AI=FALSE)", baseline_false, "XAI (AI=FALSE)", xai_false)
    
    # Comparison 2: Baseline vs. XAI when AI prediction is TRUE
    print("\nComparison 1.2: Baseline vs. XAI (When AI Model Prediction is TRUE)")
    print_descriptive_stats("Baseline (AI=TRUE)", baseline_true, "XAI (AI=TRUE)", xai_true)
    print_assumption_tests("Baseline (AI=TRUE)", baseline_true, "XAI (AI=TRUE)", xai_true)
    run_statistical_test("Baseline (AI=TRUE)", baseline_true, "XAI (AI=TRUE)", xai_true)
    
    # Comparison 3: Within Baseline group, FALSE vs. TRUE AI predictions
    print("\nComparison 1.3: FALSE vs. TRUE AI Predictions (Baseline Group)")
    print_descriptive_stats("Baseline (AI=FALSE)", baseline_false, "Baseline (AI=TRUE)", baseline_true)
    print_assumption_tests("Baseline (AI=FALSE)", baseline_false, "Baseline (AI=TRUE)", baseline_true)
    run_statistical_test("Baseline (AI=FALSE)", baseline_false, "Baseline (AI=TRUE)", baseline_true)
    
    # Comparison 4: Within XAI group, FALSE vs. TRUE AI predictions
    print("\nComparison 1.4: FALSE vs. TRUE AI Predictions (XAI Group)")
    print_descriptive_stats("XAI (AI=FALSE)", xai_false, "XAI (AI=TRUE)", xai_true)
    print_assumption_tests("XAI (AI=FALSE)", xai_false, "XAI (AI=TRUE)", xai_true)
    run_statistical_test("XAI (AI=FALSE)", xai_false, "XAI (AI=TRUE)", xai_true)
    
    # =========================================================================
    # SECTION 2: CONFIDENCE IMPROVEMENT - AI TRUE/FALSE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SECTION 2: CONFIDENCE IMPROVEMENT - AI TRUE/FALSE ANALYSIS")
    print("=" * 80)
    
    # Extract data for all conditions
    baseline_false = get_condition_data('baseline', 0, 'confidence_improvement')
    xai_false = get_condition_data('xai', 0, 'confidence_improvement')
    baseline_true = get_condition_data('baseline', 1, 'confidence_improvement')
    xai_true = get_condition_data('xai', 1, 'confidence_improvement')
    
    # Comparison 1: Baseline vs. XAI when AI prediction is FALSE
    print("\nComparison 2.1: Baseline vs. XAI (When AI Model Prediction is FALSE)")
    print_descriptive_stats("Baseline (AI=FALSE)", baseline_false, "XAI (AI=FALSE)", xai_false)
    print_assumption_tests("Baseline (AI=FALSE)", baseline_false, "XAI (AI=FALSE)", xai_false)
    run_statistical_test("Baseline (AI=FALSE)", baseline_false, "XAI (AI=FALSE)", xai_false)
    
    # Comparison 2: Baseline vs. XAI when AI prediction is TRUE
    print("\nComparison 2.2: Baseline vs. XAI (When AI Model Prediction is TRUE)")
    print_descriptive_stats("Baseline (AI=TRUE)", baseline_true, "XAI (AI=TRUE)", xai_true)
    print_assumption_tests("Baseline (AI=TRUE)", baseline_true, "XAI (AI=TRUE)", xai_true)
    run_statistical_test("Baseline (AI=TRUE)", baseline_true, "XAI (AI=TRUE)", xai_true)
    
    # Comparison 3: Within Baseline group, FALSE vs. TRUE AI predictions
    print("\nComparison 2.3: FALSE vs. TRUE AI Predictions (Baseline Group)")
    print_descriptive_stats("Baseline (AI=FALSE)", baseline_false, "Baseline (AI=TRUE)", baseline_true)
    print_assumption_tests("Baseline (AI=FALSE)", baseline_false, "Baseline (AI=TRUE)", baseline_true)
    run_statistical_test("Baseline (AI=FALSE)", baseline_false, "Baseline (AI=TRUE)", baseline_true)
    
    # Comparison 4: Within XAI group, FALSE vs. TRUE AI predictions
    print("\nComparison 2.4: FALSE vs. TRUE AI Predictions (XAI Group)")
    print_descriptive_stats("XAI (AI=FALSE)", xai_false, "XAI (AI=TRUE)", xai_true)
    print_assumption_tests("XAI (AI=FALSE)", xai_false, "XAI (AI=TRUE)", xai_true)
    run_statistical_test("XAI (AI=FALSE)", xai_false, "XAI (AI=TRUE)", xai_true)

    def difficulty_statistical_tests(df):
        """
        Perform statistical tests on prediction and confidence improvements between easy and hard samples.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the experimental data
        """
        
        # Define easy samples
        easy_samples = [
            'KSSlide133', 'NSSlide103', 'KSSlide5', 'WSSlide373',
            'UnaffectedSlide204', '22q11DSSlide210', 'UnaffectedSlide229',
            'AngelmanSlide13', 'NSSlide198'
        ]
        
        # Create indices for different groups
        i_baseline_easy = [i for i in range(0, df.shape[0]) if (df.iloc[i]['SlideName'] in easy_samples) and (df.iloc[i]['Group']=='baseline')]
        i_baseline_hard = [i for i in range(0, df.shape[0]) if (df.iloc[i]['SlideName'] not in easy_samples) and (df.iloc[i]['Group']=='baseline')]
        i_xai_easy = [i for i in range(0, df.shape[0]) if (df.iloc[i]['SlideName'] in easy_samples) and (df.iloc[i]['Group']=='xai')]
        i_xai_hard = [i for i in range(0, df.shape[0]) if (df.iloc[i]['SlideName'] not in easy_samples) and (df.iloc[i]['Group']=='xai')]
        
        # Helper function to extract participant-level means for a specific condition
        def get_condition_data(indices, metric):
            df_group = df.iloc[indices].reset_index(drop=True)
            return pd.DataFrame([
                df_group[df_group['ResponseId']==ResponseId][metric].mean() 
                for ResponseId in sorted(set(df_group['ResponseId'].tolist()))
            ]).squeeze().to_numpy()
        
        # Helper function to print descriptive statistics
        def print_descriptive_stats(name1, data1, name2, data2):
            print(f"{name1} ({len(data1)}) vs. {name2} ({len(data2)})")
            print("-" * 30)
            print(f"{name1} (M = {np.mean(data1):.3f}, SD = {np.std(data1):.3f})")
            print(f"{name2} (M = {np.mean(data2):.3f}, SD = {np.std(data2):.3f})")
            print()
        
        # Helper function to run and display normality and variance tests
        def print_assumption_tests(name1, data1, name2, data2):
            # Shapiro-Wilk test for normality
            sw1 = stats.shapiro(data1)
            sw2 = stats.shapiro(data2)
            print(f"Shapiro-Wilk ({name1})  Statistic={sw1.statistic:9.2f}, p={np.format_float_scientific(sw1.pvalue, unique=False, precision=5)}, {sw1.pvalue>0.05}")
            print(f"Shapiro-Wilk ({name2})  Statistic={sw2.statistic:9.2f}, p={np.format_float_scientific(sw2.pvalue, unique=False, precision=5)}, {sw2.pvalue>0.05}")
            
            # Levene's test for homogeneity of variance
            lev = stats.levene(data1, data2)
            print(f"Levene.                       Statistic={lev.statistic:9.2f}, p={lev.pvalue:.5f}, {lev.pvalue>0.05}")
            print()
        
        # Helper function to run Mann-Whitney U test (non-parametric)
        def run_mann_whitney_test(name1, data1, name2, data2):
            mw_result = stats.mannwhitneyu(data1, data2)
            print(f"Mann–Whitney U test.          Statistic={mw_result.statistic:9.2f}, p={mw_result.pvalue:.5f}")
            print()
            
        # =========================================================================
        # SECTION 3: PREDICTION IMPROVEMENT - SAMPLE DIFFICULTY ANALYSIS
        # =========================================================================
        print("\n" + "=" * 80)
        print("SECTION 3: PREDICTION IMPROVEMENT - SAMPLE DIFFICULTY ANALYSIS")
        print("=" * 80)
        
        # Extract data for all conditions
        baseline_easy_pred = get_condition_data(i_baseline_easy, 'prediction_improvement')
        baseline_hard_pred = get_condition_data(i_baseline_hard, 'prediction_improvement')
        xai_easy_pred = get_condition_data(i_xai_easy, 'prediction_improvement')
        xai_hard_pred = get_condition_data(i_xai_hard, 'prediction_improvement')
        
        # Comparison 1: Baseline Easy vs. XAI Easy
        print("\nComparison 3.1: Baseline Easy vs. XAI Easy")
        print_descriptive_stats("baseline_EASY", baseline_easy_pred, "xai_EASY", xai_easy_pred)
        print_assumption_tests("baseline_EASY", baseline_easy_pred, "xai_EASY", xai_easy_pred)
        run_mann_whitney_test("baseline_EASY", baseline_easy_pred, "xai_EASY", xai_easy_pred)
        
        # Comparison 2: Baseline Hard vs. XAI Hard
        print("\nComparison 3.2: Baseline Hard vs. XAI Hard")
        print_descriptive_stats("baseline_HARD", baseline_hard_pred, "xai_HARD", xai_hard_pred)
        print_assumption_tests("baseline_HARD", baseline_hard_pred, "xai_HARD", xai_hard_pred)
        run_mann_whitney_test("baseline_HARD", baseline_hard_pred, "xai_HARD", xai_hard_pred)
        
        # Comparison 3: Within Baseline group, Easy vs. Hard samples
        print("\nComparison 3.3: Easy vs. Hard samples (Baseline Group)")
        print_descriptive_stats("baseline_EASY", baseline_easy_pred, "baseline_HARD", baseline_hard_pred)
        print_assumption_tests("baseline_EASY", baseline_easy_pred, "baseline_HARD", baseline_hard_pred)
        run_mann_whitney_test("baseline_EASY", baseline_easy_pred, "baseline_HARD", baseline_hard_pred)
        
        # Comparison 4: Within XAI group, Easy vs. Hard samples
        print("\nComparison 3.4: Easy vs. Hard samples (XAI Group)")
        print_descriptive_stats("xai_EASY", xai_easy_pred, "xai_HARD", xai_hard_pred)
        print_assumption_tests("xai_EASY", xai_easy_pred, "xai_HARD", xai_hard_pred)
        run_mann_whitney_test("xai_EASY", xai_easy_pred, "xai_HARD", xai_hard_pred)
        
        # =========================================================================
        # SECTION 4: CONFIDENCE IMPROVEMENT - SAMPLE DIFFICULTY ANALYSIS
        # =========================================================================
        print("\n" + "=" * 80)
        print("SECTION 4: CONFIDENCE IMPROVEMENT - SAMPLE DIFFICULTY ANALYSIS")
        print("=" * 80)
        
        # Extract data for all conditions
        baseline_easy_conf = get_condition_data(i_baseline_easy, 'confidence_improvement')
        baseline_hard_conf = get_condition_data(i_baseline_hard, 'confidence_improvement')
        xai_easy_conf = get_condition_data(i_xai_easy, 'confidence_improvement')
        xai_hard_conf = get_condition_data(i_xai_hard, 'confidence_improvement')
        
        # Comparison 1: Baseline Easy vs. XAI Easy
        print("\nComparison 4.1: Baseline Easy vs. XAI Easy")
        print_descriptive_stats("baseline_EASY", baseline_easy_conf, "xai_EASY", xai_easy_conf)
        print_assumption_tests("baseline_EASY", baseline_easy_conf, "xai_EASY", xai_easy_conf)
        run_mann_whitney_test("baseline_EASY", baseline_easy_conf, "xai_EASY", xai_easy_conf)
        
        # Comparison 2: Baseline Hard vs. XAI Hard
        print("\nComparison 4.2: Baseline Hard vs. XAI Hard")
        print_descriptive_stats("baseline_HARD", baseline_hard_conf, "xai_HARD", xai_hard_conf)
        print_assumption_tests("baseline_HARD", baseline_hard_conf, "xai_HARD", xai_hard_conf)
        run_mann_whitney_test("baseline_HARD", baseline_hard_conf, "xai_HARD", xai_hard_conf)
        
        # Comparison 3: Within Baseline group, Easy vs. Hard samples
        print("\nComparison 4.3: Easy vs. Hard samples (Baseline Group)")
        print_descriptive_stats("baseline_EASY", baseline_easy_conf, "baseline_HARD", baseline_hard_conf)
        print_assumption_tests("baseline_EASY", baseline_easy_conf, "baseline_HARD", baseline_hard_conf)
        run_mann_whitney_test("baseline_EASY", baseline_easy_conf, "baseline_HARD", baseline_hard_conf)
        
        # Comparison 4: Within XAI group, Easy vs. Hard samples
        print("\nComparison 4.4: Easy vs. Hard samples (XAI Group)")
        print_descriptive_stats("xai_EASY", xai_easy_conf, "xai_HARD", xai_hard_conf)
        print_assumption_tests("xai_EASY", xai_easy_conf, "xai_HARD", xai_hard_conf)
        run_mann_whitney_test("xai_EASY", xai_easy_conf, "xai_HARD", xai_hard_conf)

    # Call the nested function to perform difficulty analysis
    difficulty_statistical_tests(df)

def generate_plots_for_version(version):
    """Generate all plots for a specific data version (v1 or v2)"""
    print(f"Generating plots for {version} data...")
    
    # Read data for the specified version
    read_survey_data(version=version)
    
    # Load the generated data
    df = pd.read_csv(f'output/merged_table_{version}.csv')
    
    # Create output directory for this version
    output_dir = pathlib.Path(f'output/{version}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    plot_fig2(df, output_dir)
    plot_fig3(df, output_dir)
    plot_fig3_NEW(df, output_dir)  # Generate new version of Figure 3
    plot_fig4(df, output_dir)
    plot_supp_fig1(df, output_dir)
    plot_supp_fig2(df, output_dir)
    plot_supp_fig3(df, output_dir)
    plot_supp_fig4(df, output_dir)
    plot_supp_fig5(df, output_dir)
    
    # Save correlation analysis to text file
    import io
    import sys
    
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    correlation_analysis(df)
    statistical_tests(df)
    
    sys.stdout = old_stdout
    
    with open(output_dir / f'correlation_statistical_analysis_{version}.txt', 'w') as f:
        f.write(captured_output.getvalue())
    
    print(f"✅ Plots for {version} generated in output/{version}/")

if __name__ == "__main__":
    # Generate plots for both versions
    generate_plots_for_version('v1')
    generate_plots_for_version('v2')
    print("All plots generated successfully!")