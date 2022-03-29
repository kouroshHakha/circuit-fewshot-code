from cProfile import label
import matplotlib.pyplot as plt

import pandas as pd

# from statsmodels.tsa.api import SimpleExpSmoothing
import statsmodels.api as sm


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc('text', usetex=True)


df = pd.read_csv('paper_plots/rladder_acc.csv')
print(df.columns)

rename = {
    'Step': 'step',
    'r2-10_pt_20k - valid_epoch_vdc_acc': 'valid',
    'r2-10_pt_20k - train_epoch_vdc_acc__MAX': 'train',
}

remove_cols = [c for c in df.columns if c not in rename]
df = df.drop(remove_cols, axis=1)
df = df.rename(rename, axis=1)

# ax = df.plot(x='step', y='train')
# df.plot(x='step', y='valid', ax=ax)
# plt.savefig('df.png')

train = sm.nonparametric.lowess(df['train'].values, df['step'].values, frac=0.01, it=1)
valid = sm.nonparametric.lowess(df['valid'].values, df['step'].values, frac=0.1, it=5)

_, axes = plt.subplots(2,1)
ax = axes[0]
ax.plot(train[:, 0], train[:, 1], label='train')
ax.plot(valid[:, 0], valid[:, 1], label='valid')
# ax.set_xlabel('steps')
ax.set_ylabel('Acc@100\n (RLadder)', ha="center")
ax.set_ylim(0, 1)
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.legend(loc='upper center', ncol=2, bbox_to_anchor = (0.5, 1.35))


df = pd.read_csv('paper_plots/opamp_acc.csv')
print(df.columns)

rename = {
    'trainer/global_step': 'step',
    'Grouped runs - valid_vdc_acc': 'valid',
    'Grouped runs - train_vdc_acc__MAX': 'train',
}

remove_cols = [c for c in df.columns if c not in rename]
df = df.drop(remove_cols, axis=1)
df = df.rename(rename, axis=1)

train = sm.nonparametric.lowess(df['train'].values, df['step'].values, frac=0.01, it=1)
valid = sm.nonparametric.lowess(df['valid'].values, df['step'].values, frac=0.1, it=5)

ax = axes[1]
ax.plot(train[:, 0], train[:, 1])
ax.plot(valid[:, 0], valid[:, 1])
ax.set_xlabel('steps')
ax.set_ylabel('Acc@200\n (OpAmp)', ha="center")
ax.set_ylim(0, 1)
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.tight_layout()
plt.savefig('train_curve.png')

breakpoint()