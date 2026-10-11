"""Development-only Tree SHAP interpretation of the saved final booster."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=root/'results/interpretability/shap')
    parser.add_argument('--figure-dir', type=Path, default=root/'results/figures')
    parser.add_argument('--report', type=Path, default=root/'docs/SHAP_ANALYSIS.md')
    args = parser.parse_args()
    figure_names = ['shap_global_top20.png', 'shap_class_comparison_heatmap.png'] + [f'shap_class_{c}_{kind}.png' for c in range(1,6) for kind in ['top10','beeswarm']]
    if (args.output_dir.exists() and any(args.output_dir.iterdir())) or args.report.exists() or any((args.figure_dir/n).exists() for n in figure_names):
        raise FileExistsError('Use new output paths; existing analysis outputs will not be overwritten')
    np.random.seed(42)
    model_path=root/'results/final/model.json'
    model=xgb.Booster();model.load_model(model_path);model.set_param({'nthread':4})
    features=[f'x{i}' for i in range(1,179)]
    if model.feature_names!=features or model.num_features()!=178:
        raise ValueError('Model feature mapping does not exactly match x1–x178')
    metadata=json.loads((root/'results/final/metrics.json').read_text())['metadata']
    if metadata['class_order']!=list(range(1,6)) or metadata['parameters']['objective']!='multi:softprob' or metadata['parameters']['num_class']!=5:
        raise ValueError('Unexpected saved multiclass metadata')
    split=pd.read_csv(root/'data/shared_train_test_split.csv',encoding='utf-16')
    dev_idx=split.loc[split.split=='development','row_index'].to_numpy(dtype=int)
    if len(dev_idx)!=9200 or len(set(dev_idx))!=9200:
        raise ValueError('Unexpected development split')
    # Only development labels are accessed; test rows are excluded before sampling.
    development=pd.read_csv(root/'data/data.csv').iloc[dev_idx].copy()
    if development.y.value_counts().to_dict()!={c:1840 for c in range(1,6)}:
        raise ValueError('Unexpected development class balance')
    rng=np.random.default_rng(42)
    rows=np.concatenate([rng.choice(development.index[development.y==c].to_numpy(),size=200,replace=False) for c in range(1,6)])
    rows.sort()
    sample=development.loc[rows];X=sample[features]
    print('Verified 178 features; sampled 1,000 development rows, 200/class; expected SHAP shape (1000,178,5).',flush=True)
    explainer=shap.TreeExplainer(model,model_output='raw',feature_perturbation='tree_path_dependent')
    returned=explainer.shap_values(X,check_additivity=True)
    # SHAP >=0.45 uses samples/features/outputs; support older class-list format explicitly.
    if isinstance(returned,list):
        original_shape=[list(np.shape(v)) for v in returned];values=np.stack(returned,axis=-1)
    else:
        original_shape=list(np.shape(returned));values=np.asarray(returned)
    if values.shape!=(1000,178,5):
        raise ValueError(f'Unexpected SHAP shape: {values.shape}')
    base=np.asarray(explainer.expected_value)
    margins=model.predict(xgb.DMatrix(X),output_margin=True,strict_shape=True)
    error=float(np.max(np.abs(values.sum(axis=1)+base-margins)))
    if not np.allclose(values.sum(axis=1)+base,margins,atol=1e-4,rtol=1e-5):
        raise ValueError('Raw-margin additivity check failed')
    # Pool equally across the 1,000 sampled windows and all five model outputs.
    by_class=np.abs(values).mean(axis=0)
    global_gain=by_class.mean(axis=1)
    table=pd.DataFrame({'feature':features,'position':range(1,179),'mean_abs_shap':global_gain})
    for c in range(1,6):table[f'class_{c}_mean_abs_shap']=by_class[:,c-1]
    ranked=table.sort_values(['mean_abs_shap','position'],ascending=[False,True]).reset_index(drop=True)
    ranked.insert(0,'rank',range(1,179))
    args.output_dir.mkdir(parents=True,exist_ok=True);args.figure_dir.mkdir(parents=True,exist_ok=True)
    ranked.to_csv(args.output_dir/'global_feature_importance.csv',index=False)
    sample[['column_a','y']].rename_axis('row_index').reset_index().to_csv(args.output_dir/'sample_membership.csv',index=False)
    np.savez_compressed(args.output_dir/'shap_values.npz',values=values,base_values=base,row_indices=rows,feature_names=np.array(features),class_labels=np.arange(1,6))
    class_tables=[]
    for c in range(1,6):
        t=table[['feature','position',f'class_{c}_mean_abs_shap']].rename(columns={f'class_{c}_mean_abs_shap':'mean_abs_shap'})
        t=t.sort_values(['mean_abs_shap','position'],ascending=[False,True]).reset_index(drop=True)
        t.insert(0,'rank',range(1,179));t.insert(0,'class',c);class_tables.append(t)
    pd.concat(class_tables).to_csv(args.output_dir/'class_feature_importance.csv',index=False)
    gain=pd.read_csv(root/'results/interpretability/feature_importance_gain_ranked.csv')
    comparison=ranked[['feature','rank','mean_abs_shap']].rename(columns={'rank':'shap_rank'}).merge(gain[['feature','rank','gain']].rename(columns={'rank':'gain_rank'}),on='feature',validate='one_to_one')
    comparison['shap_top20']=comparison.shap_rank<=20;comparison['gain_top20']=comparison.gain_rank<=20
    comparison.to_csv(args.output_dir/'shap_vs_gain.csv',index=False)
    common=comparison[comparison.shap_top20 & comparison.gain_top20]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titlepad':14})
    def bar(t,title,path):
        fig,ax=plt.subplots(figsize=(8,7 if len(t)==20 else 4.7),layout='constrained')
        bars=ax.barh(t.feature,t.mean_abs_shap,color='#0072B2');ax.invert_yaxis()
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in t.mean_abs_shap],padding=4,fontsize=9)
        ax.set(xlabel='Mean absolute SHAP value (raw-margin units)',title=title,xlim=(0,t.mean_abs_shap.max()*1.2))
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=0.2);ax.set_axisbelow(True)
        fig.savefig(path,dpi=300,bbox_inches='tight');plt.close(fig)
    bar(ranked.head(20),'Global Tree SHAP importance: top 20',args.figure_dir/'shap_global_top20.png')
    for c,t in enumerate(class_tables,1):
        bar(t.head(10),f'Class {c}: top 10 Tree SHAP features',args.figure_dir/f'shap_class_{c}_top10.png')
        # A single class slice is a valid 2D explanation; no multiclass stacking in beeswarms.
        np.random.seed(42)
        shap.summary_plot(values[:,:,c-1],X,plot_type='dot',max_display=20,show=False,plot_size=(9,7))
        plt.title(f'Class {c}: development-sample Tree SHAP')
        plt.xlabel('SHAP value (raw class margin)')
        plt.savefig(args.figure_dir/f'shap_class_{c}_beeswarm.png',dpi=300,bbox_inches='tight');plt.close('all')
    top=ranked.head(20)
    heat=top[[f'class_{c}_mean_abs_shap' for c in range(1,6)]].to_numpy()
    fig,ax=plt.subplots(figsize=(7,8),layout='constrained');im=ax.imshow(heat,aspect='auto',cmap='Blues',vmin=0)
    ax.set(xticks=range(5),xticklabels=[f'Class {c}' for c in range(1,6)],yticks=range(20),yticklabels=top.feature,title='Class-specific importance of global top 20 features',xlabel='Model output class',ylabel='EEG feature')
    for i in range(20):
        for j in range(5):ax.text(j,i,f'{heat[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if heat[i,j]>heat.max()*0.55 else '#172b3a')
    fig.colorbar(im,ax=ax,label='Mean absolute SHAP (raw-margin units)');fig.savefig(args.figure_dir/'shap_class_comparison_heatmap.png',dpi=300,bbox_inches='tight');plt.close(fig)
    inputs=[model_path,root/'data/data.csv',root/'data/shared_train_test_split.csv',root/'results/interpretability/feature_importance_gain_ranked.csv']
    summary={'shap_version':shap.__version__,'xgboost_version':xgb.__version__,'sample_size':1000,'samples_per_class':200,'seed':42,'raw_return_shape':original_shape,'canonical_shape':list(values.shape),'axis_order':['sample','feature','class'],'class_mapping':{str(i):i+1 for i in range(5)},'base_values':base.tolist(),'maximum_additivity_error':error,'global_aggregation':'mean(abs(SHAP), axes=(samples,classes))','perturbation':'tree_path_dependent','output_units':'raw class margins, not probabilities','top20_overlap_count':len(common),'common_top20_features':common.feature.tolist(),'input_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'training_performed':False,'test_rows_explained':0}
    (args.output_dir/'analysis_metadata.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Tree SHAP analysis of the final XGBoost model','','## Compatibility, sampling and class mapping','',f'SHAP {shap.__version__} and XGBoost {xgb.__version__} passed a two-development-row compatibility check before the full analysis. Model feature names exactly match x1–x178 in order.','',
    'A NumPy default_rng(42) sampled 200 rows without replacement from each development class (1,840 available per class). The resulting 1,000 indices were sorted in original data order. Membership is saved for reproducibility. No test samples or test labels were used; existing split assignments are unchanged. This is class-balanced window sampling, not independent-recording sampling.','',
    f'TreeExplainer returned shape **{tuple(values.shape)}**, ordered **samples × features × classes**. Base values have shape {tuple(base.shape)}. Output indices 0–4 correspond to Classes 1–5, based on the saved training pipeline’s y−1 encoding and final metadata class order. The booster alone does not preserve original clinical label meanings.','',
    f'For every explained row and output, the SHAP sum plus its base value reconstructs the saved model raw margin (maximum absolute error {error:.8g}; checked with atol=1e-4, rtol=1e-5). This verifies dimensions and output alignment.','',
    'Tree-path-dependent TreeExplainer uses training path counts in the fitted trees; no new background dataset was supplied. Values explain raw class margins, not probability contributions. A positive value increases that class margin relative to its expected margin; it does not directly equal a probability increase.','',
    '## Global importance','','Global importance is mean absolute SHAP across all 1,000 samples and all five class outputs. Absolute values are taken before averaging. Each class-specific table averages over the same 1,000 samples, not just rows whose true label is that class. Ties are resolved by feature position.','','| Rank | Feature | Mean absolute SHAP |','|---:|---|---:|']
    for r in top.itertuples():lines.append(f'| {r.rank} | {r.feature} | {r.mean_abs_shap:.6f} |')
    lines+=['','![Global top 20](../results/figures/shap_global_top20.png)','','## Class-specific importance','', '![Class comparison](../results/figures/shap_class_comparison_heatmap.png)','', 'The heatmap uses one shared color scale and the global top-20 feature order. Each cell is mean absolute SHAP for one output class. This shows relative model attribution, without assigning clinical meanings to classes.']
    for c,t in enumerate(class_tables,1):
        lines+=['',f'### Class {c}','','| Rank | Feature | Mean absolute SHAP |','|---:|---|---:|']
        for r in t.head(10).itertuples():lines.append(f'| {r.rank} | {r.feature} | {r.mean_abs_shap:.6f} |')
        lines+=['',f'![Class {c} top 10](../results/figures/shap_class_{c}_top10.png)','',f'![Class {c} beeswarm](../results/figures/shap_class_{c}_beeswarm.png)']
    lines+=['','## Beeswarm interpretation','','Each beeswarm is created from a verified (1000,178) single-class SHAP slice. Dots show sampled-window contributions; color represents the EEG feature value (low to high). Horizontal position is signed raw-margin SHAP value. Different plots can have different horizontal scales. No ambiguous multiclass beeswarm array is passed to SHAP. These plots are descriptive associations in the fitted model.','',
    '## Comparison with gain','',f'**{len(common)} of the top 20 features appear in both rankings**: '+', '.join(common.feature)+'.','','| Feature | SHAP rank | Gain rank |','|---|---:|---:|']
    for r in common.itertuples():lines.append(f'| {r.feature} | {r.shap_rank} | {r.gain_rank} |')
    lines+=['','Gain measures average split loss reduction; SHAP measures average prediction attribution magnitude for the chosen sample and explainer assumptions. Their units differ, and ranks need not agree. The complete rank comparison is saved as CSV.','',
    '## Limits and files','','**SHAP values explain the fitted model’s predictions; they do not establish causal or clinical importance.** Correlated EEG positions can share or exchange attribution. Results depend on sampling and tree-path-dependent assumptions, and do not quantify uncertainty. Existing inferred recording-group overlap remains unresolved. No feature selection, model modification, training, tuning or performance re-evaluation was performed.','',
    'Code: `src/analyze_shap.py`. Numerical outputs under `results/interpretability/shap/`: full global and class rankings, sampled indices/labels, compressed SHAP arrays, gain comparison and analysis metadata. All 12 figures are 300 dpi PNGs under `results/figures/`.','',
    'Reproduction requires shap==0.49.1 and the existing project dependencies plus Matplotlib. In this cloud instance:','','```sh','MPLCONFIGDIR=/workspace/scratch/mpl-cache PYTHONPATH=/workspace/scratch/shap-deps:/workspace/scratch/xgboost-baseline-deps python src/analyze_shap.py','```','','The script refuses existing analysis outputs. Use new `--output-dir`, `--figure-dir`, and `--report` paths for regeneration. Existing experiment outputs are read only.','']
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
