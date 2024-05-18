#!/usr/bin/env python
# coding: utf-8

# In[158]:


from glob import glob

model_result_paths = glob("./data/judgements/*/*/*.json")


# In[159]:


eval_dataset_dict = {
    "elyza__ELYZA-tasks-100": "ELYZA-tasks-100",
    "yuzuai__rakuda-questions": "Rakuda",
    "lightblue__tengu_bench": "Tengu-Bench",
    "shisa-ai__ja-mt-bench-1shot": "MT-Bench",
}


# In[160]:


import pandas as pd

all_result_dfs = []

for model_result_path in model_result_paths:
    temp_df = pd.read_json(model_result_path, lines=True)
    temp_df["judge_model"] = model_result_path.split("/")[3]
    temp_df["eval_dataset"] = eval_dataset_dict[model_result_path.split("/")[4]]
    temp_df["model_name"] = model_result_path.split("/")[5].replace(".json", "")
    
    all_result_dfs.append(temp_df)


# In[161]:


import pandas as pd

all_result_df = pd.concat(all_result_dfs)

all_result_df["dataset_category"] = all_result_df["eval_dataset"] + " " + all_result_df["Category"]

# In[162]:


eval_dataset_names = all_result_df.eval_dataset.unique()
model_names = all_result_df.model_name.unique()


# In[163]:


eval_corr_results = {}
for eval_dataset_name in eval_dataset_names:
    eval_corr_results[eval_dataset_name] = {}
    for model_name in model_names:
        eval_corr_results[eval_dataset_name][model_name] = all_result_df[(all_result_df.eval_dataset == eval_dataset_name) & (all_result_df.model_name == model_name)].score.mean()


# In[164]:


pd.DataFrame(eval_corr_results).corr().round(4)


# In[166]:


eval_res_df = pd.DataFrame(eval_corr_results)

eval_res_df['ELYZA-tasks-100'] = eval_res_df['ELYZA-tasks-100'] * 2

eval_res_df['mean'] = eval_res_df.mean(axis=1)

eval_res_df = eval_res_df.sort_values(by='mean', ascending=False)

def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

styled_df = eval_res_df.style.apply(highlight_max, axis=0)
styled_df = styled_df.format("{:.2f}")
styled_df


# In[ ]:





# In[178]:


# Latex形式で出力
# print(eval_res_df.to_latex(float_format="%.4f").replace("__","/").replace("_","\_"))


# In[37]:


# 相関を可視化
# cols = eval_res_df.columns

# for c1 in range(len(cols)):
#     for c2 in range(c1, len(cols)):
#         eval_res_df.plot(kind='scatter', x=cols[c1], y=cols[c2], title=f'{cols[c1]} vs {cols[c2]}')


# In[179]:


import re
def get_model_size(x):
    if x.name == "openchat__openchat-3.5-0106":
        return 7
    elif x.name == "CohereForAI__c4ai-command-r-v01":
        return 35
    try:
        return int(re.search("\d{1,2}[bB]",x.name).group(0)[:-1])
    except:
        return None


# In[144]:


model_size_df = eval_res_df.copy()
model_size_df['model_size'] = model_size_df.apply(get_model_size, axis=1)
size_df = model_size_df.dropna(subset=['model_size']).sort_values(by='model_size', ascending=False).groupby('model_size').mean()


# In[145]:


size_df['model_size'] = size_df.index


# In[147]:


log_size_df = size_df


# In[148]:


from math import log
log_size_df["model_size"] = log_size_df["model_size"].apply(lambda x: log(x))


# In[152]:


import matplotlib.pyplot as plt
import seaborn as sns

# 図のサイズを設定
plt.figure(figsize=(10, 6))

# データごとに違う色で散布図と回帰直線を描画
colors = ['blue', 'green', 'red', 'gold']  # 色を指定
for i, column in enumerate(['ELYZA-tasks-100', 'Rakuda', 'Tengu-Bench', 'MT-Bench']):
    sns.regplot(x='model_size', y=column, data=log_size_df, scatter=True, 
                label=column, ci=None, )


plt.xticks([log(7), log(13), log(32), log(70)], ["7B", "13B", "32B", "70B"])
# 凡例を表示
plt.legend()
# タイトルを設定
plt.title('Model Size vs Scores with Regression Lines')
# グラフを表示
plt.savefig("model-size_vs_score.svg")
plt.show()


# In[198]:


print(all_result_df[all_result_df['eval_dataset']=="Tengu-Bench"].groupby("Category").score.mean().sort_values(ascending=False).to_frame().to_latex())


# In[4]:


all_result_df.groupby(
    ["model_name", "eval_dataset"]
).score.mean().reset_index(drop=False).pivot_table(values="score", index="model_name", columns="eval_dataset")


# In[199]:


mean_df = all_result_df.groupby(
    ["eval_dataset", "Category", "model_name"]
).score.mean().reset_index(level=1, drop=False).pivot_table(index='model_name', columns=['eval_dataset', 'Category'], values='score')


# In[205]:


mean_df


# In[204]:


print(mean_df.to_markdown(index=True, floatfmt='.2f'))


# In[200]:


mean_df


# In[213]:


mean_df = all_result_df.groupby(
    ["model_name", "eval_dataset"]
).score.mean().reset_index(level=1, drop=False)


# In[217]:


mean_df['score_'] = mean_df.apply(lambda x: x.score*2 if x.eval_dataset=="ELYZA-tasks-100" else x.score, axis=1)

mean_df.to_csv('output.csv', index=True)

import sys
sys.exit()

# In[214]:


unique_models = mean_df.index.unique()


# In[219]:


import plotly.graph_objects as go
from plotly import offline

fig = go.Figure()

for unique_model in unique_models:
    
    model_mean_df = mean_df.loc[unique_model]
    
    fig.add_trace(go.Scatterpolar(
          r=model_mean_df["score_"],
          theta=model_mean_df["eval_dataset"],
          fill='toself',
          name=unique_model
    ))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 10]
    )),
  showlegend=True
)

# fig.show()
offline.plot(fig)


# In[ ]:




