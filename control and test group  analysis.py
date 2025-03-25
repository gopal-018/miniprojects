#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

control_group = pd.read_csv("control_group.csv", sep=";")
experiment_group = pd.read_csv("test_group.csv", sep=";")


# In[34]:


print(" Control Group Sample Data:")
print(control_group.head(), "\n")
print(" Experiment Group Sample Data:")
print(experiment_group.head(), "\n")


# In[17]:


numeric_cols = ["Spend [USD]", "# of Impressions", "Reach", "# of Website Clicks", 
                "# of Searches", "# of View Content", "# of Add to Cart", "# of Purchase"]

for col in numeric_cols:
    control_group[col] = pd.to_numeric(control_group[col], errors='coerce')
    experiment_group[col] = pd.to_numeric(experiment_group[col], errors='coerce')


control_group.fillna(0, inplace=True)
experiment_group.fillna(0, inplace=True)

control_group["conversion_rate"] = control_group["# of Purchase"] / control_group["# of Website Clicks"]
experiment_group["conversion_rate"] = experiment_group["# of Purchase"] / experiment_group["# of Website Clicks"]


control_group.replace([np.inf, -np.inf], np.nan, inplace=True)
experiment_group.replace([np.inf, -np.inf], np.nan, inplace=True)
control_group.dropna(subset=["conversion_rate"], inplace=True)
experiment_group.dropna(subset=["conversion_rate"], inplace=True)


# In[33]:



print(" Control Group Summary Statistics:")
print(control_group.describe(), "\n")
print(" Experiment Group Summary Statistics:")
print(experiment_group.describe(), "\n")


# In[32]:



plt.figure(figsize=(12, 6))
sns.histplot(control_group['conversion_rate'], label='Control Group', kde=True, color='blue', bins=30, alpha=0.6)
sns.histplot(experiment_group['conversion_rate'], label='Experiment Group', kde=True, color='red', bins=30, alpha=0.6)
plt.legend()
plt.title(" Conversion Rate Distribution (Histogram & KDE)")
plt.xlabel("Conversion Rate")
plt.ylabel("Frequency")
plt.show()


# In[20]:



plt.figure(figsize=(10, 6))
sns.boxplot(data=[control_group["conversion_rate"], experiment_group["conversion_rate"]], palette=["blue", "red"])
plt.xticks([0, 1], ["Control Group", "Experiment Group"])
plt.title("📊 Conversion Rate Comparison (Boxplot)")
plt.ylabel("Conversion Rate")
plt.show()


# In[31]:



plt.figure(figsize=(8, 5))
avg_conversion_rates = [control_group["conversion_rate"].mean(), experiment_group["conversion_rate"].mean()]
sns.barplot(x=["Control Group", "Experiment Group"], y=avg_conversion_rates, palette=["blue", "red"])
plt.title(" Average Conversion Rate")
plt.ylabel("Conversion Rate")
plt.show()


# In[22]:


control_group['Date'] = pd.to_datetime(control_group['Date'], errors='coerce', dayfirst=True)
experiment_group['Date'] = pd.to_datetime(experiment_group['Date'], errors='coerce', dayfirst=True)

control_trend = control_group.groupby("Date")["conversion_rate"].mean()
experiment_trend = experiment_group.groupby("Date")["conversion_rate"].mean()


# In[30]:


plt.figure(figsize=(12, 6))
plt.plot(control_trend, label="Control Group", color='blue', marker='o')
plt.plot(experiment_trend, label="Experiment Group", color='red', marker='o')
plt.legend()
plt.title(" Conversion Rate Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Conversion Rate")
plt.xticks(rotation=45)
plt.show()


# In[29]:


stat, p_value = stats.ttest_ind(control_group['conversion_rate'].dropna(), experiment_group['conversion_rate'].dropna())
print(f" T-test Statistic: {stat:.4f}, P-value: {p_value:.4f}")


# In[28]:


mean_diff = experiment_group["conversion_rate"].mean() - control_group["conversion_rate"].mean()
pooled_std = np.sqrt((control_group["conversion_rate"].std()**2 + experiment_group["conversion_rate"].std()**2) / 2)
cohens_d = mean_diff / pooled_std
print(f" Effect Size (Cohen's d): {cohens_d:.4f}")


# In[27]:


if p_value < 0.05:  
    if cohens_d > 0.5:
        print(" The new version has a **statistically significant** impact with a **moderate to large effect**.")
        print(" Conversion rate trend suggests an **improvement**.")
        print(" **Recommendation: Upgrade to the new version.**")
    elif cohens_d < -0.5:
        print(" The new version has a **statistically significant negative impact**.")
        print(" Conversion rate has **decreased** compared to the control group.")
        print(" **Recommendation: Revert to the old version.**")
    else:
        print(" The new version has a significant impact, but the effect size is **small**.")
        print(" Consider **other business factors** before making a decision.")
else:
    print(" There is **no statistically significant difference** between the control and experiment groups.")
    print(" The new version **does not add enough value** to justify a change.")
    print(" **Recommendation: Stick to the current version.**")


# In[ ]:




