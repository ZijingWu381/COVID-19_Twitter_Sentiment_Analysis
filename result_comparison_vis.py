#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:51:11 2020

@author: Zijing Wu (Miles)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()

columns = ["models", "accuracy %", "F1 (Positive %)", "F1 (Negative %)"]
test_data = pd.DataFrame.from_records([("LR", .7354, .7425, .7278),\
                                       ("NB", .7716, .7670, .7760),\
                                      ("SVM", .7894, .7732, .8061),\
                                          ("BiLSTM", .8384, .8441, .8321)],\
                                       columns=columns)

# working example but with unreadable values_a and values_b
test_data_melted = pd.melt(test_data, id_vars=columns[0],\
                           var_name="metrics", value_name="score %")
g = sns.barplot(x=columns[0], y="score %", hue="metrics",\
                data=test_data_melted)
plt.ylim(0.6, 1)
plt.show()


