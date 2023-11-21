import matplotlib.pyplot as plt
import numpy as np

data = length_list

src = []
ref = []
gpt = []
plan = []
pro = []

# 抽样间隔
sampling_interval = 7

for i, li in enumerate(data):
    if i % sampling_interval == 0:
        src.append(li[0])
        ref.append(li[1])
        gpt.append(li[2])
        plan.append(li[3])
        pro.append(li[4])

# 设置图形的大小和分辨率
fig, ax = plt.subplots(figsize=(5.5, 6.5), dpi=300)

# 设置x轴和y轴的标签
ax.set_xlabel('Source Length(tokens)')
ax.set_ylabel('Result Length(tokens)')

# 绘制散点图
colors = ['green', 'red', 'blue', 'orange']  # 设置每个数据系列的颜色
ax.scatter(src, ref, marker='o', color=colors[0], label='Reference')
ax.scatter(src, gpt, marker='o', color=colors[1], label='ChatGPT')
ax.scatter(src, plan, marker='o', color=colors[2], label='PlanSimp+ICL')
ax.scatter(src, pro, marker='o', color=colors[3], label='ProgDS+ICL+Iteration')

# 添加图例
ax.legend()

plt.savefig("scatter_plot.png", dpi=300)
# 显示图形
plt.show()