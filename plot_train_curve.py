import torch
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})  # 设置全局字体大小
plt.rcParams['figure.dpi'] = 600  # 设置全局图像 DPI

# 尝试使用文泉驿微米黑（WenQuanYi Micro Hei），Arch Linux 默认有
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Serif CN']
plt.rcParams['axes.unicode_minus'] = False
history = torch.load("training_history.pth")


# 奖励曲线
plt.figure()
plt.plot(history["rewards"])
plt.title("奖励曲线")
plt.xlabel("回合")
plt.ylabel("奖励值")
plt.tight_layout()
plt.savefig("reward_curve.png")

# 优势值曲线
plt.figure()
plt.plot(history["advantages"])
plt.title("优势值曲线")
plt.xlabel("回合")
plt.ylabel("优势值")
plt.tight_layout()
plt.savefig("advantage_curve.png")

# 熵值曲线
plt.figure()
plt.plot(history["entropies"])
plt.title("熵值曲线")
plt.xlabel("回合")
plt.ylabel("熵")
plt.tight_layout()
plt.savefig("entropy_curve.png")

# 成本曲线
plt.figure()
plt.plot(history["costs"])
plt.title("成本曲线")
plt.xlabel("回合")
plt.ylabel("路径成本")
plt.tight_layout()
plt.savefig("cost_curve.png")

print("四张训练曲线已分别保存为 reward_curve.png, advantage_curve.png, entropy_curve.png, cost_curve.png")