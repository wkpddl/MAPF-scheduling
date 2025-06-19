import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Source Han Serif CN']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 20})  # 设置全局字体大小
plt.rcParams['figure.dpi'] = 600  # 设置全局图像 DPI

class Task:
    def __init__(self, id, priority, timestamp):
        self.id = id
        self.priority = priority  # 1或3
        self.timestamp = timestamp

def assign_robot_to_task(robot, task, timestep, log):
    log.append({
        "timestep": timestep,
        "robot": robot,
        "task_id": task.id,
        "priority": task.priority,
        "task_timestamp": task.timestamp
    })

priority_queue = deque()
normal_queue = deque()
priority_threshold = 2  # 3为高优先级，1为普通
log = []

def add_task(task):
    if task.priority > priority_threshold:
        priority_queue.append(task)
    else:
        normal_queue.append(task)

def schedule_cycle(robots_num, timestep, log):
    # 第一次分配：优先高优先级，不够用低优先级补齐
    robots = [f"R{i}" for i in range(robots_num)]
    sorted_priority = sorted(priority_queue, key=lambda t: t.timestamp)
    sorted_normal = sorted(normal_queue, key=lambda t: t.timestamp)

    num_high = min(len(sorted_priority), robots_num)
    for idx in range(num_high):
        task = sorted_priority[idx]
        robot = robots[idx]
        assign_robot_to_task(robot, task, timestep, log)
        priority_queue.remove(task)
    robots_left = robots[num_high:]
    if robots_left:
        num_low = min(len(sorted_normal), len(robots_left))
        for idx in range(num_low):
            task = sorted_normal[idx]
            robot = robots_left[idx]
            assign_robot_to_task(robot, task, timestep, log)
            normal_queue.remove(task)
    print(f"时间步{timestep}：分配了{num_high + (num_low if robots_left else 0)}个任务")

    # 第二次分配：只分配低优先级
    robots = [f"R{i}" for i in range(robots_num)]
    sorted_normal = sorted(normal_queue, key=lambda t: t.timestamp)
    num_low = min(len(sorted_normal), robots_num)
    for idx in range(num_low):
        task = sorted_normal[idx]
        robot = robots[idx]
        assign_robot_to_task(robot, task, timestep + 1, log)
        normal_queue.remove(task)
    print(f"时间步{timestep+1}：分配了{num_low}个任务")

    # 第三次分配：只分配低优先级
    robots = [f"R{i}" for i in range(robots_num)]
    sorted_normal = sorted(normal_queue, key=lambda t: t.timestamp)
    num_low = min(len(sorted_normal), robots_num)
    for idx in range(num_low):
        task = sorted_normal[idx]
        robot = robots[idx]
        assign_robot_to_task(robot, task, timestep + 2, log)
        normal_queue.remove(task)
    print(f"时间步{timestep+2}：分配了{num_low}个任务")

def plot_delay_hist(delays, title, color, max_delay, filename):
    plt.figure(figsize=(8, 5), dpi=120)
    plt.hist(delays, bins=range(0, max_delay+2), edgecolor='black', alpha=0.8, color=color)
    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel("任务延迟步数", fontsize=18)
    plt.ylabel("任务数量", fontsize=18)
    plt.axvline(max_delay, color='red', linestyle='--', linewidth=2, label=f'最大延迟={max_delay}')
    plt.legend(fontsize=16)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

if __name__ == "__main__":
    robots_num = 20
    total_timesteps = 200
    task_id = 0
    p_high = 0.3  # 高优先级任务生成概率

    for timestep in range(0, total_timesteps, 3):
        # 每个调度周期流入新任务（可多任务流入）
        for _ in range(random.randint(20, 30)):
            if random.random() < p_high:
                add_task(Task(task_id, 3, timestep))
            else:
                add_task(Task(task_id, 1, timestep))
            task_id += 1

        schedule_cycle(robots_num, timestep, log)

    # 打印分配日志
    for record in log:
        print(f"时间步{record['timestep']}：机器人{record['robot']}分配到任务{record['task_id']}（优先级{record['priority']}，生成于{record['task_timestamp']}）")

    # 统计所有任务的延迟
    delays = [record['timestep'] - record['task_timestamp'] for record in log]
    max_delay = max(delays)
    print(f"\n所有任务最大延迟：{max_delay} 步")

    # 分别统计高优先级和普通任务的延迟
    high_delays = [record['timestep'] - record['task_timestamp'] for record in log if record['priority'] == 3]
    low_delays = [record['timestep'] - record['task_timestamp'] for record in log if record['priority'] == 1]

    # 绘制学术风格延迟分布图
    plot_delay_hist(
        delays,
        "所有任务延迟分布直方图",
        color='mediumseagreen',
        max_delay=max_delay,
        filename="all_task_delay.png"
    )

    if high_delays:
        plot_delay_hist(
            high_delays,
            "高优先级任务延迟分布直方图",
            color='royalblue',
            max_delay=max(high_delays),
            filename="high_priority_task_delay.png"
        )

    if low_delays:
        plot_delay_hist(
            low_delays,
            "普通任务延迟分布直方图",
            color='orange',
            max_delay=max(low_delays),
            filename="low_priority_task_delay.png"
        )