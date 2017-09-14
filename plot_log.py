from matplotlib import pyplot as plt
import json
import numpy as np

rewards = []
rew_means = []
ep_len = 0

jaco = "logs/openai-2017-09-06-18-41-55/monitor.json"
r2500 = "logs/openai-2017-09-06-16-04-39/monitor.json"
r5000 = "logs/openai-2017-09-07-17-42-02/monitor.json"
r1000 = "logs/openai-2017-09-07-17-46-17/monitor.json"

trainings = [r2500, r1000, r5000]
episode_lengths = [500, 200, 1000]
results = []

def slice_mean(arr, slice_len):
    res = []
    for i in range(int(len(arr)/slice_len)):
        arr_part = np.array(arr[(i*slice_len):(i+1)*slice_len])
        res.append(arr_part.mean())
    return res

for i,tr in enumerate(trainings):
    with open(tr) as file:
        for line in file:
            j = json.loads(line)
            if "r" in j:
                rew = j["r"]
                if np.isnan(rew):
                    break
                else:
                    rewards.append(rew)
            if "l" in j:
                ep_len = j["l"]

    results.append(slice_mean(rewards, episode_lengths[i]))
    plt.plot(results[i])
    rewards = []
plt.show()
