import json
import numpy as np
import matplotlib.pyplot as plt

coverage_rates = 0.7+0.01*np.arange(20)

# with open('records.json') as f:
#     data = json.load(f)

# for tw in data.keys():
#     timeouts = data[tw]["timeouts"]
#     runtimes = data[tw]["runtimes"]
#     counts = data[tw]["counts"]
#     msdsizes = data[tw]["msdsizes"]

#     # Plot 1 : Plot the rates of timeouts at a given timewall for each method against the user-defined coverage rate
#     fig, ax = plt.subplots()

#     ax.plot(coverage_rates, timeouts["mnc"], 'g-', label="Minimum Nonconformity Change")
#     ax.plot(coverage_rates, timeouts["mod"], 'r-', label="Decomposable")
#     ax.plot(coverage_rates, timeouts["filtered"], 'b-', label="Filtered")
#     ax.set_xlabel('User defined coverage rate')
#     ax.set_ylabel('Ratio of timeouts', color='g')
#     plt.legend(loc="upper left")
#     # plt.title('Mean runtime of confidence set enumeration for a single sample with respect to the user defined coverage rate and the enumeration method')
#     plt.savefig(f"plots/timeouts@{tw}.svg")
#     plt.show()

#     # Plot 2 : Plot runtimes of finished runs for each method against the user-defined coverage rate
#     fig, ax = plt.subplots()

#     ax.plot(coverage_rates, runtimes["mnc"], 'g-', label="Minimum Nonconformity Change")
#     ax.plot(coverage_rates, runtimes["mod"], 'r-', label="Decomposable")
#     ax.plot(coverage_rates, runtimes["filtered"], 'b-', label="Filtered")
#     ax.set_xlabel('User defined coverage rate')
#     ax.set_ylabel('Mean runtime')
#     ax.set_yscale('log')
#     plt.legend(loc="upper left")
#     # plt.title('Mean runtime of confidence set enumeration for a single sample with respect to the user defined coverage rate and the enumeration method')
#     plt.savefig(f"plots/runtimes@{tw}.svg")
#     plt.show()

# Plot 3 : Plot the sizes of search spaces for each method against the user-defined coverage rate
with open('sizes.json') as f:
    data = json.load(f)

sizes = data["sizes"]
std = data["stds"]

fig, ax = plt.subplots()

ax.plot(coverage_rates[:15], sizes["mnc"], 'g-', label="Minimum Nonconformity Change")
ax.plot(coverage_rates[:15], sizes["mod"], 'r-', label="Decomposable")
ax.plot(coverage_rates[:15], sizes["filtered"], 'b-', label="Filtered")
ax.set_xlabel('User defined coverage rate')
ax.set_ylabel('Size of the search space', color='g')
ax.set_yscale('log')
plt.legend(loc="upper left")
# plt.title('Average size of search space with respect to the user defined coverage rate')
plt.savefig("plots/searchspacesizes.svg")
plt.show()

# Plot 4 : Plot runtime
with open('runtimes.json') as f:
    data = json.load(f)

runtimes = data["runtimes"]
counts = data["counts"]

lens = {k:len(v) for (k,v) in runtimes.items()}

fig, ax = plt.subplots()

ax.plot(coverage_rates[:lens["mnc"]], runtimes["mnc"], 'g-', label="Minimum Nonconformity Change")
ax.plot(coverage_rates[:lens["dec"]], runtimes["dec"], 'r-', label="Decomposable")
ax.plot(coverage_rates[:lens["filtered"]], runtimes["filtered"], 'b-', label="Filtered")
ax.set_xlabel('User defined coverage rate')
ax.set_ylabel('Average runtimes')
plt.legend(loc="upper left")
ax.set_yscale('log')
# plt.title('Average runtime with respect to the user defined coverage rate')
plt.savefig("plots/runtimes.svg")
plt.show()

# Plot 4 : Ratio between the size of search space and runtime
# fig, ax = plt.subplots()

# ax.plot(coverage_rates[:lens["mnc"]], np.divide(np.array(runtimes["mnc"]), np.array(sizes["mnc"][:lens["mnc"]])), 'g-', label="Minimum Nonconformity Change")
# ax.plot(coverage_rates[:lens["dec"]], np.divide(np.array(runtimes["dec"]), np.array(sizes["mod"][:lens["dec"]])), 'r-', label="Decomposable")
# # ax.plot(coverage_rates[:lens["filtered"]], np.divide(np.array(runtimes["filtered"]), np.array(sizes["filtered"][:lens["filtered"]])), 'b-', label="Filtered")
# ax.set_xlabel('User defined coverage rate')
# ax.set_ylabel('Ratio between runtime and search space')
# plt.legend(loc="upper left")
# # plt.title('Mean runtime of confidence set enumeration for a single sample with respect to the user defined coverage rate and the enumeration method')
# plt.savefig("plots/ratios.svg")
# plt.show()

