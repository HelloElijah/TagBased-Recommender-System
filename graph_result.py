import matplotlib.pyplot as plt
# conda activate base

# K=9, alpha=0.02, lambda=0.005
# x = [1,2,3,4,5,6,7,8,9,10]
# no_tag =   [1.18306, 1.17426, 1.15719, 1.15570, 1.16489, 1.17442, 1.19016, 1.19881, 1.20211, 1.20776]
# item_tag = [1.16492, 1.14126, 1.11733, 1.10972, 1.11548, 1.12870, 1.13622, 1.15109, 1.15613, 1.16964]
# both_tag = [1.15237, 1.12283, 1.11252, 1.10959, 1.10985, 1.11261, 1.12469, 1.14383, 1.14887, 1.16486]
# weight   = [1.15717, 1.12646, 1.09985, 1.09594, 1.09466, 1.11263, 1.12611, 1.13272, 1.16063, 1.16932]
# plt.plot(x, no_tag, color='black', label='No Tag')
# plt.plot(x, item_tag, color='blue', label='Item Tag')
# plt.plot(x, both_tag, color='green', label='All Tag')
# plt.plot(x, weight, color='red', label='weight')
# plt.xlabel('Number of Iteration')
# plt.ylabel('Mean Square Error')
# plt.title('K=9, alpha=0.02, lambda=0.005')
# plt.legend()
# plt.show()

# iteration=4, lambda=0.005, Weight Model
# x = [1,2,3,4,5,6,7,8,9,10]
# alpha_0015 = [1.17996, 1.15131, 1.13531, 1.12068, 1.10091, 1.09602, 1.09704, 1.09541, 1.10711, 1.11795]
# alpha_002  = [1.13725, 1.13933, 1.13146, 1.12098, 1.11627, 1.11256, 1.09722, 1.10640, 1.09159, 1.10002]
# alpha_0025 = [1.14624, 1.13429, 1.12391, 1.12385, 1.10600, 1.10980, 1.12140, 1.11137, 1.09042, 1.09979]
# alpha_003  = [1.15562, 1.12347, 1.11842, 1.11493, 1.12977, 1.13369, 1.12581, 1.12820, 1.12534, 1.13244]
# plt.plot(x, alpha_0015, color='red', label='Alpha = 0.015')
# plt.plot(x, alpha_002, color='black', label='Alpha = 0.02')
# plt.plot(x, alpha_0025, color='blue', label='Alpha = 0.025')
# plt.plot(x, alpha_003, color='green', label='Alpha = 0.03')
# plt.xlabel('Number of latent dimensions')
# plt.ylabel('Mean Square Error')
# plt.title('Iteration=4, lambda=0.005')
# plt.legend()
# plt.show()

# iteration=4, alpha=0.025, Weight Model
x = [1,2,3,4,5,6,7,8,9,10]
lambda_0002 = [1.13761, 1.14278, 1.11072, 1.12000, 1.11517, 1.10761, 1.11002, 1.10860, 1.11100, 1.09367]
lambda_0005 = [1.14624, 1.13429, 1.12391, 1.12385, 1.10600, 1.10980, 1.12140, 1.11137, 1.09042, 1.09979]
lambda_0008 = [1.14819, 1.13819, 1.11567, 1.10222, 1.11590, 1.11353, 1.10247, 1.09255, 1.10727, 1.09432]
plt.plot(x, lambda_0002, color='black', label='lambda = 0.002')
plt.plot(x, lambda_0005, color='blue', label='lambda = 0.005')
plt.plot(x, lambda_0008, color='green', label='lambda = 0.008')
plt.xlabel('Number of latent dimensions')
plt.ylabel('Mean Square Error')
plt.title('Iteration=4, alpha=0.025')
plt.legend()
plt.show()