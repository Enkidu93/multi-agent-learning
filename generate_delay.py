from numpy import random as random
from numpy import exp
import matplotlib.pyplot as plt
import random as r

class ExpdistDelayGenerator:
    def __init__(self,avg_val=20,seed=None):
        self.rand = r
        self.rand.seed(seed)
        random.seed(seed)
        self.dist = random.exponential(scale=avg_val,size=1_000)


    def generate_expdist_delay(self):
        return round(self.rand.choice(self.dist),2)

class WeibullDelayGenerator:
    def __init__(self,seed=None,m=10,d=100):
        self.rand = r
        self.rand.seed(seed)
        random.seed(seed)
        self.dist = m + d*random.weibull(a=0.75,size=1_000)


    def generate_weibulldist_delay(self):
        return round(self.rand.choice(self.dist),2)


# test = ExpdistDelayGenerator()
# # max = -1
# # min = 1000
# for _ in range(10):
#     print(test.generate_expdist_delay())
# #     curval = test.generate_expdist_delay()
# #     if curval < min:
# #         min = curval
# #     if curval > max:
# #         max = curval
# # print(min,max)

# test = WeibullDelayGenerator(m=10,d=100)
# x = []
# y = []
# sum = 0
# for _ in range(100):
#     x.append(test.generate_weibulldist_delay())
#     y.append(_)
#     sum += x[-1]
# print(sum/100)
# plt.plot(x,y)
# plt.show()


