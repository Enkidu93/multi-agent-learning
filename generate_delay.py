from numpy import random as random
import random as r

class ExpdistDelayGenerator:
    def __init__(self,avg_val=20,seed=None):
        self.rand = r
        self.rand.seed(seed)
        random.seed(seed)
        self.dist = random.exponential(scale=avg_val,size=1_000)


    def generate_expdist_delay(self):
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

