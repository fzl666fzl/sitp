'''
TODO:
    2024.01.09
    此文件为50_2的验证，遗传算法+规则
'''
import random
import math
import matplotlib.pyplot as plt
import Fitness2GA
import fitness_dis
import pandas as pd
# 读取数据

init_pulse = 500
# 计算适应度，也就是距离分之一，这里用伪欧氏距离
def calcfit(gene,CT):
    fits = []
    fitness_dis.generate_episode(gene,CT,fits)
    # Fitness2GA.generate_episode(gene,CT,fits)
    print(fits)
    return 1 / fits[0],fits[1],fits[2]


# 每个个体的类，方便根据基因计算适应度
class Person:
    def __init__(self, gene,CT):
        self.gene = gene
        self.fit,self.myct,self.sii = calcfit(gene,CT)



class Group:
    def __init__(self):
        self.GroupSize = 30  # 种群规模
        self.GeneSize = 30  # 基因数量，也就是城市数量
        self.initGroup()
        self.upDate()
        self.CT = init_pulse

    # 初始化种群，随机生成若干个体
    def initGroup(self):
        self.group = []
        i = 0
        while (i < self.GroupSize):
            i += 1
            # gene如果在for以外生成只会shuffle一次
            gene = [random.uniform(0, 1) for i in range(self.GeneSize)]
            random.shuffle(gene)
            tmpPerson = Person(gene,init_pulse)
            self.group.append(tmpPerson)

    # 获取种群中适应度最高的个体
    def getBest(self):
        bestFit = self.group[0].fit
        best = self.group[0]
        for person in self.group:
            if (person.fit > bestFit):
                bestFit = person.fit
                best = person
        return best




    # 计算种群中所有个体的平均距离
    def getAvg(self):
        sum = 0
        for p in self.group:
            sum += 1 / p.fit
        return sum / len(self.group)

    # 根据适应度，使用轮盘赌返回一个个体，用于遗传交叉
    def getOne(self):
        # section的简称，区间
        sec = [0]
        sumsec = 0
        for person in self.group:
            sumsec += person.fit
            sec.append(sumsec)
        p = random.random() * sumsec
        for i in range(len(sec)):
            if (p > sec[i] and p < sec[i + 1]):
                # 这里注意区间是比个体多一个0的
                return self.group[i]

    # 更新种群相关信息
    def upDate(self):
        self.best = self.getBest()

#
#
# 遗传算法的类，定义了遗传、交叉、变异等操作
class GA:
    def __init__(self,pulses):
        self.group = Group()
        self.pCross = 0.35  # 交叉率
        self.pChange = 0.1  # 变异率
        self.Gen = 1  # 代数
        self.CT = init_pulse

    # 变异操作
    def change(self, gene):
        # 把列表随机的一段取出然后再随机插入某个位置
        # length是取出基因的长度，postake是取出的位置，posins是插入的位置
        geneLenght = len(gene)
        index1 = random.randint(0, geneLenght - 1)
        index2 = random.randint(0, geneLenght - 1)
        newGene = gene[:]  # 产生一个新的基因序列，以免变异的时候影响父种群
        newGene[index1], newGene[index2] = newGene[index2], newGene[index1]
        return newGene

    # 交叉操作
    def cross(self, p1, p2):
        geneLenght = len(p1.gene)
        index1 = random.randint(0, geneLenght - 1)
        index2 = random.randint(index1, geneLenght - 1)
        tempGene = p2.gene[index1:index2]  # 交叉的基因片段
        newGene = []
        p1len = 0
        for g in p1.gene:
            if p1len == index1:
                newGene.extend(tempGene)  # 插入基因片段
                p1len += 1
            if g not in tempGene:
                newGene.append(g)
                p1len += 1
        return newGene

    # 获取下一代
    def nextGen(self):
        self.Gen += 1
        # nextGen代表下一代的所有基因
        nextGen = []
        # 将最优秀的基因直接传递给下一代
        nextGen.append(self.group.getBest().gene[:])
        while (len(nextGen) < self.group.GroupSize):
            pChange = random.random()
            pCross = random.random()
            p1 = self.group.getOne()
            if (pCross < self.pCross):
                p2 = self.group.getOne()
                newGene = self.cross(p1, p2)
            else:
                newGene = p1.gene[:]
            if (pChange < self.pChange):
                newGene = self.change(newGene)
            nextGen.append(newGene)
        self.group.group = []
        CT = self.group.best.myct
        pulses.append(CT)
        for gene in nextGen:
            self.group.group.append(Person(gene,CT))
            self.group.upDate()

    # 打印当前种群的最优个体信息
    def showBest(self):
        print("第{}代\t当前最优{}\t当前平均{}\t".format(self.Gen, self.group.getBest().myct, self.group.getAvg()))

    # n代表代数，遗传算法的入口
    def run(self, n):
        Gen = []  # 代数
        dist = []  # 每一代的最优距离
        avgDist = []  # 每一代的平均距离
        # 上面三个列表是为了画图
        i = 1
        while (i < n):
            self.nextGen()
            self.showBest()
            i += 1
            Gen.append(i)
            dist.append(1 / self.group.getBest().fit)
            avgDist.append(self.group.getAvg())


pulses = []
ga = GA(pulses)
ga.run(200)
df_loss = pd.DataFrame(pulses)
# df_loss.to_excel('节拍变化表GA1130.xlsx')
print("进行3000代后最优解：", 1 / ga.group.getBest().fit)
'''
##最优动作01-445：
[0.08730194707980743, 0.008009234987428338, 0.6717533182795816], [0.7861525313849503, 0.3803996400349695, 0.4664505620274282]
[0.9842476491315433, 0.2915309754910067, 0.10638532098366227], [0.31054017852037263, 0.2499312130393091, 0.29456457960869076]
[0.20407700907615267, 0.615757769643056, 0.8831587289984958], [0.005364790353058924, 0.4537896070555164, 0.48630468248868153]
[0.010405262532213744, 0.6839063709470027, 0.8717575725564934], [0.5426639151645263, 0.5952932912818384, 0.4974718523209968]
[0.7481197777984777, 0.5579229314240739, 0.2747906341204508], [0.9451848611104011, 0.5778877100796752, 0.15781415325243364]
##02-692：
[0.011782911262146478, 0.37753912558169433, 0.8302676244557575], [0.9407195023588002, 0.829343448929692, 0.36521612317557994]
[0.8341710377137805, 0.14894796120294274, 0.3527639510515185], [0.8457460199558103, 0.44664770160699296, 0.480085920108412]
[0.6493472689918748, 0.49313601279393837, 0.4599288050860896], [0.6561328097924142, 0.22827468346604685, 0.906230421288839]
[0.12134813117487908, 0.6412944472303559, 0.4807104260005435], [0.3914709973048244, 0.42890447217862404, 0.26058958033999635]
[0.1435391619184604, 0.3658038048944545, 0.9456939520759186], [0.7783435323603485, 0.27809785518975505, 0.7238992742785236]

'''
