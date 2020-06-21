import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from scipy.special import gamma, comb
from random import shuffle


class ProbabilisticAnalyser:

    def __init__(self, u):
        self.obs = []
        self.count = 165
        self.heads = 0
        for i in range(165):
            x = 1 if (i <= int(u * 165)) else 0
            self.obs.append(x)
            self.heads += x
        shuffle(self.obs)
        print(self.obs)
        
    def beta_plotter(self, a, b, idx):
        prob, xs, u = [], [], 0
        norm_fac = (gamma(a+b)/(gamma(a)*gamma(b)))
        while u <= 1:
            pval = norm_fac * (u ** (a-1)) * ((1-u) ** (b-1))
            prob.append(pval)
            xs.append(u)
            u += 1e-5

        # plotting the graphs
        plt.figure(1)
        ax = plt.gca()
        ax.set_ylim(0, 15)
        plt.xlabel("u")
        plt.ylabel("prob_den")
        ax.plot(xs, prob)
        ax.xaxis.set_major_locator(mt.FixedLocator([i*0.1 for i in range(1, 1)]))
        plt.title("Gamma Distribution params a:{} b:{}.Samples:{}".format(a, b, idx + 1))
        plt.savefig("distribution_for_samples_{}".format(idx + 1) + str(".png"))
        plt.close(1)

    def sequential_bayesian(self):
        count = 0
        a, b = 2, 3
        for ob in self.obs:
            print("mean =", (a/(a+b)))
            # self.beta_plotter(a, b, count)
            if ob == 0:
                a, b = a, b+1
            if ob == 1:
                a, b = a+1, b
            count += 1
        print("final mean =", (a/(a+b)))
        # self.beta_plotter(a, b, count)

    def concurrent_bayesian(self):
        a, b = 2, 3
        m, N = self.heads, self.count
        a += m
        b += N-m
        print("final mean =", (a/(a+b)))
        print(comb(N, m) * (0.5 ** m) * (0.5 ** (N-m)))
        self.beta_plotter(a, b, 0)

analyser = ProbabilisticAnalyser(0.20)
analyser.sequential_bayesian()
analyser.concurrent_bayesian()

