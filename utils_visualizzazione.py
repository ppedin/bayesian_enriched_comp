import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def mostraGrafico(punti):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(punti)