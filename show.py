import matplotlib.pyplot as plt
import  numpy as np

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.2, 1.03*height, '%s' % float(height))



#变化影响因素，决策过程
Del_Difficulty=[0.439,0.505,0.585,0.613,0.613,0.634,0.641,0.69]
Del_DomainEntropy=[0.495,0.533,0.631,0.721,0.732,0.763,0.819,0.871]
Del_Sim=[0.484,0.568,0.617,0.655,0.645,0.683,0.718,0.791]
OREN=[0.53,0.655,0.732,0.774,0.784,0.84,0.913,0.934]
x=[1.94,3,4,5,6,7,8,9]
#折线图绘制
plt.plot(x, Del_Difficulty, marker='o',label='Del_Difficulty')
plt.plot(x, Del_DomainEntropy, marker='*',label='Del_DomainEntropy')
plt.plot(x, Del_Sim, marker='^',label='Del_Sim')
plt.plot(x, OREN,marker='s',label='OREN')
#
plt.ylabel("Accuracy")
#plt.title("Effect of Influent factors on Accuracy")
plt.xlim(2, 9)
plt.legend(loc='lower right')
plt.show()