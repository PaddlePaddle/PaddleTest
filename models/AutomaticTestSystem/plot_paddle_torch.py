import matplotlib.pyplot as plt
import sys
import re
import math

def getdata(filename, delimiter1, delimiter2):
    data=[]
    with open(filename,'rt') as f:
       for line in f:
          pattern=re.compile(delimiter1+'(.+)'+delimiter2) 
#          pattern=re.compile('loss:(.+), ') 
          result=pattern.findall(line)
          if len(result)>0:
              # print(float(result[0]))
              data.append(float(result[0]))
    return data

def getdata_custom(filename, delimiter1, delimiter2):
    data=[]
    i=1
    with open(filename,'rt') as f:
       for line in f:
          pattern=re.compile(delimiter1+'(.+)'+delimiter2)
#          pattern=re.compile('loss:(.+), ')
          result=pattern.findall(line)
          if len(result)>0  and i%5==0:
             # print(float(result[0]))
              data.append(float(result[0]))
          i=i+1

    return data

def plot_paddle_loss(data, model):
    ydata=data[:30000]
    xdata=list(range(0,len(ydata)))
    # plot the data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xdata,ydata, label= model+'_train_loss', color='b',linewidth=2)

    # set the legend
    ax.legend()
    # set the limits
    ax.set_xlim([0, len(xdata)])
    ax.set_ylim([0, math.ceil(max(ydata))])

    ax.set_xlabel("iteration")
    ax.set_ylabel("train loss")
    ax.grid()
    ax.set_title(model)

    # display the plot
    plt.show()
    plt.savefig("train_loss.png")



def plot_paddle_torch_loss(data1, data2, model):
    ydata1=data1
    xdata1=list(range(0,len(ydata1)))
    ydata2=data2
    xdata2=list(range(0,len(ydata2)))
    ydata1=data1[:len(data2)]
    xdata1=list(range(0,len(ydata1)))

    # plot the data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(xdata1,ydata1, label= 'paddle_train_loss', color='b',linewidth=2)
    ax.plot(xdata2,ydata2, label= 'torch_train_loss', color='r',linestyle=':',linewidth=2)
    
    # set the legend
    ax.legend()
    # set the limits
    ax.set_xlim([0, len(xdata1)])
    ax.set_ylim([0, math.ceil(max(ydata1))])

    ax.set_xlabel("iteration")
    ax.set_ylabel("train loss")
    ax.grid()
    ax.set_title(model)

    # display the plot
    plt.show()
    plt.savefig("paddle_torch_train_loss.png")


if __name__ == '__main__':
    data1=getdata('rec_vitstr_none_ce.log', 'loss:', ', avg_reader_cost')
    data2=getdata('vitstr_torch.log', 'tensor\\(', ', device=')

    plot_paddle_torc_lossh(data1, data2, 'rec_vitstr_none_ce')
#    plot(data2, 'rec')

   
    
