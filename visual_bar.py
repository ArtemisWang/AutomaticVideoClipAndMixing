# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

name_list = ['0.2', '0.4', '0.6', '0.8']

num_list = [0.0161, 0.009, 0.0089, 0.0135] ## p1
num_list1 = [0.0037, 0.0065, 0.0054, 0.0065]
num_list2 = [0.0247, 0.0151, 0.0133, 0.0199] ## p2
num_list3 = [0.0025, 0.0033, 0.0033, 0.0033]
num_list4 = [0.0197, 0.011, 0.0101, 0.019] ## p3
num_list5 = [0.0022, 0.0026, 0.0025, 0.0026]
tick = ['p1']*4+['p2']*4+['p3']*4
x = list(range(len(num_list)))
z = list(range(len(num_list)))
total_width, n = 0.6, 3
width = total_width / n
bar_width = width - 0.02
# ax2 = plt.twiny()
plt.bar(x, num_list, width=bar_width, label='DFMR', fc='powderblue')
for xx,yy in zip(x, num_list):
    plt.text(xx, yy+0.0002, 'N=1', ha='center')
#'khaki' 'darkkhaki'
plt.bar(x, num_list1, width=bar_width, label='DFBM', fc='cadetblue')
for i in range(len(x)):
    x[i] = x[i] + width
z = z + x  # 'peachpuff' 'peru'
plt.bar(x, num_list2, width=bar_width, fc='powderblue')
plt.bar(x, num_list3, width=bar_width, fc='cadetblue')
plt.bar(x, num_list, width=0, tick_label=name_list)
for xx,yy in zip(x, num_list2):
    plt.text(xx, yy+0.0002, 'N=2', ha='center')
for i in range(len(x)):
    x[i] = x[i] + width
z = z + x
plt.bar(x, num_list4, width=bar_width, fc='powderblue')
plt.bar(x, num_list5, width=bar_width, fc='cadetblue')
for xx,yy in zip(x, num_list4):
    plt.text(xx, yy+0.0002, 'N=3', ha='center')
plt.xlabel('$\mu$')
plt.ylabel('BLEU')
# ax2.bar(z, num_list*3, width=0, tick_label=tick)
# ax2.set_xlabel('p')
# plt.bar(y, num_list, width=0, tick_labecpl=name_list)
plt.legend()
plt.savefig('bar_cp.pdf', dpi=600)
plt.show()


