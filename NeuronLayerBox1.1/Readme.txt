0.2:
增加了input的kernel计算框架
增加了输出所有层的框架
增加了多线程处理openGL与共享内存数据循环
修复了突触t1,t2错误, 修复了突触电导率数量级问题
统一了突触NMDA与AMPA方程
增加加了流的优先级

error:
Neuron/RS.cu(5): warning: attribute "global" does not apply here
Neuron/RS.cu(5): error: expected a ";"
struct没加";"

0.3:
分离出了view部件,单独kernel运行
增加了视网膜on cell
增加了页锁内存

0.5
重新定义了突触结构,增加了输入参数,突触计算式子按照SMART model式子计算
重新定义了神经元计算结构,分为三段计算,树突末端,树突近端,胞体计算
syanapse.txt与neuron.txt里的参数重新整合,按照SMART model参数计算
增加了轴突长度
去除了神经元代码中的中文乱码
找到了突触中number未加的错误

0.6
修改了突触模型的V2-V1顺序问题，之前电流方向反了。
修复量layer[5]与layer[6]赋值时顺序问题，layer[6]对应g_distal，layer[5]对应g_proximal。
更改了突触受体变化计算模型，原先的模型错误。
修复了post突触中中第二个计算单元最后number要加1
增加了cpu中的tau，原先的tau无法在cpu中读取
增加了学习率(未能实现)

error：
SM1-0.cu(468): warning: expression has no effect
SM1-0.cu(917): warning: expression has no effect
SM1-0.cu(919): warning: expression has no effect
i++或者j++放在后面

0.7
修改突触电压计算，全部用Izh形式，但神经元的几何模型仍然是grossberg的
神经元比较大小换回原来的if形式
找到量TC层输出有间断格子原因，原因在于第一颗虚拟计算单元注释问题

0.7.1
更改了元件前后是否有的bug
修改了可塑性突触的bug
增加了学习率(已实现)

0.7.2
无树突版本，直接给neuron
增加了学习率(已实现)
修改了可塑性突触的bug

0.7.3
在0.7.2基础上调整了neuron的输入，使之可以有随意的输入尺寸
修复了层内的xy坐标错误

1.0
在0.7.3基础上修改
增加了python接口
增加了多个层输出数据
增加了spike数据输出
增加了多个层输入数据
增加了q键退出功能
增加了保存加载功能

1.1
增加了图形化网络连接
对于neuron的txt文件格式修改，原先用行号的层标记改成了自定义层标记，写入到第一列
