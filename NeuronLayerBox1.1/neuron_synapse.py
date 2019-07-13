#encoding: utf-8
from __future__ import print_function, division
import numpy as np
input_kernel_size=5

neuron_file=open("load_data/draw_layer.txt",'r')
synapse_file=open("load_data/draw_synapse.txt",'r')

layer=[]
synapse=[]
type=0
line=neuron_file.readline()
while(line):
    if line=='':
        break
    line=line.strip().replace(' ','').split('\t')
    if len(line[0])>=6:
        if line[1]=='INPUT':
            type=7
        elif line[1]=='RS':
            type=1
        elif line[1]=='FS':
            type=2
        elif line[1]=='TC':
            type=3
        elif line[1]=='TI':
            type=4
        elif line[1]=='TRN':
            type=5
        elif line[1]=='LTS':
            type=6
        else:
            print("layer error\n")
        is_output=0
        if line[2]=="true":
            is_output=1
        layer.append([int(line[0].replace('Layer','')),type,int(line[3]),int(line[4]),is_output])
    line=neuron_file.readline()
neuron_file.close()

line=synapse_file.readline()
while(line):
    if line=='':
        break
    line=line.strip().replace(' ','').split('\t')
    if len(line[0])==10:
        if line[0][-2:]=='b2':
            type=1
        elif line[0][-2:]=='b1':
            type=2
        elif line[0][-2:]=='b3':
            type=3
        elif line[0][-2:]=='b4':
            type=4
        elif line[0][-2:]=='b5':
            type=5
        elif line[0][-2:]=='b6':
            type=6
        else:
            print("synapse error\n")

        prelyaer=int(line[1])
        postlyaer=int(line[2])
        inter=int(line[3])
        outer=int(line[4])
        connectpoint=int(line[5])
        g1=float(line[6])
        g2=float(line[7])
        t1_r=float(line[8])
        t1_f=float(line[9])
        t2_r=float(line[10])
        t2_f=float(line[11])
        Axon_delay=float(line[12])
        synapse.append([type,prelyaer,postlyaer,inter,outer,connectpoint,g1,g2,t1_r,t1_f,t2_r,t2_f,Axon_delay])
    line=synapse_file.readline()
synapse_file.close()

#input
input=[]
input_layer=[]
for Le in layer:
    if Le[1]==7:
            input.append([7,Le[0],input_kernel_size,Le[2],Le[3]])
            input_layer.append(Le[0])

#neuron
neuro_file=open("load_data/neuron.txt",'w')
for Le in layer:
    if Le[1]!=7:
        neuro_file.write(str(Le[0])+' '+str(Le[1])+' '+str(Le[2])+' '+str(Le[3])+'\n')
    else:
        neuro_file.write(str(Le[0])+' '+str(3)+' '+str(Le[2])+' '+str(Le[3])+'\n')
neuro_file.close()

#synapse
Syn=[]
synap_file=open("load_data/synapse.txt",'w')
for sy in synapse:
    synap_file.write(str(sy[0])+' '+str(sy[1])+' '+str(sy[2])+' '+str(sy[3])+' '+str(sy[4])+' '+str(sy[5])+' '+str(sy[6])+' '+str(sy[7])+' '+str(sy[8])+' '+str(sy[9])+' '+str(sy[10])+' '+str(sy[11])+' '+str(sy[12])+'\n')
synap_file.close()

input_file=open("load_data/input.txt",'w')
for inp in input:
    input_file.write(str(inp[0])+' '+str(inp[1])+' '+str(inp[2])+' '+str(inp[3])+' '+str(inp[4])+'\n')
input_file.close()

output_file=open("load_data/output.txt",'w')
for Le in layer:
    if Le[-1]:
        output_file.write(str(Le[0])+'\n')
output_file.close()
