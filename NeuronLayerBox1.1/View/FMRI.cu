#include "cuda_runtime.h"
#include <stdio.h>

__global__ static void out(struct axon *neuro, unsigned char *spike,int *output_image, struct OutPutLayer *outputlayer, int *output_timeN)
{
const int tid = threadIdx.x;
const int bid = blockIdx.x;
int number=bid *320 + tid;
int timeN=output_timeN[number];
int layernum;
int neuron_num;
int spike_strem;
if(number<outputlayer[0].all_length)
{
  for(int i=0;i<outputlayer[0].all_layer_num;i++)
  {
    if(number<=outputlayer[i].last_num)
    {
      layernum=i;
      if(i>0)
      {
        neuron_num=number-(outputlayer[i-1].last_num+1);
      }
      else
      {
        neuron_num=number;
      }
      break;
    }
  }
   spike_strem=output_image[number];
   if(timeN==0)
   {spike_strem=(spike_strem<<1)&0x03ff;}
   spike_strem=spike_strem|spike[neuron_num+outputlayer[layernum].addr]; //step=200us ,1ms store 1 spike
   output_image[number]=spike_strem;
   /*
  if(spike[neuron_num+outputlayer[layernum].addr]==1)
  {
    printf("number=%d\n",number);
  }*/

}
timeN=(timeN+1)%5;
output_timeN[number]=timeN;
}

__global__ static void fmri(struct axon *neuro, unsigned char *spike,int *output_image, struct OutPutLayer *outputlayer)
{
const int tid = threadIdx.x;
const int bid = blockIdx.x;
int number=bid *320 + tid;
float x;
float t=100;
int layernum;
int neuron_num;
int spike_strem;
if(number<outputlayer[0].all_length)
{
  for(int i=0;i<outputlayer[0].all_layer_num;i++)
  {
    if(number<=outputlayer[i].last_num)
    {
      layernum=i;
      if(i>0)
      {
        neuron_num=number-(outputlayer[i-1].last_num+1);
      }
      else
      {
        neuron_num=number;
      }
      break;
    }
  }
  x=output_image[number]/1000000.0;
  if(spike[neuron_num+outputlayer[layernum].addr]==1)
  {
    x=x+0.01;
  }
  x=x+tau*(-x/t);
  output_image[number]=(int)(x*1000000);
}

}
