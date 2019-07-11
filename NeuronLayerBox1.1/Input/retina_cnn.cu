#include "cuda_runtime.h"
#include <stdio.h>

struct __align__(16) Retina1{
	 int input_number;
	 int kernelsize;
	 int postneuron_number;
};

//neurontype:retina
__global__ static void Retina_CNN(int *input,struct Retina1 *retina,int *THREAD_NUM,struct neuron_I *Ix,int *input_L)
{
const int tid = threadIdx.x;
const int bid = blockIdx.x;
int number=(bid * (*THREAD_NUM) + tid)*10;
float I;
int i=0;
int all_num=*input_L;
for(i=0;i<10;i++)
{
	if(number+i<all_num)
	{
		I=input[retina[number+i].input_number];
		Ix[retina[number+i].postneuron_number].I=(int)I*10;
		//if(I>0){printf("%d=%d\n",retina[number+i].postneuron_number,(int)I);}
	}
}
}

int Search_index(int *list, int x,int n)
{
    int i;
    for(i=0; i<n; i++)
        if(list[i] == x)
            return i;
    printf("search error,not find element\n");
    return -1;
}

int Distribution_input_func(int *input_type,int *Postlayer,int *kernelsize,int *X_max,int *Y_max,int maxline,int *Layer_index,int Nlines,int *layer_X_max,int *layer_Y_max,int *THREAD,int *BLOCK)
{
  int i=0;
  int error=0;
  int num=0;
  for(i=0;i<maxline;i++)
  {
    if(input_type[i]!=7)
    {
      printf("input neuron type error i=%d\n",i);
    }
    if(X_max[i]!=layer_X_max[Search_index(Layer_index,Postlayer[i],Nlines)] || Y_max[i]!=layer_Y_max[Search_index(Layer_index,Postlayer[i],Nlines)])
    {
        error=-1;printf("X_max=%d or Y_max=%d error i=%d\n",X_max[i],Y_max[i], i);break;
    }
    if(kernelsize[i]>=X_max[i] || kernelsize[i]>=X_max[i] || kernelsize[i]<=0)
    {
       error=-1;printf("kernelsize error i=%d\n", i);break;
    }
    num+=X_max[i]*Y_max[i];
  }
  printf("retina input number = %d\n",num);
  if(num==0)
  {
    *THREAD=0;
    *BLOCK=0;
    printf("no input\n");
    error=-1;
  }
  else if(num%1280==0)
  {
    *BLOCK=num/1280;
    *THREAD=128;
  }
  else if(num%960==0)
  {
    *BLOCK=num/960;
    *THREAD=96;
  }
  else if(num%640==0)
  {
    *BLOCK=num/640;
    *THREAD=64;
  }
  else if(num%320==0)
  {
    *BLOCK=num/320;
    *THREAD=32;
  }
  else
  {
		*BLOCK=ceil(num/320.0);
    *THREAD=32;
  }
  return error;
}

int connect_Retina_func(struct Retina1 *Retina_neuron,int *Postlayer,int *kernelsize,int *X_max,int *Y_max,int maxline,struct axon *neuron_copy)
{
 int i=0;
 int j=0;
 int N=0;
 int post_addr;
 int error=0;
 for(i=0;i<maxline;i++)
 {
	 j=0;
	 while(1)
	 {
		 if(neuron_copy[j].layer==Postlayer[i])            //寻找突触后神经元对应neuronbox首地址
		 {post_addr=j;j=0;break;}
		j++;
	 }
	 printf("i=%d\n",i);
	 printf("post_addr=%d\n",post_addr);
	 printf("Postlayer=%d\n",Postlayer[i]);
	 for(j=0;j<X_max[i]*Y_max[i];j++)
	 {
		 Retina_neuron[j+N].input_number=j;
		 Retina_neuron[j+N].kernelsize=kernelsize[i];
		 Retina_neuron[j+N].postneuron_number=post_addr+j;
		 if(neuron_copy[post_addr+j].layer!=Postlayer[i])
		 {
			 printf("error: i=%d\n",i);
			 printf("numbers=%d\n",j+N);
			 printf("post_addr=%d\n",post_addr+j);
			 printf("post_layer=%d\n",neuron_copy[post_addr+j].layer);
			 error=-1;break;
		 }
	 }
  N+=X_max[i]*Y_max[i];
 }
return error;
}
