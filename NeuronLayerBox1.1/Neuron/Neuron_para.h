#include "cuda_runtime.h"
#include <stdio.h>
//type:RS(1),FS(2),TC(3),TI(4),TRN(5),LTS(6),LS(7)


__constant__ float tau=0.2;    //Computation period(GPU)
float tau_cpu=0.2;  //Computation period(cpu) tau_cpu must equal tau

struct __align__(16) axon{
int neuron_numble;
int layer;
int x;
int y;
float v;
float u;
char type;
};

struct __align__(16) OutPutLayer
{
  int layer;
  int addr;
  int Xmax;
  int Ymax;
  int timeN;
  char type;
  int last_num;
  int all_layer_num;
  int all_length;
};

struct __align__(16) neuron_I
{
  int I;
};
