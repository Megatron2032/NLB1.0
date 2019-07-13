#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <math.h>
#include <termios.h>

//type:RS(1),FS(2),TC(3),TI(4),TRN(5),LTS(6),LS(7)
#include "Neuron/Neuron_para.h"
#include "Neuron/RS.cu"
#include "Neuron/FS.cu"
#include "Neuron/TC.cu"
#include "Neuron/TI.cu"
#include "Neuron/TRN.cu"
#include "Neuron/LTS.cu"
//type:DualAND(1),Post(2),Excit(3),Inhi(4),ExcitHabituation(5),InhiHabituation(6)
#include "synapse/synapse_para.h"
#include "synapse/DualANDSynapse.cu"
#include "synapse/PostSynapse.cu"
#include "synapse/ExcitSynapse.cu"
#include "synapse/InhiSynapse.cu"
#include "synapse/ExcitHabituationSynapse.cu"
#include "synapse/InhiHabituationSynapse.cu"
#include "Input/retina_cnn.cu"
#include "View/FMRI.cu"
#include "connect.cu"

static struct termios initial_settings, new_settings;
static int peek_character = -1;

int kbhit();
int readch();
void init_keyboard();
void close_keyboard();
void time_init();
void time_1MS_fun(int signo);
int time1MS_flag=0;
int time100MS_flag=0;
int time1S_flag=0;
int global_run=1;

struct shared_use_st
{
    int written;//作为一个标志，非0：表示可读，0表示可写
    int _switch_;
    int timen;
    int Nlines;
    int input_num;
    int output_num;
    int step;
    int output_spike;   //1:spike  0:fmri
    int model;  //1:step  0:realtime
    int save;
    int restore;
    char restore_file[100];
};

struct shared_iamge
{
    int written_flag;//作为一个标志，非0：表示可读，1表示可写
    int read_flag;
    int length;
    int layer;
    int data[1280*720];//记录写入和读取的文本
    int width;
    int height;
    int next_frame;
};


#define neuron_BOX 6
#define synapse_BOX 6


#define checkCudaErrors(err)  __checkCudaErrors(err, __FILE__, __LINE__)
#define getLastCudaError(msg)  __getLastCudaError(msg, __FILE__, __LINE__)

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
    if(cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
        return ;
    }
}

// This will output the proper error string when calling cudaGetLastError
inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
        file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        return ;
    }
}


bool InitCUDA()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }
    printf("There is %d device.\n",deviceCount);
    int device;
    int device_ok=0;
    for (device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        if(cudaGetDeviceProperties(&deviceProp, device) == cudaSuccess)
        {
          printf("Device %d has compute capability %d.%d.\n",device, deviceProp.major, deviceProp.minor);
          if (!deviceProp.deviceOverlap)
  	     {
  		     printf("\nDevice %d will not handle overlaps.\n",device);
  	     }
         else
         {
           printf("Device %d will handle overlaps.\n",device);
         }
         printf("Device%d:\"%s\"\n", device, deviceProp.name);
         printf("Total amount of global memory                   %lu bytes\n", deviceProp.totalGlobalMem);
         printf("Number of mltiprocessors                        %d\n", deviceProp.multiProcessorCount);
         printf("Total amount of constant memory:                %lu bytes\n", deviceProp.totalConstMem);
         printf("Total amount of shared memory per block         %lu bytes\n", deviceProp.sharedMemPerBlock);
         printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
         printf("Warp size                                       %d\n", deviceProp.warpSize);
         printf("Maximum number of threada per block:            %d\n", deviceProp.maxThreadsPerBlock);
         printf("Maximum sizes of each dimension of a block:     %d x %d x %d\n", deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
         printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
         printf("Maximum memory pitch :                          %lu bytes\n", deviceProp.memPitch);
         printf("Texture alignmemt                               %lu bytes\n", deviceProp.texturePitchAlignment);
         printf("Clock rate                                      %.2f GHz\n", deviceProp.clockRate*1e-6f);
          if(deviceProp.major >= 6) {
                 device_ok++;
             }
        }

    }
    if(device_ok == 0) {
        fprintf(stderr, "There is no device supporting CUDA 6.5.\n");
        return false;
    }

    cudaSetDevice(0);
    return true;
}

//#define ln(x)  (x-1)-((x-1)*(x-1)/2.0)+((x-1)*(x-1)*(x-1)/3.0)-((x-1)*(x-1)*(x-1)*(x-1)/4.0)+((x-1)*(x-1)*(x-1)*(x-1)*(x-1)/5.0)
//#define e(x) 1+x+(x*x/2.0)+(x*x*x/6.0)+(x*x*x*x/24.0)+(x*x*x*x*x/120.0)

float pars[20][5]={{0.02,       0.2 ,    -65 ,     6   ,    14 },    // tonic spiking
  	               {0.02,      0.25,    -65  ,    6    ,   0.5 },    // phasic spiking
  	               {0.02,      0.2 ,    -50  ,    2    ,   15  },    // tonic bursting
  	               {0.02,      0.25,   -55   ,  0.05   ,  0.6  },    // phasic bursting
  	               {0.02,      0.2 ,    -55  ,   4    ,    10  },    // mixed mode
  	               {0.01,      0.2 ,    -65  ,   8    ,    30  },    // spike frequency adaptation
  	               {0.02,      -0.1,   -55   ,  6     ,    0   },    // Class 1
                   {0.2,       0.26,  -65    ,   0    ,    0   },    // Class 2
  	               {0.02,      0.2 ,   -65   ,  6     ,    7   },    // spike latency
  	               {0.05,      0.26,    -60  ,   0    ,    0   },    // subthreshold oscillations
  	               {0.1,       0.26,    -60  ,   -1   ,    0   },    // resonator
  	               {0.02,      -0.1,    -55  ,   6    ,    0   },    // integrator
  	               {0.03,      0.25,    -60  ,   4    ,    0   },    // rebound spike
  	               {0.03,      0.25,    -52  ,   0    ,    0   },    // rebound burst
  	               {0.03,      0.25,    -60  ,   4    ,    0   },    // threshold variability
  	               {1   ,      1.5 ,    -60  ,   0    ,  -65   },    // bistability
  	               {1 ,        0.2 ,    -60  ,   -21  ,    0   },    // DAP
  	               {0.02 ,     1   ,    -55  ,   4    ,    0   },    // accomodation
  	               {-0.02 ,     -1  ,    -60  ,   8    ,    80 },    // inhibition-induced spiking
                   {-0.026,     -1  ,    -45  ,   0    ,    80 }};    // inhibition-induced bursting




void Distribution_func(int box,int maxline,int *layer1,int *layer2,int *layer3,int *_THREAD_NUM,int *_BLOCK_NUM)
{
  int n=0;
  int i=0;
  for(i=0;i<maxline;i++)
  {
    if(layer1[i]==box){n+=layer2[i]*layer3[i];}   //RS_neuron   FS_neuron   TC_neuron   TI_neuron  TRN_neuron   LTS_neuron
  }
  if(n==0)
  {
    *_THREAD_NUM=0;
    *_BLOCK_NUM=0;
  }
  else if(n%1280==0)
  {
    *_BLOCK_NUM=n/1280;
    *_THREAD_NUM=128;
  }
  else if(n%960==0)
  {
    *_BLOCK_NUM=n/960;
    *_THREAD_NUM=96;
  }
  else if(n%640==0)
  {
    *_BLOCK_NUM=n/640;
    *_THREAD_NUM=64;
  }
  else if(n%320==0)
  {
    *_BLOCK_NUM=n/320;
    *_THREAD_NUM=32;
  }
  else
  {
    *_BLOCK_NUM=ceil(n/320.0);
    *_THREAD_NUM=32;
  }

}




int main(void)
{
  int c;
  int i,j,l;
  int n,m;
  printf("init Variable bytes\n");
  printf("int=%ld bytes\n",sizeof(int));
  printf("unsigned long int=%ld bytes\n",sizeof(unsigned long int));
  printf("float=%ld bytes\n",sizeof(float));
  printf("double=%ld bytes\n",sizeof(double));
  printf("char=%ld\n",sizeof(char));
  /*****************共享内存设置***************************/
  void *shm = NULL;
  struct shared_use_st *shared = NULL;
  int shmid;
  //创建共享内存
  shmid = shmget((key_t)2886, sizeof(struct shared_use_st), 0666|IPC_CREAT);
  if(shmid == -1)
  {
      fprintf(stderr, "shmget failed\n");
      exit(EXIT_FAILURE);
  }
  else
  {
    printf( "shared memory shmid=%d\n",shmid);
  }
  //将共享内存连接到当前进程的地址空间
  shm = shmat(shmid, (void*)0, 0);
  if(shm == (void*)-1)
  {
      fprintf(stderr, "shmat failed\n");
      exit(EXIT_FAILURE);
  }
  printf("Memory attached at %ld\n", (long)shm);
  //设置共享内存
  shared = (struct shared_use_st*)shm;
  shared->written=0;
  /*****************共享内存设置output***************************/
  FILE *fout = fopen("load_data/output.txt", "r");
  if(!fout)
  {
   printf("can't open output.txt\n");
   return -1;
  }
  int outputNlines=0;
  if(fout)
   {
       while((c=fgetc(fout)) != EOF)
           {if(c=='\n') outputNlines++;}
       fclose(fout);
   }
  printf("outputNlines=%d\n",outputNlines);
  int *output_layer=(int *)malloc(sizeof(int)*outputNlines);

  fout = fopen("load_data/output.txt", "r");
  i=0;
  while (!feof(fout))
  {
    fscanf(fout,"%d",&n);
    output_layer[i]=n;
    i++;
  }
  fclose(fout);

  shared->output_num=outputNlines;

  void *shm_image = NULL;//分配的共享内存的原始首地址
  struct shared_iamge *shared_image;//指向shm_image
  checkCudaErrors(cudaHostAlloc((void **)&shared_image,sizeof(struct shared_iamge)*outputNlines,cudaHostAllocDefault));
  int shmid_image;//共享内存标识符
  //创建共享内存
  shmid_image = shmget((key_t)1031, sizeof(struct shared_iamge)*outputNlines, 0666|IPC_CREAT);
  if(shmid_image == -1 )
  {
      fprintf(stderr, "shmget_image failed\n");
      exit(EXIT_FAILURE);
  }
  else
  {
    printf( "output shmid_image=%d\n",shmid_image);
  }
  //将共享内存连接到当前进程的地址空间
  shm_image = shmat(shmid_image, 0, 0);
  if(shm_image == (void*)-1)
  {
      fprintf(stderr, "shmat_image failed\n");
      exit(EXIT_FAILURE);
  }
  printf("\nimage Memory attached at %ld\n", (long)shm_image);
  //设置共享内存
  shared_image = (struct shared_iamge*)shm_image;
  for(i=0;i<outputNlines;i++)
  {
    shared_image[i].layer=output_layer[i];
    shared_image[i].written_flag = 0;
  }
  /******************************************************/
 float nf;
 int error=0;
 int sum_num=0;
 int *N=(int*)malloc(sizeof(int)*6);
 int Host_THREAD_NUM[6];
 int Host_BLOCK_NUM[6];
 int Host_THREAD_SYN[6];
 int Host_BLOCK_SYN[6];
 int *boxnum;
 int *THREAD_NUM;
 int *BLOCK_NUM;
 int *THREAD_SYN;
 int *BLOCK_SYN;
 int THREAD_Input_NUM;
 int Block_Input_NUM;
 int *THREAD_Input_device_NUM;
 int input_length;

/***********input data********************/
 FILE *fin = fopen("load_data/input.txt", "r");
 if(!fin)
 {
  printf("can't open input.txt\n");
  return -1;
 }
 int inputNlines=0;
 if(fin)
  {
      while((c=fgetc(fin)) != EOF)
          {if(c=='\n') inputNlines++;}
      fclose(fin);
  }
 printf("inputNlines=%d\n",inputNlines);
 shared->input_num=inputNlines;
 int *inputN1=(int*)malloc(sizeof(int)*(inputNlines));
 int *inputN2=(int*)malloc(sizeof(int)*(inputNlines));
 int *inputN3=(int*)malloc(sizeof(int)*(inputNlines));
 int *inputN4=(int*)malloc(sizeof(int)*(inputNlines));
 int *inputN5=(int*)malloc(sizeof(int)*(inputNlines));
 i=0;
 fin = fopen("load_data/input.txt", "r");
 while (!feof(fin))
 {
 fscanf(fin,"%d",&n);
 inputN1[i]=n;
 fscanf(fin,"%d",&n);
 inputN2[i]=n;
 fscanf(fin,"%d",&n);
 inputN3[i]=n;
 fscanf(fin,"%d",&n);
 inputN4[i]=n;
 fscanf(fin,"%d",&n);
 inputN5[i]=n;
 if(i<inputNlines)
 {
   printf("inputN1[%d]=%d  inputN2[%d]=%d  inputN3[%d]=%d  inputN4[%d]=%d  inputN5[%d]=%d\n",i,inputN1[i],i,inputN2[i],i,inputN3[i],i,inputN4[i],i,inputN5[i]);
 }
 i++;
 //读出的数在n里，一次一个数
 }
 fclose(fin); //读完就退出循环
 /*****************共享内存设置input***************************/
 struct shared_iamge *shared_input_image;//指向shm_input_image
 void *shm_input_image;
 checkCudaErrors(cudaHostAlloc((void **)&(shared_input_image),sizeof(struct shared_iamge)*inputNlines,cudaHostAllocDefault));
 input_length=0;


 int shmid_input_image;//共享内存标识符
 //创建共享内存
 shmid_input_image = shmget((key_t)2887, sizeof(struct shared_iamge)*inputNlines, 0666|IPC_CREAT);
 if(shmid_input_image == -1 )
 {
     fprintf(stderr, "shmget_input_image failed\n");
     exit(EXIT_FAILURE);
 }
 else
 {
   printf( "shmid_input_image=%d\n",shmid_input_image);
 }
 //将共享内存连接到当前进程的地址空间
 shm_input_image = shmat(shmid_input_image, 0, 0);
 if(shm_input_image == (void*)-1)
 {
     fprintf(stderr, "shmat_input_image failed\n");
     exit(EXIT_FAILURE);
 }
 printf("\nimage Memory attached at %ld\n", (long)shm_input_image);
 //设置共享内存
 shared_input_image=(struct shared_iamge *)shm_input_image;

 for(i=0;i<inputNlines;i++)
 {
 int Xmax=inputN4[i];   //输入最大
 int Ymax=inputN5[i];   //输入最小
 shared_input_image[i].width=Xmax;
 shared_input_image[i].height=Ymax;
 shared_input_image[i].length=Xmax*Ymax;
 input_length+=shared_input_image[i].length;
 printf("shared_input_image[%d]->length=%d\n",i,shared_input_image[i].length);
 shared_input_image[i].written_flag = 0;
 }

/*
 int shmid_input_data[inputNlines];//共享内存标识符
 void *shm_input_data[inputNlines];//分配的共享内存的原始首地址
 int *input_data;
 input_length=0;
 for(i=0;i<inputNlines;i++)
 {
   int Xmax=inputN4[i];   //输入最大
   int Ymax=inputN5[i];   //输入最小
   shared_input_image[i]->width=Xmax;
   shared_input_image[i]->height=Ymax;
   shared_input_image[i]->length=Xmax*Ymax;
   input_length+=shared_input_image[i]->length;
   input_data=(int *)malloc(sizeof(int)*shared_input_image[i]->length);
   //checkCudaErrors(cudaHostAlloc((void **)&(shared_input_image[i]->data),sizeof(int)*Xmax*Ymax,cudaHostAllocDefault));

   //创建共享内存
   shmid_input_data[i] = shmget((key_t)(2888+inputNlines+i), sizeof(int)*Xmax*Ymax, 0666|IPC_CREAT);
   if(shmid_input_data[i] == -1 )
   {
       fprintf(stderr, "shmget_input_image failed\n");
       exit(EXIT_FAILURE);
   }
   else
   {
     printf( "shmid_input_data[%d]=%d\n",i,shmid_input_data[i]);
   }
   //将共享内存连接到当前进程的地址空间
   shm_input_data[i] = shmat(shmid_input_data[i], 0, 0);
   if(shm_input_data[i] == (void*)-1)
   {
       fprintf(stderr, "shmat_input_image failed\n");
       exit(EXIT_FAILURE);
   }
   printf("\nimage Memory attached at %ld\n", (long)shm_input_image);
   input_data=(int *)shm_input_data[i];
   shared_input_image[i]->data=input_data;
 }*/
/**********************************************/

/*********************neuron data*****************/
 FILE *fp = fopen("load_data/neuron.txt", "r");
 if(!fp)
 {
  printf("can't open neuron.txt\n");
  return -1;
 }
 int Nlines=0;
 if(fp)
  {
      while((c=fgetc(fp)) != EOF)
          {if(c=='\n') Nlines++;}
      fclose(fp);
  }
 printf("Nlines=%d\n",Nlines);
 int *Layer_index=(int*)malloc(sizeof(int)*(Nlines));
 int *Layer1=(int*)malloc(sizeof(int)*(Nlines));
 int *Layer2=(int*)malloc(sizeof(int)*(Nlines));
 int *Layer3=(int*)malloc(sizeof(int)*(Nlines));
 fp = fopen("load_data/neuron.txt", "r");
 if(!fp)
 {
  printf("can't open neuron.txt\n");
  return -1;
 }
 i=0;
 while (!feof(fp))
 {
 fscanf(fp,"%d",&n);
 Layer_index[i]=n;
 fscanf(fp,"%d",&n);
 Layer1[i]=n;
 fscanf(fp,"%d",&n);
 Layer2[i]=n;
 fscanf(fp,"%d",&n);
 Layer3[i]=n;
 if(i<Nlines)
 {
   printf("Layer1[%d]=%d  Layer2[%d]=%d  Layer3[%d]=%d\n",i,Layer1[i],i,Layer2[i],i,Layer3[i]);
 }
 i++;
 //读出的数在n里，一次一个数
 }
 printf("1\n");
 //fclose(fp);
 printf("2\n");
/***************************************************/

/**************************synapse data***************************/
int Sy1=0,Sy2=0,Sy3=0,Sy4=0,Sy5=0,Sy6=0;
printf("load synapse data start\n");
 FILE *fd = fopen("load_data/synapse.txt", "r");
 if(!fd)
 {
  printf("can't open synapse.txt\n");
  return -1;
 }
 int Slines=0;
 if(fd)
  {
      while((c=fgetc(fd)) != EOF)
          {if(c=='\n') Slines++;}
      fclose(fd);
  }
 printf("Synapse lines=%d\n",Slines);

 struct syn_loaddata *syn_load=(struct syn_loaddata*)malloc(sizeof(struct syn_loaddata));

 syn_load->type=(int*)malloc(sizeof(int)*(Slines));
 syn_load->Prelayer=(int*)malloc(sizeof(int)*(Slines));
 syn_load->Postlayer=(int*)malloc(sizeof(int)*(Slines));
 syn_load->inter=(int*)malloc(sizeof(int)*(Slines));
 syn_load->outer=(int*)malloc(sizeof(int)*(Slines));
 syn_load->connect_point=(int*)malloc(sizeof(int)*(Slines));
 syn_load->g1=(float*)malloc(sizeof(float)*(Slines));
 syn_load->g2=(float*)malloc(sizeof(float)*(Slines));
 syn_load->t1r=(float*)malloc(sizeof(float)*(Slines));
 syn_load->t1f=(float*)malloc(sizeof(float)*(Slines));
 syn_load->t2r=(float*)malloc(sizeof(float)*(Slines));
 syn_load->t2f=(float*)malloc(sizeof(float)*(Slines));
 syn_load->axon_delay=(float*)malloc(sizeof(float)*(Slines));

 i=0;
 fd = fopen("load_data/synapse.txt", "r");
 while (!feof(fd))
 {
  fscanf(fd, "%d", &m);
  syn_load->type[i]=m;
  fscanf(fd, "%d", &m);
  syn_load->Prelayer[i]=m;
  fscanf(fd, "%d", &m);
  syn_load->Postlayer[i]=m;
  fscanf(fd, "%d", &m);
  syn_load->inter[i]=m;
  fscanf(fd, "%d", &m);
  syn_load->outer[i]=m;
  fscanf(fd, "%d", &m);
  syn_load->connect_point[i]=m;
  fscanf(fd, "%f", &nf);
  syn_load->g1[i]=nf;
  fscanf(fd, "%f", &nf);
  syn_load->g2[i]=nf;
  fscanf(fd, "%f", &nf);
  syn_load->t1r[i]=nf;
  fscanf(fd, "%f", &nf);
  syn_load->t1f[i]=nf;
  fscanf(fd, "%f", &nf);
  syn_load->t2r[i]=nf;
  fscanf(fd, "%f", &nf);
  syn_load->t2f[i]=nf;
  fscanf(fd, "%f", &nf);
  syn_load->axon_delay[i]=nf;
  if(i<Slines)
  {
  printf("type[%d]=%d  ",i,syn_load->type[i]);
  printf("Prelayer[%d]=%d  ",i,syn_load->Prelayer[i]);
  printf("Postlayer[%d]=%d  ",i,syn_load->Postlayer[i]);
  printf("inter[%d]=%d  ",i,syn_load->inter[i]);
  printf("outer[%d]=%d  ",i,syn_load->outer[i]);
  printf("connect_point[%d]=%d  ",i,syn_load->connect_point[i]);
  printf("g1[%d]=%f  ",i,syn_load->g1[i]);
  printf("g2[%d]=%f  ",i,syn_load->g2[i]);
  printf("t1r[%d]=%f  ",i,syn_load->t1r[i]);
  printf("t1f[%d]=%f  ",i,syn_load->t1f[i]);
  printf("t2r[%d]=%f  ",i,syn_load->t2r[i]);
  printf("t2f[%d]=%f  ",i,syn_load->t2f[i]);
  printf("axon_delay[%d]=%f\n",i,syn_load->axon_delay[i]);
  }
 i++;
 //读出的数在n里，一次一个数
 }
 fclose(fd); //读完就退出循环
 for(i=0;i<Slines;i++)
 {
   if(syn_load->type[i]==1){Sy1++;}    //Dual_AND_Synapse
   if(syn_load->type[i]==2){Sy2++;}    //post_Synapse
   if(syn_load->type[i]==3){Sy3++;}    //excit_Synapse
   if(syn_load->type[i]==4){Sy4++;}    //inhi_Synapse
   if(syn_load->type[i]==5){Sy5++;}    //excit_habituation_Synapse
   if(syn_load->type[i]==6){Sy6++;}    //inhi_habituation_Synapse
 }
 printf("load synapse data end\n");
/***************************************************/

/******************neuron Distribution*******************************/
 printf("start Distribution neuron\n");
 for(i=0;i<neuron_BOX;i++)
 {
  Distribution_func(i+1,Nlines,Layer1,Layer2,Layer3,&Host_THREAD_NUM[i],&Host_BLOCK_NUM[i]);
  printf("THREAD_NUM[%d]=%d;BLOCK_NUM[%d]=%d;\n",i,Host_THREAD_NUM[i],i,Host_BLOCK_NUM[i]);
 }
 int all_neuron_number=(Host_THREAD_NUM[0]*Host_BLOCK_NUM[0]+Host_THREAD_NUM[1]*Host_BLOCK_NUM[1]+Host_THREAD_NUM[2]*Host_BLOCK_NUM[2]+
 Host_THREAD_NUM[3]*Host_BLOCK_NUM[3]+Host_THREAD_NUM[4]*Host_BLOCK_NUM[4]+Host_THREAD_NUM[5]*Host_BLOCK_NUM[5])*10;
 printf("all_neuron_number=%d\n",all_neuron_number);

  printf("neuron cpu malloc start\n");
  unsigned char Signal[all_neuron_number];
  struct axon *N_para=(struct axon*)malloc(sizeof(struct axon)*all_neuron_number);
  int *image=(int*)malloc(sizeof(int)*20*20);
	printf("neuron cpu malloc end\n");


  printf("Neuron Parameter Loading ...\n");
  N[0]=Host_THREAD_NUM[0]*Host_BLOCK_NUM[0]*10;
  N[1]=N[0]+Host_THREAD_NUM[1]*Host_BLOCK_NUM[1]*10;
  N[2]=N[1]+Host_THREAD_NUM[2]*Host_BLOCK_NUM[2]*10;
  N[3]=N[2]+Host_THREAD_NUM[3]*Host_BLOCK_NUM[3]*10;
  N[4]=N[3]+Host_THREAD_NUM[4]*Host_BLOCK_NUM[4]*10;
  N[5]=N[4]+Host_THREAD_NUM[5]*Host_BLOCK_NUM[5]*10;
  printf("N1=%d\n",N[0]);
  printf("N2=%d\n",N[1]);
  printf("N3=%d\n",N[2]);
  printf("N4=%d\n",N[3]);
  printf("N5=%d\n",N[4]);
  printf("N6=%d\n",N[5]);

int num_flag=0;
for(l=0;l<6;l++)
{
  for(i=0;i<Nlines;i++)
  {
    if(Layer1[i]==(l+1))
    {
      for(j=num_flag;j<num_flag+Layer2[i]*Layer3[i];j++)
      {
       N_para[j].neuron_numble=j;
       N_para[j].u=-20;
       N_para[j].v=-60;
       Signal[j]=0;
       N_para[j].type=l+1;
       N_para[j].layer=Layer_index[i];  //Layer_index=i
       N_para[j].x=(j-num_flag+1)/Layer3[i];
       N_para[j].y=(j-num_flag+1)%Layer3[i]-1;
      }
      num_flag=j;
    }
  }
  printf("num_flag=%d\n",num_flag);
 if(num_flag>N[l])
 {printf("N%d error\n",l);return -1;}
 N[l]=num_flag-1;
}


printf("Neuron Parameter Loading Complete\n");
/*************************************************/

printf("start Distribution output neuron layer addr\n");
if(remove("load_data/NeuronLayerAddr.txt")==0)
{
  printf("load_data/NeuronLayerAddr.txt\n");
}
fout = fopen("load_data/NeuronLayerAddr.txt", "wb");
if(!fout)
{
 printf("can't open load_data/NeuronLayerAddr.txt\n");
 return -1;
}
int layer_addr;
j=0;
for(i=0;i<Nlines;i++)
{
  while(1)
  {
    if(N_para[j].layer==Layer_index[i])  //Layer_index=i           //寻找突触前神经元对应neuronbox首地址
    {layer_addr=j;j=0;break;}
    j++;
  }
  fprintf(fout, "%d %d",Layer_index[i], layer_addr);
  fprintf(fout, "\n");
  //fwrite(&layer_addr,sizeof(int),1, fout);
  //fwrite("\n",sizeof(int),1, fout);
  printf("layer%d=%d\n",Layer_index[i],layer_addr);
}
fclose(fout);
/*********************************************************/

shared->Nlines=Nlines;
shared->written=1;
printf("wait open switch\n");
while(shared->_switch_!=1);
printf("switch ON\n");

struct synapse *Plastic_synap1;
struct synapse *Plastic_synap2;
struct synapse *neuro_synap3;
struct synapse *neuro_synap4;
struct synapse *neuro_synap5;
struct synapse *neuro_synap6;
int *N_SYN=(int*)malloc(sizeof(int)*6);

int c_flag;
if(shared->restore)
{
  struct synapse *SYN[6];
  printf("restore Synapse CUDA kernel start\n");

  c_flag=read_synapse_func(shared->restore_file,Host_THREAD_SYN,Host_BLOCK_SYN,SYN,N_SYN);
  Plastic_synap1=SYN[0];
  Plastic_synap2=SYN[1];
  neuro_synap3=SYN[2];
  neuro_synap4=SYN[3];
  neuro_synap5=SYN[4];
  neuro_synap6=SYN[5];

  if(c_flag<0)
  {
    global_run=0;
  }
  printf("restore Synapse CUDA kernel end\n");
}
else
{
  printf("Synapse CUDA analysis Loading ...\n");
  FILE *fp_data=fopen("load_data/data.txt","w");
  fclose(fp_data);

  for(i=0;i<synapse_BOX;i++)
  {
   c_flag=analysis_synapse_func(syn_load,Layer_index,Nlines,Layer2,Layer3,Slines,i+1,&Host_THREAD_SYN[i],&Host_BLOCK_SYN[i]);
   printf("THREAD_SYN[%d]=%d;BLOCK_SYN[%d]=%d;\n",i,Host_THREAD_SYN[i],i,Host_BLOCK_SYN[i]);
   if(c_flag<0)
   {return -1;}
  }
  printf("Synapse CUDA analysis Complete\n");

printf("synapse malloc start\n");
Plastic_synap1=(struct synapse*)malloc(sizeof(struct synapse)*(Host_THREAD_SYN[0]*Host_BLOCK_SYN[0]*10));
Plastic_synap2=(struct synapse*)malloc(sizeof(struct synapse)*(Host_THREAD_SYN[1]*Host_BLOCK_SYN[1]*10));
neuro_synap3=(struct synapse*)malloc(sizeof(struct synapse)*(Host_THREAD_SYN[2]*Host_BLOCK_SYN[2]*10));
neuro_synap4=(struct synapse*)malloc(sizeof(struct synapse)*(Host_THREAD_SYN[3]*Host_BLOCK_SYN[3]*10));
neuro_synap5=(struct synapse*)malloc(sizeof(struct synapse)*(Host_THREAD_SYN[4]*Host_BLOCK_SYN[4]*10));
neuro_synap6=(struct synapse*)malloc(sizeof(struct synapse)*(Host_THREAD_SYN[5]*Host_BLOCK_SYN[5]*10));
printf("synapse malloc success\n");

printf("Synapse Connect Loading ...\n");
printf("synapse1\n");
c_flag=connect_synapse_func(syn_load,Layer_index,Nlines,Layer2,Layer3,1,&Host_THREAD_SYN[0],&Host_BLOCK_SYN[0],Plastic_synap1,N_para,N_SYN);
if(c_flag<0)
{printf("error1\n");return -1;}
printf("synapse2\n");
c_flag=connect_synapse_func(syn_load,Layer_index,Nlines,Layer2,Layer3,2,&Host_THREAD_SYN[1],&Host_BLOCK_SYN[1],Plastic_synap2,N_para,N_SYN);
if(c_flag<0)
{printf("error2\n");return -1;}
printf("synapse3\n");
c_flag=connect_synapse_func(syn_load,Layer_index,Nlines,Layer2,Layer3,3,&Host_THREAD_SYN[2],&Host_BLOCK_SYN[2],neuro_synap3,N_para,N_SYN);
if(c_flag<0)
{printf("error3\n");return -1;}
printf("synapse4\n");
c_flag=connect_synapse_func(syn_load,Layer_index,Nlines,Layer2,Layer3,4,&Host_THREAD_SYN[3],&Host_BLOCK_SYN[3],neuro_synap4,N_para,N_SYN);
if(c_flag<0)
{printf("error4\n");return -1;}
printf("synapse5\n");
c_flag=connect_synapse_func(syn_load,Layer_index,Nlines,Layer2,Layer3,5,&Host_THREAD_SYN[4],&Host_BLOCK_SYN[4],neuro_synap5,N_para,N_SYN);
if(c_flag<0)
{printf("error5\n");return -1;}
printf("synapse6\n");
c_flag=connect_synapse_func(syn_load,Layer_index,Nlines,Layer2,Layer3,6,&Host_THREAD_SYN[5],&Host_BLOCK_SYN[5],neuro_synap6,N_para,N_SYN);
if(c_flag<0)
{printf("error6\n");return -1;}
printf("Synapse Connect Loading Complete\n");

//printf("Synapse Parameter Loading ...\n");
/*
for(j=0;j<Host_THREAD_SYN[0]*Host_BLOCK_SYN[0]*10;j++)
{
 Plastic_synap1[j].w=w_init;
 Plastic_synap1[j].timeN=0;
 Plastic_synap1[j].connect_point=syn_load->connect_point[Plastic_synap1[j].synaplayer];
 Plastic_synap1[j].g1=syn_load->g1[Plastic_synap1[j].synaplayer];
 Plastic_synap1[j].g2=syn_load->g2[Plastic_synap1[j].synaplayer];
 Plastic_synap1[j].s1=0;
 Plastic_synap1[j].R1=0;
 Plastic_synap1[j].s2=0;
 Plastic_synap1[j].R2=0;
 Plastic_synap1[j].t1_r=syn_load->t1r[Plastic_synap1[j].synaplayer];
 Plastic_synap1[j].t1_f=syn_load->t1f[Plastic_synap1[j].synaplayer];
 Plastic_synap1[j].t2_r=syn_load->t2r[Plastic_synap1[j].synaplayer];
 Plastic_synap1[j].t2_f=syn_load->t2f[Plastic_synap1[j].synaplayer];
 Plastic_synap1[j].axon_delay=floor(syn_load->axon_delay[Plastic_synap1[j].synaplayer]/tau_cpu);
}
for(j=0;j<Host_THREAD_SYN[1]*Host_BLOCK_SYN[1]*10;j++)
{
 Plastic_synap2[j].w=w_init;
 Plastic_synap2[j].timeN=0;
 Plastic_synap2[j].connect_point=syn_load->connect_point[Plastic_synap2[j].synaplayer];
 Plastic_synap2[j].g1=syn_load->g1[Plastic_synap2[j].synaplayer];
 Plastic_synap2[j].g2=syn_load->g2[Plastic_synap2[j].synaplayer];
 Plastic_synap2[j].s1=0;
 Plastic_synap2[j].R1=0;
 Plastic_synap2[j].s2=0;
 Plastic_synap2[j].R2=0;
 Plastic_synap2[j].t1_r=syn_load->t1r[Plastic_synap2[j].synaplayer];
 Plastic_synap2[j].t1_f=syn_load->t1f[Plastic_synap2[j].synaplayer];
 Plastic_synap2[j].t2_r=syn_load->t2r[Plastic_synap2[j].synaplayer];
 Plastic_synap2[j].t2_f=syn_load->t2f[Plastic_synap2[j].synaplayer];
 Plastic_synap2[j].axon_delay=floor(syn_load->axon_delay[Plastic_synap2[j].synaplayer]/tau_cpu);
}
for(j=0;j<Host_THREAD_SYN[2]*Host_BLOCK_SYN[2]*10;j++)
{
 neuro_synap3[j].z=z_init;
 neuro_synap3[j].timeN=0;
 neuro_synap3[j].connect_point=syn_load->connect_point[neuro_synap3[j].synaplayer];
 neuro_synap3[j].g1=syn_load->g1[neuro_synap3[j].synaplayer];
 neuro_synap3[j].g2=syn_load->g2[neuro_synap3[j].synaplayer];
 neuro_synap3[j].s1=0;
 neuro_synap3[j].R1=0;
 neuro_synap3[j].s2=0;
 neuro_synap3[j].R2=0;
 neuro_synap3[j].t1_r=syn_load->t1r[neuro_synap3[j].synaplayer];
 neuro_synap3[j].t1_f=syn_load->t1f[neuro_synap3[j].synaplayer];
 neuro_synap3[j].t2_r=syn_load->t2r[neuro_synap3[j].synaplayer];
 neuro_synap3[j].t2_f=syn_load->t2f[neuro_synap3[j].synaplayer];
 neuro_synap3[j].axon_delay=floor(syn_load->axon_delay[neuro_synap3[j].synaplayer]/tau_cpu);
}
for(j=0;j<Host_THREAD_SYN[3]*Host_BLOCK_SYN[3]*10;j++)
{
 neuro_synap4[j].z=z_init;
 neuro_synap4[j].timeN=0;
 neuro_synap4[j].connect_point=syn_load->connect_point[neuro_synap4[j].synaplayer];
 neuro_synap4[j].g1=syn_load->g1[neuro_synap4[j].synaplayer];
 neuro_synap4[j].g2=syn_load->g2[neuro_synap4[j].synaplayer];
 neuro_synap4[j].s1=0;
 neuro_synap4[j].R1=0;
 neuro_synap4[j].s2=0;
 neuro_synap4[j].R2=0;
 neuro_synap4[j].t1_r=syn_load->t1r[neuro_synap4[j].synaplayer];
 neuro_synap4[j].t1_f=syn_load->t1f[neuro_synap4[j].synaplayer];
 neuro_synap4[j].t2_r=syn_load->t2r[neuro_synap4[j].synaplayer];
 neuro_synap4[j].t2_f=syn_load->t2f[neuro_synap4[j].synaplayer];
 neuro_synap4[j].axon_delay=floor(syn_load->axon_delay[neuro_synap4[j].synaplayer]/tau_cpu);
}
for(j=0;j<Host_THREAD_SYN[4]*Host_BLOCK_SYN[4]*10;j++)
{
 neuro_synap5[j].z=z_init;
 neuro_synap5[j].timeN=0;
 neuro_synap5[j].connect_point=syn_load->connect_point[neuro_synap5[j].synaplayer];
 neuro_synap5[j].g1=syn_load->g1[neuro_synap5[j].synaplayer];
 neuro_synap5[j].g2=syn_load->g2[neuro_synap5[j].synaplayer];
 neuro_synap5[j].s1=0;
 neuro_synap5[j].R1=0;
 neuro_synap5[j].s2=0;
 neuro_synap5[j].R2=0;
 neuro_synap5[j].t1_r=syn_load->t1r[neuro_synap5[j].synaplayer];
 neuro_synap5[j].t1_f=syn_load->t1f[neuro_synap5[j].synaplayer];
 neuro_synap5[j].t2_r=syn_load->t2r[neuro_synap5[j].synaplayer];
 neuro_synap5[j].t2_f=syn_load->t2f[neuro_synap5[j].synaplayer];
 neuro_synap5[j].axon_delay=floor(syn_load->axon_delay[neuro_synap5[j].synaplayer]/tau_cpu);
}
for(j=0;j<Host_THREAD_SYN[5]*Host_BLOCK_SYN[5]*10;j++)
{
 neuro_synap6[j].z=z_init;
 neuro_synap6[j].timeN=0;
 neuro_synap6[j].connect_point=syn_load->connect_point[neuro_synap6[j].synaplayer];
 neuro_synap6[j].g1=syn_load->g1[neuro_synap6[j].synaplayer];
 neuro_synap6[j].g2=syn_load->g2[neuro_synap6[j].synaplayer];
 neuro_synap6[j].s1=0;
 neuro_synap6[j].R1=0;
 neuro_synap6[j].s2=0;
 neuro_synap6[j].R2=0;
 neuro_synap6[j].t1_r=syn_load->t1r[neuro_synap6[j].synaplayer];
 neuro_synap6[j].t1_f=syn_load->t1f[neuro_synap6[j].synaplayer];
 neuro_synap6[j].t2_r=syn_load->t2r[neuro_synap6[j].synaplayer];
 neuro_synap6[j].t2_f=syn_load->t2f[neuro_synap6[j].synaplayer];
 neuro_synap6[j].axon_delay=floor(syn_load->axon_delay[neuro_synap6[j].synaplayer]/tau_cpu);
}*/
//printf("Synapse Parameter Loading Complete\n");
}

/******************************input****************************************/
printf("start Distribution input neuron\n");
error=Distribution_input_func(inputN1,inputN2,inputN3,inputN4,inputN5,inputNlines,Layer_index,Nlines,Layer2,Layer3,&THREAD_Input_NUM,&Block_Input_NUM);
if(error!=0){printf("Distribution_input_func error\n");return -1;}
printf("THREAD_Input_NUM=%d,Block_Input_NUM=%d\n",THREAD_Input_NUM,Block_Input_NUM);
struct Retina1 *Retina_neuron=(struct Retina1*)malloc(sizeof(struct Retina1)*THREAD_Input_NUM*Block_Input_NUM*10);
error=connect_Retina_func(Retina_neuron,inputN2,inputN3,inputN4,inputN5,inputNlines,N_para);
if(error!=0){printf("connect_Retina_func error\n");return -1;}

printf("init over\n");


/***********************outputlayer***************************/
struct OutPutLayer *out_information=(struct OutPutLayer*)malloc(sizeof(struct OutPutLayer)*outputNlines);
int out_length=0;
for(i=0;i<outputNlines;i++)
{
  printf("output_layer=%d\n",output_layer[i]);
  if(output_layer[i]<0)
  {printf("output_layer error %d\n",output_layer[i]);return -1;}
  int flag=0;
  for(int ind=0;ind<Nlines;ind++)
  {
    if(output_layer[i]==Layer_index[ind])
    {flag=1;}
  }
  if(flag==0)
  {printf("output_layer error %d\n",output_layer[i]);return -1;}
  if(Layer2[output_layer[i]]*Layer3[output_layer[i]]>1280*720)
  {printf("output_layer outstrip 1280*720\n");}
  j=0;
  while(1)
  {
    if(N_para[j].layer==output_layer[i])             //寻找突触前神经元对应neuronbox首地址
    {out_information[i].addr=j;j=0;break;}
    j++;
  }
  printf("output layer addr = %d\n",out_information[i].addr);
  out_information[i].layer=output_layer[i];
  out_information[i].type=Layer1[output_layer[i]];
  out_information[i].Xmax=Layer2[output_layer[i]];
  out_information[i].Ymax=Layer3[output_layer[i]];
  out_length+=out_information[i].Xmax*out_information[i].Ymax;
  out_information[i].last_num=out_length-1;
  out_information[i].all_layer_num=outputNlines;
}
for(i=0;i<outputNlines;i++)
{out_information[i].all_length=out_length-1;}

int out_block_num=(int)ceil((float)(out_length)/320.0);
printf("out_block_num=%d\n",out_block_num);


float *out_timeN=(float *)malloc(sizeof(int)*out_length);
c=0;
for(l=0;l<outputNlines;l++)
{
  shared_image[l].width=out_information[l].Xmax;
  shared_image[l].height=out_information[l].Ymax;
  shared_image[l].length=out_information[l].Xmax*out_information[l].Ymax;
  for(i=0;i<out_information[l].Xmax;i++)
  {
    for(j=0;j<out_information[l].Ymax;j++)
    {
      out_timeN[c]=0;
      shared_image[l].data[i*out_information[l].Ymax+j]=0;
      c++;
    }
  }
}
/*******************************************************************/

  shared->timen=0;
  shared->written=1;

	if(!InitCUDA()) {
        return 0;
    }
    printf("CUDA initialized.\n");

   struct  synapse *Plastic_synapse1;
   struct  synapse *Plastic_synapse2;
   struct  synapse *neuro_syn3;
   struct  synapse *neuro_syn4;
   struct  synapse *neuro_syn5;
   struct  synapse *neuro_syn6;
	 int *input;
   int *output_image;
   struct axon *neuro;
   struct Retina1 *retina;
   struct OutPutLayer *outputlayer_information;
   int *output_timeN;
	 unsigned char *spike;
   struct neuron_I *Ix;
   int *syn_box_num;
   int *input_L;
   int input_num;

   checkCudaErrors(cudaMalloc((void**) &input_L, sizeof(int)));
   checkCudaErrors(cudaMalloc((void**) &retina, sizeof(struct Retina1)*THREAD_Input_NUM*Block_Input_NUM*10));
   checkCudaErrors(cudaMalloc((void**) &THREAD_Input_device_NUM, sizeof(int)));
   checkCudaErrors(cudaMalloc((void**) &neuro, sizeof(struct axon)*all_neuron_number));
	 checkCudaErrors(cudaMalloc((void**) &spike, sizeof(unsigned char) * all_neuron_number));
	 checkCudaErrors(cudaMalloc((void**) &Ix, sizeof(struct neuron_I) * all_neuron_number));
   checkCudaErrors(cudaMalloc((void**) &boxnum, sizeof(int)*6));
   checkCudaErrors(cudaMalloc((void**) &syn_box_num, sizeof(int)*6));
   checkCudaErrors(cudaMalloc((void**) &THREAD_NUM, sizeof(int)*6));
   checkCudaErrors(cudaMalloc((void**) &BLOCK_NUM, sizeof(int)*6));
   checkCudaErrors(cudaMalloc((void**) &THREAD_SYN, sizeof(int)*6));
   checkCudaErrors(cudaMalloc((void**) &BLOCK_SYN, sizeof(int)*6));
   checkCudaErrors(cudaMalloc((void**) &Plastic_synapse1, sizeof(struct synapse)*(Host_THREAD_SYN[0] * Host_BLOCK_SYN[0] *10)));
   checkCudaErrors(cudaMalloc((void**) &Plastic_synapse2, sizeof(struct synapse)*(Host_THREAD_SYN[1] * Host_BLOCK_SYN[1] *10)));
   checkCudaErrors(cudaMalloc((void**) &neuro_syn3, sizeof(struct synapse)*(Host_THREAD_SYN[2] * Host_BLOCK_SYN[2] *10)));
   checkCudaErrors(cudaMalloc((void**) &neuro_syn4, sizeof(struct synapse)*(Host_THREAD_SYN[3] * Host_BLOCK_SYN[3] *10)));
   checkCudaErrors(cudaMalloc((void**) &neuro_syn5, sizeof(struct synapse)*(Host_THREAD_SYN[4] * Host_BLOCK_SYN[4] *10)));
   checkCudaErrors(cudaMalloc((void**) &neuro_syn6, sizeof(struct synapse)*(Host_THREAD_SYN[5] * Host_BLOCK_SYN[5] *10)));
	 checkCudaErrors(cudaMalloc((void**) &input, sizeof(int)*input_length));
   checkCudaErrors(cudaMalloc((void**) &outputlayer_information, sizeof(struct OutPutLayer)*outputNlines));
   checkCudaErrors(cudaMalloc((void**) &output_image, sizeof(int)*out_length));
   checkCudaErrors(cudaMalloc((void**) &output_timeN, sizeof(int)*out_length));

   checkCudaErrors(cudaMemcpy(input_L, &input_length, sizeof(int),cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(retina, Retina_neuron, sizeof(struct Retina1)*THREAD_Input_NUM*Block_Input_NUM*10,cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(outputlayer_information, out_information, sizeof(struct OutPutLayer)*outputNlines,cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(THREAD_Input_device_NUM, &THREAD_Input_NUM, sizeof(int),cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(boxnum, N, sizeof(int)*6,cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(syn_box_num, N_SYN, sizeof(int)*6,cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(THREAD_NUM, Host_THREAD_NUM, sizeof(int)*6,cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(BLOCK_NUM, Host_BLOCK_NUM, sizeof(int)*6,cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(THREAD_SYN, Host_THREAD_SYN, sizeof(int)*6,cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(BLOCK_SYN, Host_BLOCK_SYN, sizeof(int)*6,cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(neuro, N_para, sizeof(struct axon)*all_neuron_number,cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(spike, Signal, sizeof(unsigned char)*all_neuron_number,cudaMemcpyHostToDevice));
	 checkCudaErrors(cudaMemcpy(Plastic_synapse1, Plastic_synap1, sizeof(struct synapse)*(Host_THREAD_SYN[0]*Host_BLOCK_SYN[0]*10) ,cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(Plastic_synapse2, Plastic_synap2, sizeof(struct synapse)*(Host_THREAD_SYN[1]*Host_BLOCK_SYN[1]*10) ,cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(neuro_syn3, neuro_synap3, sizeof(struct synapse)*(Host_THREAD_SYN[2]*Host_BLOCK_SYN[2]*10) ,cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(neuro_syn4, neuro_synap4, sizeof(struct synapse)*(Host_THREAD_SYN[3]*Host_BLOCK_SYN[3]*10) ,cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(neuro_syn5, neuro_synap5, sizeof(struct synapse)*(Host_THREAD_SYN[4]*Host_BLOCK_SYN[4]*10) ,cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(neuro_syn6, neuro_synap6, sizeof(struct synapse)*(Host_THREAD_SYN[5]*Host_BLOCK_SYN[5]*10) ,cudaMemcpyHostToDevice));
   //checkCudaErrors(cudaMemcpy(output_video,shared_image->data, sizeof(int)*shared_image->length ,cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(output_timeN, out_timeN, sizeof(int)*out_length,cudaMemcpyHostToDevice));
	 printf("good\n");
	int timen=0;

	sleep(2);


  struct  timeval start_host;
  struct  timeval end_host;
  unsigned long diff;


  printf("step time=%fms\n",tau_cpu);
	printf("start\n");
	sleep(1);


  cudaStream_t s0;
  cudaStreamCreate(&s0);
  cudaStream_t s1;
  cudaStreamCreate(&s1);
  cudaStream_t s2;
  cudaStreamCreate(&s2);
  cudaStream_t s3;
  cudaStreamCreate(&s3);
  cudaStream_t s4;
  cudaStreamCreate(&s4);
  //cudaSetDevice(0); // Set device 0 as current
  input_num=0;
  for(i=0;i<inputNlines;i++)
  {
  shared_input_image[i].next_frame=1;
  cudaMemcpyAsync(input+input_num,&(shared_input_image[i].data),sizeof(int)*shared_input_image[i].length,cudaMemcpyHostToDevice,s0);
  input_num+=shared_input_image[i].length;
  }
time_init();
int ch = 0;
init_keyboard();
while(global_run)
{
  //break;
  //gettimeofday(&start_host,NULL);
  // cudaEvent_t start, stop;
  // float timex;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0 );
  input_num=0;
   for(i=0;i<inputNlines;i++)
   {
     if(shared_input_image[i].next_frame!=1)
     {
       cudaMemcpyAsync(input+input_num,&(shared_input_image[i].data),sizeof(int)*shared_input_image[i].length,cudaMemcpyHostToDevice,s0);
       shared_input_image[i].next_frame=1;
     }
   input_num+=shared_input_image[i].length;
   }


  for(i=0;i<5;i++)
	{
    /***********************Multi-kernel compute******************************/
    Retina_CNN<<<Block_Input_NUM, THREAD_Input_NUM, 0, s2>>>(input,retina,THREAD_Input_device_NUM,Ix,input_L);
    //neuron compute
    RS_neuron<<<Host_BLOCK_NUM[0], Host_THREAD_NUM[0], 0, s1>>>(input,neuro,spike,Ix,boxnum,THREAD_NUM,BLOCK_NUM);
    FS_neuron<<<Host_BLOCK_NUM[1], Host_THREAD_NUM[1], 0, s2>>>(input,neuro,spike,Ix,boxnum,THREAD_NUM,BLOCK_NUM);
    TC_neuron<<<Host_BLOCK_NUM[2], Host_THREAD_NUM[2], 0, s1>>>(input,neuro,spike,Ix,boxnum,THREAD_NUM,BLOCK_NUM);
    TI_neuron<<<Host_BLOCK_NUM[3], Host_THREAD_NUM[3], 0, s2>>>(input,neuro,spike,Ix,boxnum,THREAD_NUM,BLOCK_NUM);
    TRN_neuron<<<Host_BLOCK_NUM[4], Host_THREAD_NUM[4], 0, s1>>>(input,neuro,spike,Ix,boxnum,THREAD_NUM,BLOCK_NUM);
    LTS_neuron<<<Host_BLOCK_NUM[5], Host_THREAD_NUM[5], 0, s2>>>(input,neuro,spike,Ix,boxnum,THREAD_NUM,BLOCK_NUM);
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    //synapse compute
    Dual_AND_Synapse<<<Host_BLOCK_SYN[0], Host_THREAD_SYN[0], 0, s1>>>(Plastic_synapse1,neuro,spike,Ix,syn_box_num,THREAD_SYN,BLOCK_SYN);
    post_Synapse<<<Host_BLOCK_SYN[1], Host_THREAD_SYN[1], 0, s2>>>(Plastic_synapse2,neuro,spike,Ix,syn_box_num,THREAD_SYN,BLOCK_SYN);
    excit_Synapse<<<Host_BLOCK_SYN[2], Host_THREAD_SYN[2], 0, s1>>>(neuro_syn3,neuro,spike,Ix,syn_box_num,THREAD_SYN,BLOCK_SYN);
    inhi_Synapse<<<Host_BLOCK_SYN[3], Host_THREAD_SYN[3], 0, s2>>>(neuro_syn4,neuro,spike,Ix,syn_box_num,THREAD_SYN,BLOCK_SYN);
    excit_habituation_Synapse<<<Host_BLOCK_SYN[4], Host_THREAD_SYN[4], 0, s1>>>(neuro_syn5,neuro,spike,Ix,syn_box_num,THREAD_SYN,BLOCK_SYN);
    inhi_habituation_Synapse<<<Host_BLOCK_SYN[5], Host_THREAD_SYN[5], 0, s2>>>(neuro_syn6,neuro,spike,Ix,syn_box_num,THREAD_SYN,BLOCK_SYN);

    if(shared->output_spike)
    {fmri<<<out_block_num,320,0,s4>>>(neuro,spike,output_image,outputlayer_information);}
    else
    {out<<<out_block_num,320,0,s4>>>(neuro,spike,output_image,outputlayer_information,output_timeN);}

  }

  timen++;
  // cudaEventRecord( stop, 0 );
  // cudaEventSynchronize( stop );
  // cudaEventElapsedTime( &timex, start, stop );
  // cudaEventDestroy( start );
  // cudaEventDestroy( stop );
  //  printf("GPU_t=%fus\n",timex*1000);

  if(time1S_flag>=1000)
  {
   time1S_flag=0;
   shared->timen=timen;
   printf("N=%d\n\r",timen);
   timen=0;
  }

  cudaDeviceSynchronize();
  if(shared->save)
  {
    shared->save=0;
    time_t timep;
    struct tm *p;
    time (&timep);
    p=gmtime(&timep);
    char file_name[100];
    sprintf(file_name, "load_data/SaveData/%d_%d_%d_%d_%d_%d.txt", 1900+p->tm_year,1+p->tm_mon,p->tm_mday,8+p->tm_hour,p->tm_min,p->tm_sec);
    checkCudaErrors(cudaMemcpy(Plastic_synap1, Plastic_synapse1, sizeof(struct synapse)*(Host_THREAD_SYN[0]*Host_BLOCK_SYN[0]*10) ,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(Plastic_synap2, Plastic_synapse2, sizeof(struct synapse)*(Host_THREAD_SYN[1]*Host_BLOCK_SYN[1]*10) ,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(neuro_synap3, neuro_syn3, sizeof(struct synapse)*(Host_THREAD_SYN[2]*Host_BLOCK_SYN[2]*10) ,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(neuro_synap4, neuro_syn4, sizeof(struct synapse)*(Host_THREAD_SYN[3]*Host_BLOCK_SYN[3]*10) ,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(neuro_synap5, neuro_syn5, sizeof(struct synapse)*(Host_THREAD_SYN[4]*Host_BLOCK_SYN[4]*10) ,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(neuro_synap6, neuro_syn6, sizeof(struct synapse)*(Host_THREAD_SYN[5]*Host_BLOCK_SYN[5]*10) ,cudaMemcpyDeviceToHost));
    FILE *fp_save=fopen(file_name,"w");
    fclose(fp_save);
    save_synapse_func(file_name,Host_THREAD_SYN[0],Host_BLOCK_SYN[0],Plastic_synap1,1,N_SYN[0]);
    save_synapse_func(file_name,Host_THREAD_SYN[1],Host_BLOCK_SYN[1],Plastic_synap2,2,N_SYN[1]);
    save_synapse_func(file_name,Host_THREAD_SYN[2],Host_BLOCK_SYN[2],neuro_synap3,3,N_SYN[2]);
    save_synapse_func(file_name,Host_THREAD_SYN[3],Host_BLOCK_SYN[3],neuro_synap4,4,N_SYN[3]);
    save_synapse_func(file_name,Host_THREAD_SYN[4],Host_BLOCK_SYN[4],neuro_synap5,5,N_SYN[4]);
    save_synapse_func(file_name,Host_THREAD_SYN[5],Host_BLOCK_SYN[5],neuro_synap6,6,N_SYN[5]);
    printf("saved model in load_data/SaveData/%d_%d_%d_%d_%d_%d.txt\n",1900+p->tm_year,1+p->tm_mon,p->tm_mday,8+p->tm_hour,p->tm_min,p->tm_sec);
  }

  if(shared->model)
  {
    shared->step--;
    if(shared->step<=0)
    {
      j=0;
      for(i=0;i<outputNlines;i++)
      {
        if(i>0){j=out_information[i-1].last_num+1;}
        cudaMemcpyAsync(&(shared_image[i].data), output_image+j, sizeof(int)*out_information[i].Xmax*out_information[i].Ymax,cudaMemcpyDeviceToHost,s3);
      }
      while(shared->step<=0)
      {
        if(shared->save)
        {
          shared->step=1;
          break;
        }
        if(kbhit())
        {
          if(readch()=='q')
          {global_run=0;break;}
        }
      }
    }
  }
  else
  {
    if(shared_image[0].read_flag)
     {
       j=0;
       shared_image[0].read_flag=0;
       for(i=0;i<outputNlines;i++)
       {
         if(i>0){j=out_information[i-1].last_num+1;}
         cudaMemcpyAsync(&(shared_image[i].data), output_image+j, sizeof(int)*out_information[i].Xmax*out_information[i].Ymax,cudaMemcpyDeviceToHost,s3);
       }
       shared_image[0].written_flag=0;
     }
  }

  //gettimeofday(&end_host,NULL);
  //diff = 1000000 * (end_host.tv_sec-start_host.tv_sec)+ end_host.tv_usec-start_host.tv_usec;
  //printf("host_time is %ld us\n",diff);
  //gettimeofday(&start_host,NULL);
  if(shared->_switch_!=1){printf("switch OFF\n");break;}
  while(time1MS_flag==0);   //判断是否到达1ms
  time1MS_flag=0;

  if(kbhit())
  {
    if(readch()=='q')
    {global_run=0;break;}
  }

}
  close_keyboard();
  printf("train over\n");
  cudaStreamDestroy(s0);
  cudaStreamDestroy(s1);
  cudaStreamDestroy(s2);
  cudaStreamDestroy(s3);
  cudaStreamDestroy(s4);
  cudaFree(neuro);
  cudaFree(THREAD_SYN);
  cudaFree(BLOCK_SYN);
	cudaFree(Plastic_synapse1);
  cudaFree(Plastic_synapse2);
  cudaFree(neuro_syn3);
  cudaFree(neuro_syn4);
  cudaFree(neuro_syn5);
  cudaFree(neuro_syn6);
  cudaFree(spike);
	cudaFree(Ix);

  if(shmdt(shm) == -1)
    {
      fprintf(stderr, "shmdt failed\n");
      exit(EXIT_FAILURE);
    }
  if(shmctl(shmid, IPC_RMID, NULL) < 0)
    {
             perror("rmerrr1\n");
    }
  if(shmdt(shm_input_image) == -1)
    {
        fprintf(stderr, "shmdt_image failed\n");
        exit(EXIT_FAILURE);
    }
  if(shmctl(shmid_input_image, IPC_RMID, NULL) < 0)
    {
               perror("rmerrr2\n");
    }
  if(shmdt(shm_image) == -1)
    {
      fprintf(stderr, "shmdt_image failed\n");
      exit(EXIT_FAILURE);
    }
  if(shmctl(shmid_image, IPC_RMID, NULL) < 0)
    {
               perror("rmerrr3\n");
    }

 return 0;
}

void init_keyboard()
{
    tcgetattr(0, &initial_settings);
    new_settings = initial_settings;
    new_settings.c_lflag &= ~ICANON;
    new_settings.c_lflag &= ~ECHO;
    new_settings.c_lflag &= ~ISIG;
    new_settings.c_cc[VMIN] = 1;
    new_settings.c_cc[VTIME] = 0;
    tcsetattr(0,TCSANOW, &new_settings);
}

void close_keyboard()
{
    tcsetattr(0,TCSANOW, &initial_settings);
}

int kbhit()
{
    char ch;
    int nread;

    if(peek_character != -1)
    {
        return -1;
    }
    new_settings.c_cc[VMIN] = 0;
    tcsetattr(0, TCSANOW, &new_settings);
    nread = read(0, &ch, 1);
    new_settings.c_cc[VMIN] = 1;
    tcsetattr(0,TCSANOW, &new_settings);

    if(nread == 1)
    {
        peek_character = ch;
        return 1;
    }
    return 0;
}

int readch()
{
    char ch;
    if(peek_character != -1)
    {
        ch = peek_character;
        peek_character = -1;
        return ch;
    }
    read(0, &ch, 1);
    return ch;
}

void time_init()
{

  struct itimerval tick1ms;
  signal(SIGALRM, time_1MS_fun);
  memset(&tick1ms, 0, sizeof(tick1ms));
  //Timeout to run first time
  tick1ms.it_value.tv_sec = 1;
  tick1ms.it_value.tv_usec = 1000;
 //After first, the Interval time for clock
  tick1ms.it_interval.tv_sec = 0;
  tick1ms.it_interval.tv_usec = 1000;
  setitimer(ITIMER_REAL, &tick1ms, NULL);
}

void time_1MS_fun(int signo)
{
  time1MS_flag=1;
  time100MS_flag++;
  time1S_flag++;
}
