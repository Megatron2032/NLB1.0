#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/shm.h>
#include <termios.h>
//gcc -fPIC -shared Interface/c/init.c Interface/c/input.c Interface/c/output.c -o libNLB.so
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

#define OUTPUT_LAYER 9

struct shared_use_st* NLB_init(int step_ms,int model)
{
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
   {printf( "shared_use_st=%d\n",shmid);}
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

 /*********************************************/
   printf("wait signal\n");
   while(shared->written!=1);
   printf("signal come\n");
   int c;
   int i=0,j=0;
   int n;
   int Nlines=shared->Nlines;
   FILE *fout = fopen("load_data/NeuronLayerAddr.txt", "r");
   if(!fout)
   {
    printf("can't open load_data/NeuronLayerAddr.txt\n");
   }
   printf("Nlines=%d\n",Nlines);
   /*
   int layeraddr[Nlines];
   i=0;
   while (!feof(fout))
   {
       fscanf(fout,"%d",&n);
       layeraddr[i]=n;
       if(i<Nlines)
       {
         printf("layeraddr[%d]=%d\n",i,layeraddr[i]);
       }
       i++;
   }*/
   fclose(fout);

   shared->written=0;
   shared->step=step_ms;
   shared->model=model;

  return shared;
}

void NLB_step(int step_ms,struct shared_use_st *shared)
{
  if(shared->model)
  {
  while(shared->step>0);
  shared->step=step_ms;
  while(shared->step>0);
  }
}
