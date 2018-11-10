#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <sys/shm.h>
#include <termios.h>

struct shared_iamge
{
    int written_flag;//作为一个标志，非0：表示可读，1表示可写
    int read_flag;
    int length;
    int layer;
    int data[1280*720];
    int width;
    int height;
    int next_frame;
};


struct shared_iamge* input_init(int inputNlines)
{
  /*****************共享内存设置input***************************/
  struct shared_iamge *shared_input_image;
  void *shm_input_image;
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
  shared_input_image=(struct shared_iamge *)shm_input_image;
  /*
  int shmid_input_data[inputNlines];//共享内存标识符
  void *shm_input_data[inputNlines];//分配的共享内存的原始首地址
  int *input_data;
  for(int i=0;i<inputNlines;i++)
  {
    //创建共享内存
    //input_data=(int *)malloc(sizeof(int)*shared_input_image[i]->length);
    shmid_input_data[i] = shmget((key_t)(2888+inputNlines+i), sizeof(int)*shared_input_image[i]->length, 0666|IPC_CREAT);
    if(shmid_input_data[i] == -1 )
    {
        fprintf(stderr, "shmget_input_image failed\n");
        exit(EXIT_FAILURE);
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
 return shared_input_image;
}

void input(int *data,struct shared_iamge *shared_input_image,int i,int model)
{
  if(!model)
  {while(shared_input_image[i].next_frame==0);}
  memcpy(shared_input_image[i].data,data,sizeof(int)*shared_input_image[i].length);
  shared_input_image[i].next_frame=0;
}
