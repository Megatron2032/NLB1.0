#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/shm.h>
#include <termios.h>
#include <curses.h>    //sudo apt-get install libncurses5-dev
#include <term.h>
static struct termios initial_settings, new_settings;
static int peek_character = -1;
void init_keyboard();
void close_keyboard();
int kbhit();
int readch();

int width;
int height;

struct shared_use_st
{
   int written;//作为一个标志，非0：表示可读，0表示可写
   int text[10];//记录写入和读取的文本
};
struct shared_iamge
{
    int written_flag;//作为一个标志，非0：表示可读，1表示可写
    int read_flag;
    int length;
    int image[1280*720];//记录写入和读取的文本
};


int main()
{
  /*****************共享内存设置***************************/
   void *shm = NULL;
   struct shared_use_st *shared = NULL;
   int shmid;
   //创建共享内存
   shmid = shmget((key_t)1234, sizeof(struct shared_use_st), 0666|IPC_CREAT);
   if(shmid == -1)
   {
       fprintf(stderr, "shmget failed\n");
       exit(EXIT_FAILURE);
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

  /*****************共享内存设置output***************************/
   void *shm_image = NULL;//分配的共享内存的原始首地址
   struct shared_iamge *shared_image;//指向shm_image
   int shmid_image;//共享内存标识符
   //创建共享内存
   shmid_image = shmget((key_t)1235, sizeof(struct shared_iamge), 0666|IPC_CREAT);
   if(shmid_image == -1 )
   {
       fprintf(stderr, "shmget_image failed\n");
       exit(EXIT_FAILURE);
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
   shared_image->read_flag = 1;

   /*****************共享内存设置input***************************/
   void *shm_input_image = NULL;//分配的共享内存的原始首地址
   struct shared_iamge *shared_input_image;//指向shm_image
   int shmid_input_image;//共享内存标识符
   //创建共享内存
   shmid_input_image = shmget((key_t)1236, sizeof(struct shared_iamge), 0666|IPC_CREAT);
   if(shmid_input_image == -1 )
   {
       fprintf(stderr, "shmget_input_image failed\n");
       exit(EXIT_FAILURE);
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
   shared_input_image = (struct shared_iamge*)shm_input_image;
   /******************************************************/
   int c;
   int i=0;
   int n;
   int Nlines=shared->text[7];
   FILE *fout = fopen("load_data/NeuronLayerAddr.txt", "r");
   if(!fout)
   {
    printf("can't open load_data/NeuronLayerAddr.txt\n");
    return -1;
   }
   printf("Nlines=%d\n",Nlines);
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
   }
   fclose(fout);

   printf("wait signal\n");
   while(shared->written!=1);
   printf("signal come\n");
   shared->written=0;
   shared->text[3]=1;
   shared->text[2]=1;
   printf("switch ON\n");
   printf("wait signal\n");
   while(shared->written!=1);
   printf("signal come\n");
   width=shared->text[4];
   height=shared->text[5];
   printf("output_width=%d output_height=%d\n",width,height);

  int ch = 0;
  init_keyboard();
  int num=0;
  while(ch != 'q')
  {
    if(shared_image->written_flag > num)
   {
    num=shared_image->written_flag;

  //  fwrite(shared_image->image, 1, shared_SD->length, fp0);

    shared_image->read_flag=shared_image->read_flag+1;
   }

   if(kbhit()){ch = readch();}
  }
  shared->text[2]=0;
  printf("switch OFF\n");
  close_keyboard();
  if(shmdt(shm) == -1)
  {
    fprintf(stderr, "shmdt failed\n");
    exit(EXIT_FAILURE);
  }
  if(shmdt(shm_image) == -1)
  {
   fprintf(stderr, "shm_image failed\n");
   exit(EXIT_FAILURE);
  }
  exit(0);
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
    read (0, &ch, 1);
    return ch;
}
