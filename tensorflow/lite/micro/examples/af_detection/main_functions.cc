/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <iostream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include <errno.h>
#include <cstdlib>
#include <sys/time.h>
#include <string>
#include <sstream>
#include "tensorflow/lite/micro/examples/af_detection/main_functions.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/models/af_detect_model_data.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
// const char *write_buffer = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 136 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
int fd ;
}  // namespace

uint8_t result_new[8]={0};// 55 AA  f null   f:0 su 1 fail 2 data
/* USER CODE END 0 */

union r_dat{
	float   fdat;
	uint8_t udat[4];
};
struct r_tm{
	float *cost;
};
// return usec
long long get_timestamp(void)//获取时间戳函数
{
    long long tmp;
    struct timeval tv;

    gettimeofday(&tv, NULL);
    tmp = tv.tv_sec;
    tmp = tmp * 1000 * 1000;
    tmp = tmp + tv.tv_usec;
    return tmp;
}
//usart
int set_interface_attribs(int fd, int speed) {
    struct termios tty;
    memset(&tty, 0, sizeof tty);
    if (tcgetattr(fd, &tty) != 0) {
        perror("tcgetattr");
        return -1;
    }

    // ÉèÖÃ²šÌØÂÊ
    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);

    // ÉèÖÃCSIZEÎªCS8£¬ŒŽ8Î»ÊýŸÝ³€¶È
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    // ÉèÖÃCLOCALºÍCREAD£¬Ê¹ÄÜœÓÊÕÆ÷ºÍ±ŸµØÄ£Êœ
    tty.c_cflag |= (CLOCAL | CREAD);

    // ÉèÖÃPARENBÎª0£¬ŒŽÎÞÐ£ÑéÎ»
    tty.c_cflag &= ~PARENB;

    // ÉèÖÃCSTOPBÎª0£¬ŒŽ1Î»Í£Ö¹Î»
    tty.c_cflag &= ~CSTOPB;

    // ÉèÖÃCRTSCTSÎª0£¬ŒŽ²»Ê¹ÓÃÓ²ŒþÁ÷¿ØÖÆ
    tty.c_cflag &= ~CRTSCTS;

    // ÉèÖÃICANONÎª0£¬ŒŽ·Ç¹æ·¶Ä£Êœ£¬ÕâÑùreadŸÍ²»»áÊÜÐÐ»º³åµÄÓ°Ïì
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);

    // ÉèÖÃOPOSTÎª0£¬ŒŽœûÓÃÊä³öŽŠÀí
    tty.c_oflag &= ~OPOST;

    // ÉèÖÃICANONÎª0£¬ŒŽ·Ç¹æ·¶Ä£Êœ
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);

    // ÉèÖÃVMINÎª1£¬VMAXÎª0£¬ÕâÑùreadŸÍ»áÒ»Ö±×èÈû£¬Ö±µœÓÐÊýŸÝ¿É¶Á
    tty.c_cc[VMIN] = 1;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        return -1;
    }

    return 0;
}

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_af_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.

  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<4> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
  //usart
}

// The name of this function is important for Arduino compatibility.
void loop() {
  //usart
    fd = open("/dev/ttyS7", O_RDWR | O_NOCTTY);
    if (fd == -1) {
        perror("open_port: Unable to open /dev/ttyS7");
    }
    if (set_interface_attribs(fd, B115200) == -1) {
        close(fd);
    }
    sleep(1);
    //get data from usart

    float* test_data = input->data.f;
    std::stringstream ss;
    float result[2]={0};
    volatile struct r_tm  rtm;
    volatile union r_dat 	rdat;
    uint8_t dat;
    int len=0,index=0,datalen=0;
    int status=0;
    bool flag = false;
    int ret;
    // long long utime_start=0;
    // long long utime_end=0;
    result_new[0] = 0x55;
    result_new[1] = 0xaa;
    result_new[3] = 0x00;
    rtm.cost = (float *)&result_new[4];
    while(1){
        if((ret = read(fd, &dat, 1)) < 0 ){
            len=0;
            index=0;
            datalen=0;
            status=0;
        }
        switch(status){
            case 0:
                if(dat==0xaa){
                    status=1;
                    //utime_start = get_timestamp();
                }
                break;
            case 1:
                if(dat==0x55){
                    status=2;
                    index = 0;
                }
                else{
                    status=0;
                }
                break;
            case 2:
                if(index==0){
                    datalen=dat;
                    index=1;
                }
                else{
                    datalen|=dat<<8;
                    status=3;
                    len=0;
                    index = 0;
                }
                break;
            case 3:
                rdat.udat[index++] = dat;
                if(index>=4){
                    index = 0;
                    test_data[len++] = rdat.fdat;
                }
                if(len>=datalen){
                    flag = true;
                    status=0;
                    //utime_end = get_timestamp();
                }else if(len>=1250){
                    status=0;
                }
                break;
            default:
                status=0;
                break;
            }
        if(flag){
            flag = false;
			// AI run and record time
            long long start_time = get_timestamp();
			//aiRun((void*)input,(void*)result);
			if (kTfLiteOk != interpreter->Invoke()) {
                MicroPrintf("Invoke failed.");
            }
			long long end_time = get_timestamp();
			long long cost_time = end_time - start_time;
			// *rtm.end = getCurrentMicros();
			*rtm.cost = (float)(cost_time/1000.0);
			// replay results
			result_new[2]  = 0x00;
			TfLiteTensor* output = interpreter->output(0);
			result[0] = output->data.f[0];
            result[1] = output->data.f[1];
			if(result[0]>result[1]){
                result_new[3]  = 0x00;
			}
			else{
				result_new[3]  = 0x01;
			}
			// 55aa00XX  start_tm  end_tm
			write(fd, result_new, 8);
            // std::cout << "result: " <<result[0]<<" "<<result[1]<< std::endl;
		}
		else if(ret < 0){
			// replay failed info
			result_new[2]  = 0x01;
			// 55aa01XX
			write(fd, result_new, 4);

		}
	}
}

