################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Program.cpp \
../oclBodySystemOpencl.cpp \
../oclBodySystemOpenclLaunch.cpp \
../oclScan_launcher.cpp 

OBJS += \
./Program.o \
./oclBodySystemOpencl.o \
./oclBodySystemOpenclLaunch.o \
./oclScan_launcher.o 

CPP_DEPS += \
./Program.d \
./oclBodySystemOpencl.d \
./oclBodySystemOpenclLaunch.d \
./oclScan_launcher.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/cuda/include -I"/home/miki/workspace/CUDA SDK/NVIDIA_GPU/shared/inc" -I"/home/miki/workspace/CUDA SDK/NVIDIA_GPU/OpenCL/common/inc" -I/usr/local/include -I/usr/include/qt4 -I/home/dud0/NVIDIA_GPU_Computing_SDK/shared/inc -I/home/dud0/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


