TEMPLATE = app
TARGET = 
DEPENDPATH += . Perlin
INCLUDEPATH += . Perlin
INCLUDEPATH +=	/usr/local/cuda/include
INCLUDEPATH +=	/home/miki/workspace/CUDA_SDK/NVIDIA_GPU/shared/inc
INCLUDEPATH +=	/home/miki/workspace/CUDA_SDK/NVIDIA_GPU/OpenCL/common/inc
INCLUDEPATH +=	/home/dud0/NVIDIA_GPU_Computing_SDK/shared/inc
INCLUDEPATH +=	/home/dud0/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc

LIBS += -lVLCore
LIBS += -lVLGraphics
LIBS += -lGLEW
LIBS += -lOpenCL
LIBS += -lVLQt4
LIBS += -L/home/miki/workspace/CUDA_SDK/NVIDIA_GPU/shared/lib -lshrutil_x86_64
LIBS += -L/home/miki/workspace/CUDA_SDK/NVIDIA_GPU/OpenCL/common/lib -loclUtil_x86_64
LIBS += -L/home/dud0/NVIDIA_GPU_Computing_SDK/shared/lib -lshrutil_x86_64
LIBS += -L/home/dud0/NVIDIA_GPU_Computing_SDK/OpenCL/common/lib -loclUtil_x86_64

# Input
HEADERS += BaseDemo.hpp \
           CLManager.hpp \
           defines.h \
           MCApplet.hpp \
           oclBodySystem.h \
           oclBodySystemOpencl.h \
           oclBodySystemOpenclLaunch.h \
           oclScan_common.h \
           qcontrolswidget.h \
           tables.h \
           Perlin/perlin.h \
           Perlin/perlin.c \
	   ConfigurationData.hpp
FORMS += qcontrolswidget.ui
SOURCES += oclBodySystemOpencl.cpp \
           oclBodySystemOpenclLaunch.cpp \
           oclScan_launcher.cpp \
           Program.cpp \
           qcontrolswidget.cpp \
           Perlin/perlin.c
