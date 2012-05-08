TEMPLATE = app
TARGET = 
DEPENDPATH += . Perlin
INCLUDEPATH += . Perlin
INCLUDEPATH +=	/usr/local/cuda/include
INCLUDEPATH +=	. inc

LIBS += -lVLCore
LIBS += -lVLGraphics
LIBS += -lGLEW
LIBS += -lOpenCL
LIBS += -lVLQt4
LIBS += -L./lib -lshrutil_x86_64 -loclUtil_x86_64

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
