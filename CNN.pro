TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    lenet.cpp \
    batch.cpp
INCLUDEPATH += /usr/include \
            /usr/include/eigen3

LIBS += -L/usr/lib/ -lopenblas -lopencv_core -lopencv_imgproc -lopencv_highgui

HEADERS += \
    im2col.hpp \
    net.hpp \
    numeric.hpp \
    solve.hpp \
    cnn.hpp \
    blas.hpp \
    layers.hpp \
    batch.h
