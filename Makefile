ONNX_DIR = /home/ronf/Software/onnxruntime-linux-x64-gpu-1.21.0

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -g\
           -I./include \
           -I$(ONNX_DIR)/include \
           $(shell pkg-config --cflags opencv4)
LDFLAGS = -L$(ONNX_DIR)/lib -lonnxruntime $(shell pkg-config --libs opencv4)

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
TARGET = $(BIN_DIR)/yolo_detector

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) 