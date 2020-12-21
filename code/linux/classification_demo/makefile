CC := g++
CFLAGS := -std=c++11
INC := -I /usr/local/include -I /usr/local/include/opencv -I /usr/local/include/opencv2 
LIB :=  -L /usr/local/lib64 -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core -lopencv_dnn
OBJ := main.o
EXCE := classification_demo
all: $(OBJ)
	$(CC) $(CFLAGS) $(INC) $(LIB) $(OBJ) -o $(EXCE)
	rm -rf *.o
main.o : main.cpp
	$(CC) -c main.cpp
clean:
	rm -rf *.o
	rm -rf $(EXCE)
