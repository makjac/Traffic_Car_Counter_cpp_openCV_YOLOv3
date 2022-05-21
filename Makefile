CC = "g++"
PROJECT = trafficCarCounter
SRC = main.cpp
LIBS = `pkg-config --libs --cflags opencv4`

$(PROJECT) : $(SRC)
	$(CC) -std=c++14 $(SRC) -o $(PROJECT) $(LIBS) 