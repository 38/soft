CXX=g++#clang++-3.6
OBJS=main.o
CFLAGS=-g
INCLUDEDIR=-Iinclude

main: $(OBJS)
	$(CXX) $(OBJS) -o main

-include $(OBJS:.o=.d)

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $(INCLUDEDIR) $*.cpp -o $*.o
	g++ -MM $(CFLAGS) $(INCLUDEDIR) $*.cpp > $*.d

clean:
	rm -f main *.o *.d
