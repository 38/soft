
INCLUDEDIR=-I$(LIBDIR)/include

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@

-include $(OBJS:.o=.d)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDEDIR) $*.cpp -o $*.o
	g++ -MM -D__CUDACC__ $(INCLUDEDIR) $*.cpp > $*.d

clean:
	rm -f main *.o *.d $(TARGET)
