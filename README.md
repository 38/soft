#What is SOFT?
libsoft is a domain-specific language embeded in C++ implemented by C++ templates
for numerical PED solving. 
It provides programmer a high-level interface to describe the mathematical formulas such as
```f <<= Div< X >(g)```

which means assign the divergence of g on the X direction to the field f.

libsoft generates the executable code in compile time for multiple devices. 
The library also hides the device details to the programmer. 

#How to build the library?
Surprisingly, you do not need to build the library. The library does not have any
source code files, just include the headers and go ahead!

#Where can I find examples?
Currently, a demo for 2D heat equation is avaiable in examples/heat\_equation/

