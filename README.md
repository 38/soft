SOFTLib
======
SOFTLib is a domain-specific language embeded in C++ implemented by C++ templates
for numerical PED solving. 
It provides programmer a high-level interface to describe the mathematical formulas such as
```f <<= Div< X >(g)```

which means assign the divergence of g on the X direction to the field f.

SOFTLib generates the executable code in compile time for multiple devices. 
The library also hides the device details to the programmer. 

Currently, a demo for 2D heat equation is avaiable in examples/heat\_equation/
