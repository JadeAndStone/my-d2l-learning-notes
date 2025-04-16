import numpy
class Vector:
    __a,__b=0,0
    def __init__(self,a,b):
        self.__a,self.__b=a,b
    def __str__(self):
        return 'Vector(%d %d)'%(self.__a,self.__b)
    def __add__(self,other):
        return Vector(self.__a+other.__a,self.__b+other.__b)
    def __mul__(self,other):
        return self.__a*other.__a+self.__b*other.__b
a=Vector(1,3)
b=Vector(2,4)
print(a+b,a*b)