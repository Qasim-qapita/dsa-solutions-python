


import math



#7. Reverse Integer
#Approach: Illegal (question asked to assume we dont have 64 bit space to store number sill we did store our res in 64 before checking)
#TC:O(n) -> n is digits in number
#SC: O(1)
#Intuition:
#we just simple convert number into string reversed it then re convert that string into interger and checked whether
#it is in range or not and returned result
def reverse(self, x: int) -> int:
    MAX= 2**31 -1
    MIN= 2**-31
    val =str(abs(x))
    val=val[::-1]
    new_val=int(val)
    if MIN<=new_val<=MAX:
        return new_val if x>0 else -new_val
    else:
        return 0


#Tip: watch this https://www.youtube.com/watch?v=HAgLH58IgJQ
#Approach: Math
#TC:O(log(n))
#SC: O(1)
#Intuition:
#there are two parts in this problem
#1) Is to reverse number
#2) To check the number is 32 bit or not (this is ard to do, cause how to know even before that our number will be out of 32 bit?)
#Sol 1) For first its pretty straightforward , to get last digit we can do mod, and to remove last digit we do divide
    #doing this will give us number in reverse
#Sol 2) Now for hard aprt, we need to ensure before even converting entire digit in reverse is that the reverse number will fit in 32 bit, 
    #to do that we only reverse number till very last digit left, and check that leftover digit with very last digit of MAX or MIn, if 
    #it is out of range we return 0
#Aslo, we check tat if our res which is not fully reversed yet is larger than MAx or lesser than MIn then also we return 0
def reverse(x: int) -> int:
    MAX= 2**31 -1
    MIN= -2**31
    res=0
    while x:
        digit=int(math.fmod(x,10))  #python mods -1 %10 into 9, so to avoid it we use math lib
        x=int(x/10)                  #python divide -1 //10 into -1 instead of 0, to avoid it we do this
        if (res>MAX//10 or (res==MAX//10 and digit>MAX%10)) :
            return 0
        if (res<MIN//10 or (res==MIN//10 and digit<MIN%10)):
            return 0
        res= (res*10) + digit   #this ensures that we append new digits into result properly
    return res   

reverse(123)