from ast import List
from collections import defaultdict



#2013. Detect Squares
#Approach: Math
#TC: O(n)
#SC: O(n)
#Intuition:
#With just one little trick this problem can be solved easily,
#lets think that for new qeury point we need to know out of list of so many query points 
#we need to check which one will form , it would take O(n^3)
#but if we just find point diagonal it take only O(n) time to find how many square will it form 

class DetectSquares:

    def __init__(self):
        self.pointDict=defaultdict(int)
        self.points=[]

    def add(self, point) -> None:
        self.pointDict[tuple(point)]+=1
        self.points.append(point)

    def count(self, point: List[int]) -> int:
        res=0
        qx,qy= point
        for x,y in self.points:
            if abs(x-qx)!=abs(y-qy) or x==qx or y==qy:  #if current point is not diagonal to query point we skip, also we skip if our sqaure is non positive square,
                                                        #means let say we have all four point in one place (dot) it is not positive sqaure so we skip it
                continue
            res+=(self.pointDict[(x,qy)]*self.pointDict[(qx,y)])    #we first multiple different freq of point and add it to our result, 
                                                                    #remember we are using list and not dictionary so we dont need to multiple freq of diagonal point, as it will be accounted in list
        return res