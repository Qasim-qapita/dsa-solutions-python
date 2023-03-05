
from ast import List
from collections import defaultdict
import uuid



#75. Sort Colors
#Approach: Naive Brute Force
#TC:O(n)  - > this is O(n) but it is not single pass, its double pass
#SC:O(1)
def sortColors(nums) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    r,w,b=0,0,0
    for i in nums:
        if i==0:
            r+=1
        elif i==1:
            w+=1
        else:
            b+=1
    
    i=0
    while r>0:
        nums[i]=0
        r-=1
        i+=1
    while w>0:
        nums[i]=1
        w-=1
        i+=1
    while b>0:
        nums[i]=2
        b-=1
        i+=1

#75. Sort Colors
#Tip: watch this https://www.youtube.com/watch?v=aZiqMvaLuSE  ,for understanding the edge case watch this:https://www.youtube.com/watch?v=4xbWSRZHqac
#Approach: Dutch Flag Algorithm
#TC:O(n) -> single pass
#SC:O(1)
#Intuition:
#Dutch flag algotithm works to partition array in three groups, mainly here 0,1,2
#so we take three pointer for solving this problem,
#1st pointer will work as left pointer, 2nd as right pointer and 3rd as mid pointer
#we use mid pointer for traversing the entire array, while left and right pointer is to keep track of 0 and 2
#so when mid enocunter '0' we swap mid with left pointer and incrment both ,(why?): because left can have two values , 0,1 why not 2? because if left had 2
                        #it would have been swapped to right till then, so we never will have 2 on left pointer, so when we swap mid wiht left we increment mid 
                        #and left cause left is now on '0' so we move it next so that it cna be swapped with other vaue again, while for mid
                        #it will have 1 and we have a condition that when mid==1 we increment it, so increment mid as well
#when mid encounter  '2' we swap mid with right and only increment right pointer and we dont increment mid this time (why?): 
                        #because when we swap value between right and mid, afterswapping the mid can have any value which is 0,1,2 and if the new mid value is 0
                        #and we increment mid we wont be able to swap that new '0' value towards left and our algo breaks      
def sortColors(nums) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    left=0
    right=len(nums)-1
    mid=0
    while mid<=right:
        if nums[mid]==0:
            nums[left],nums[mid]=nums[mid],nums[left]
            left+=1
            mid+=1
        elif nums[mid]==2:
            nums[right],nums[mid]=nums[mid],nums[right]
            right-=1
        else:
            mid+=1




#535. Encode and Decode TinyURL
#Approach: Naive Approach
#TC:O(n)
#SC:O(n)
#Intuition:
#FOr this we simply generate new guid for each url request and send shorterned version, while receiving shortened we return long url
class Codec:
    def __init__(self):
        self.UrlMap=defaultdict(str)
        self.TemplateUrl="https://tiny-pp/"       
        
    def encode(self, longUrl: str) -> str:
        """Encodes a URL to a shortened URL.
        """
        uniqueMark=str(uuid.uuid4())
        self.UrlMap[uniqueMark]=longUrl
        return self.TemplateUrl + uniqueMark
        

    def decode(self, shortUrl: str) -> str:
        """Decodes a shortened URL to its original URL.
        """
        uniqueMark=shortUrl.split(self.TemplateUrl)
        return self.UrlMap[uniqueMark[1]]

#Tip: watch this https://www.youtube.com/watch?v=VyBOaboQLGc
#Approach: Two Hashmap Approach
#TC:O(n)
#SC:O(n)
#Intuition:
#In this we keep two hashmaps to keep track of both encode and decode url mapping
#and for uniqueness we used count of encodeUrl like 1...n and put that into tinyUrl
#when user request for encoding we first check whther the long url already exist in our mapping, if it is we return that shortUrl,
#else we put new entry of it
#For decode its simple we just simply return vlaue in deocde map
class Codec2:
    def __init__(self):
        self.encodeMap=defaultdict(str)
        self.decodeMap=defaultdict(str)
        self.templateUrl="https://tiny-pp/"       

    def encode(self, longUrl: str) -> str:
        """Encodes a URL to a shortened URL.
        """
        if longUrl not in self.encodeMap:
            count=len(self.encodeMap)+1
            shortUrl=self.templateUrl+str(count)
            self.encodeMap[longUrl]=shortUrl
            self.decodeMap[shortUrl]=longUrl
        return self.encodeMap[longUrl]
        

    def decode(self, shortUrl: str) -> str:
        """Decodes a shortened URL to its original URL.
        """
        return self.decodeMap[shortUrl]







#554. Brick Wall
#Tip: watch this https://www.youtube.com/watch?v=Kkmv2h48ekw
#Approach: hashmap
#TC:O(n*m)
#SC:O(n) n-> length of wall
#Intuition:
#Instead of counting the bricks we can just count the gaps on each row of wall
#by doing that find min gap by just traversing the walls
def leastBricks(self, wall: List[List[int]]) -> int:
    gaps=defaultdict(int)
    gaps[0]=0  #this need to be set, cause gaps.values will throw error when no element are founded in hashmap
    for i in range(len(wall)):
        wallSum=0
        for j in range(len(wall[i])-1):  #also dont loop over last element, otherwise no brick will be crossed in that case
            wallSum+=wall[i][j]
            gaps[wallSum]+=1
    biggestGap=max(gaps.values())
    return len(wall)-biggestGap   #here we return lenght of wall - biggest gap, why? cause for the lenght of walls we need to get minimum crossed bricks 
                                  #and for that we have susbtract max gap with wall length 

leastBricks([[1,2,2,1],[3,1,2],[1,3,2],[2,4],[3,1,2],[1,3,1,1]])






#122. Best Time to Buy and Sell Stock II
#Tip: watch this https://www.youtube.com/watch?v=3SJ3pUkPQMc
#Approach: Greedy
#TC:O(n)
#SC:O(1)
#Intuition:
#if we think of this problem in very simple perspective, then the intuition
#to solve this problem is to buy stock when price is low, and sell it on day it price is high, that's it!!
def maxProfit(self, prices: List[int]) -> int:
    profit=0
    for i in range(1,len(prices)):
        if prices[i]>prices[i-1]:
            profit+= abs(prices[i]-prices[i-1])
    return profit