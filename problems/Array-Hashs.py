
from ast import List



#75. Sort Colors
#Approach: Naive Brute Force
#TC:O(n)  - > this is O(n) but it is not single pass, its double pass
#SC:O(1)
def sortColors(self, nums: List[int]) -> None:
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
def sortColors(self, nums: List[int]) -> None:
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