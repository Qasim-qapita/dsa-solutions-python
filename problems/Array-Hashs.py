
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
def leastBricks(wall) -> int:
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


#122. Best Time to Buy and Sell Stock II
#Tip: watch this https://www.youtube.com/watch?v=3SJ3pUkPQMc
#Approach: Greedy
#TC:O(n)
#SC:O(1)
#Intuition:
#if we think of this problem in very simple perspective, then the intuition
#to solve this problem is to buy stock when price is low, and sell it on day it price is high, that's it!!
def maxProfit(prices) -> int:
    profit=0
    for i in range(1,len(prices)):
        if prices[i]>prices[i-1]:
            profit+= abs(prices[i]-prices[i-1])
    return profit


#560. Subarray Sum Equals K
#Tip: watch this https://www.youtube.com/watch?v=fFVZt-6sgyo
#TC: O(n)
#SC: O(n)
#Intuition:
#Its' little tricky to get the inutitionfor first time right,
#so in order to solve this problem in O(n) we could just use slidind window right?
    #NO!, check the constraist of the problem, we can have negative values in our array, so due to that we dont extactly know
    #whether our array will increase or decrease when we add new element in it (when we do sliding)

#so another approach is to use prefixSum, and using that we can know for sure how many subarray exist with same k value
#So when we are iterating array, we will have our curSum, and if we found the difference (curSum-k) to be exist in our prefixSum hashmap
    #then we know for sure that subarray exist with k sum value, so we just update our result to whatever count of our prefixSum had in hashmap
    #also we update the preFixsum count for new curSum value as well

#For finding difference why curSum-k? and not other way around?
    #reason is we had made prefixSum map for prefixes of curSum, so what we want at end of day is leftover(difference) value of curSum exist in prefix or not
    #and not the leftover value for k! , so due to that we use curSum-k!

def subarraySum(nums,k: int) -> int:
    res=0
    prefixSum=defaultdict(int)
    curSum=0
    prefixSum[0]=1
    for i in range(len(nums)):
        curSum+=nums[i]
        if curSum-k in prefixSum:  
            res+=prefixSum[curSum-k]
        prefixSum[curSum]+=1
    return res



#1930. Unique Length-3 Palindromic Subsequences
#Tip: watch this https://www.youtube.com/watch?v=3THUt0vAFLU
#TC: O(26*n)~ O(n)
#SC: O(n)
#Intuition:
#Main thing to observe here is for plaindrome,
#the palindrome length should be 3, so any subsequence is palindrome if its left element and right element is equal, doesnt matter what mid element is
#using that we traverse each element (considering that element as mid elemnt) and on each traversal,
#we will check if from a-z, a char that exist in both left side and right side of our mid value,
#if it is then we will have palindrome, also to remove duplicates we will add result as pair of (left/right,mid) value
#to keep track of left side of element we can use set,
#but for right side we have to use hashmap, why? 
    #cause while traversing each element we need to know which element no more exist in our right side anymore, and for that we keep track of count of elment
    #when we hit count to 0 means that element should be moved to left side and remove from the right side
def countPalindromicSubsequence(self, s: str) -> int:
    res=set()
    left=set()
    right=collections.Counter(s)
    for i in range(len(s)):
        right[s[i]]-=1
        if right[s[i]]==0:
            right.pop(s[i])
        for c in range(26):
            cur_char=chr(ord('a')+c)
            if cur_char in left and cur_char in right:
                res.add((cur_char,s[i]))
        left.add(s[i])
    return len(res)






#1963. Minimum Number of Swaps to Make the String Balanced
#Tip: watch this https://www.youtube.com/watch?v=3YDBT9ZrfaU   and https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced/solutions/1390319/how-to-reach-the-optimal-solution-explained-with-intuition-and-diagram/
#TC: O(n)
#SC: O(1)
#Intuition:
#Observations to make in this problem is we only need to swap those braces who are not in proper order
#so for that we will get count of all the left braces in array
#now pattern for this is (left_braces+1)/2, so using this we will get our result
#this pattern occurs because let say "]]][[[" this is our array, so whenever we swap one "]" with "[" we will have our sting look like "[]][[]"
#notice something? before swapping we have 3 left unbalces braces but now after swapping we only have 1, right?!!....
#so this with more string will provide a patern that is (left_braces+1)/2 [note: that left_braces are unbalaced braces]
def minSwaps(self, s: str) -> int:
     cur_right_braces=0
     max_right_braces=0
     for i in s:
         if i == "[":
             cur_right_braces-=1
         else:
             cur_right_braces+=1
         max_right_braces=max(cur_right_braces,max_right_braces)
     return (max_right_braces+1)//2


#Tip read this https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-string-balanced/solutions/2961237/100-faster-c-java-python-go-brute-force-solution-easy-to-understand/
#TC: O(n)
#SC: O(1)
#Intuition: same as above just implementation different
def minSwaps(self, s: str) -> int:
    #left means open braces
    #right means close braces
    left=0
    res=0
    
    for i in s:
        if i=="]":
            if left>0:
                left-=1
            else:
                res+=1
                left+=1
        else:
            left+=1
    return res



def interchangeableRectangles(rectangles) -> int:
    
    def getFactorial(n):
        if n<=0:
            return 1
        return n*getFactorial(n-1)
    
    res=0
    areas=defaultdict(float)
    for i in range(len(rectangles)):
        areas[rectangles[i][0]/rectangles[i][1]]+=1
    
    for k,value in areas.items():
        v=int(value)
        if v>=2:
            comb= getFactorial(v)//(2*getFactorial(v-2))
            res+=comb
    return res

interchangeableRectangles([[4,8],[3,6],[10,20],[15,30]])

###########################################################################################
###########################################################################################


# 46 Group Anagram
# Time Complexity: O(mn) m-> total words and n-> maximum char in single word
# Space Complexity: O(mn) m-> total unique anagram and n-> duplicates of anagrams
class GroupAnagram:
    def groupAnagrams(self, strs):
        hashMap = defaultdict(list)

        for word in strs:
            arrStr = self.getArrFromStr(word)
            hashMap[arrStr].append(word)
        
        result = []
        for (key,val) in hashMap.items():
            result.append(val)
        return result
    
    def getArrFromStr(self, s:str):
        arr = [0]* 26
        for num in s:
            arr[ord(num)-ord('a')] +=1

        result = ""
        for num in arr:
            result +=  (str(num) +"-")  # this creats string like 1-4-5-.... string
            # a better way to handle the line 23 is this:
            # result += (str(num) + chr(num+ord('a'))) <- this will create string like 1a4b5s... string
        return result



# 345 Top K Frequent Element
class TopKFreq:
    # Approach: Bucket Sort
    # TC: O(n + k) n-> total number of elemnts in list, k -> elements to iterate till k is satisfied
        # Reason: In worst case scenario each element will have same freq and during that time we will iterate n times
        # but we wont ietarte k elements on each iteration of n loop,
        #  we will only do it once and after we exit the loop that why n+k and not n*k TC comes
    # SC: O(n + k)

    def topKFrequent(self, nums, k: int):
        if len(nums) == 0:
            return []
        # so we will use bucket sort, in which it will be Array<list>,
        # the max length of array will not exceed the len(nums)
        frequency_buckets = [set() for _ in range(len(nums)+1)]  # WARNING: dont create buckets like this : [[]] * len(nums)+1, 
                                                    # this will create array that reference to same empty list hashMap = defaultdict(int)

        hashMap = defaultdict(int)
        result = []
        for num in nums:
            hashMap[num]+=1
        
        for (key,val) in hashMap.items():
            frequency_buckets[val].append(key)
    
        for i in range(len(frequency_buckets)-1, -1, -1):
            if frequency_buckets[i]:
                result.extend(frequency_buckets[i])
                if(len(result)>=k):
                    return result[:k]
        return result

    # Approach: Heap Sort
    # TC: n(log n) -> heapify will take that complexity
    # SC: n 
    def topKFrequent(self, nums, k: int):
        if len(nums) == 0:
            return []
        buckets = [[] for _ in range(len(nums)+1)]  
        hashMap = defaultdict(int)
        result = []
        maxHeap = []

        for num in nums:
            hashMap[num]+=1
        
        for (key,val) in hashMap.items():
            maxHeap.append([-val,key])

        heapq.heapify(maxHeap)
        
        # this loop TC will be around k(log(n)), k times we will pop element from heap
        while len(maxHeap)>0:
            if len(result) == k:
                return result

            val1,key = heapq.heappop(maxHeap)
            result.append(key)
        return result


# 238 Product of Array Except Self
class ProductArrayExceptSelf:
    
    # Approach : Two Pass
    # TC: O(n) n-> number of elements in array
    # SC: O(n) n-> length of result
    def productExceptSelf(self, nums):
        if len(nums) == 0:
            return [0]
        if len(nums) == 1:
            return [nums[0]]

        n = len(nums)
        result = [1] * n
        prefixSoFar = 1
        postfixSoFar = 1

        for i in range(n):
            result[i] *= prefixSoFar
            prefixSoFar *= nums[i] 
        
        for i in range(n-1,-1,-1):
            result[i] *=  postfixSoFar
            postfixSoFar *= nums[i] 

        return result

    # Other Approaches: Brute Force(O(n2)), Use Division(O(n))[ But breaks rule of not using division]

class LongestConsecutiveSequence:
    # TC: O(n) 
        # Reason: every element is checked at most twice
        # (once in the first loop and once in the second loop in the worst case),
        # leading to a time complexity of O(2n), which simplifies to O(n).
    # SC: O(n)

    def longestConsecutive(self, nums) -> int:
        numSet = set(nums)
        longest = 0
        for n in numSet:
            # check if its the start of a sequence
            if (n - 1) not in numSet:
                length = 1
                while (n + length) in numSet:
                    length += 1
                longest = max(length, longest)
        return longest

    #Other Approach: Brute Force (O(nlog(n))) [Sort the array then iterate over it till u find longest sequence]

class ValidPaindrome:

    # TC: O(n) n-> number of letters the string have
    # SC: O(n) we are generating intermidiate sanitized string 
    def isPalindrome(self, s: str) -> bool:
        if len(s) == 1:
            return True

        lowerStr = s.lower()
        finalStr = ""
        for letter in lowerStr:
            if letter.isdigit():
                finalStr += letter
            else:
                letterNum = ord(letter) - ord('a')
                if  letterNum >=0 and letterNum <26 :
                    finalStr += letter

        i = 0
        j = len(finalStr)-1
        while i <= j:
            if finalStr[i] != finalStr[j]:
                return False
            i +=1
            j -= 1
        return True

    # TC: O(n) n-> number of letters the string have
    # SC: O(1)
    def isPalindrome1(self, s: str) -> bool:
        if len(s) == 1:
            return True
        
        left =0
        right =len(s)-1

        # for two pointer the TC is O(n/2), but we take it O(n)
        # Reason: as the size n of your input grows, the size of input of n/2 also grows, 
        # the number of operations grows linearly, which is why both are O(n).
        while left < right:

            while left < right and not s[left].isalnum():
                left +=1
            
            while left < right and not s[right].isalnum():
                right -=1
            
            if s[left].lower() != s[right].lower():
                return False
            
            left +=1
            right -=1

        return True


class ThreeSum:

    # Two pointer
    # TC: O(nlog(n)) + O(n^2) 
    # SC: O(1)
    def threeSum(self, nums):
        if len(nums) < 3:
            return []
        result = []
        nums.sort()

        for (idx,val) in enumerate(nums):
            # Skip positive integers, because once our element is in positive then there is no way now triplet will ever become zero,
            # cause the array is sorted
            if val > 0:
                break
            if idx > 0 and nums[idx] == nums[idx-1]:
                continue
            left = idx + 1
            right = len(nums) - 1
            
            while left < right:
                threeSum = nums[left] + nums[right] + val
                if threeSum == 0:
                    result.append([nums[left], nums[right], val])
                    left +=1
                    right-=1
                    while nums[left] == nums[left-1] and left<right:
                        left+=1
                elif threeSum > 0:
                    right -= 1
                else:
                    left += 1 
        return result



class ContainerWithMostWater:

    # TC: O(n)
    # SC: O(1)
    def maxArea(self, height) -> int:
        if len(height) <=1 :
            return 0
        l, r = 0, len(height) - 1
        result = 0 # will not have area in -ve, so zero is min for now
        while l < r:
            curArea = min(height[l],height[r]) * (r-l)
            result = max(result,curArea)

            # main intuition is, i want to maximize the area,
            # for that we already taken left and right pointer to be far as possible
            # now in order to maximize the area we need to also maximize the height as well
            # so if my rigth height is bigger than my left height then why i need to shift the right?
            # instead i should shift the height which is smaller
            if height[l] < height[r]: 
                l += 1
            else:
                r -= 1
        return result

class BestTimeToBuyAndSellStock:
    
    # Sliding Window
    # TC: O(n)
    # SC: O(1) 
    def maxProfit(self, prices) -> int:
        if len(prices) <=1 :
            return 0

        result = 0
        curStock = prices[0]
        for i in range(1,len(prices)):
            if prices[i] <= curStock:
                curStock = prices[i]
            else:
                curVal = prices[i] - curStock
                result = max(result, curVal)
        return result

    # Kandane Algo
    # TC: O(n)
    # SC: O(1) 
    def maxProfit(self, prices) -> int:
        if len(prices) <=1 :
            return 0

        result = 0
        currentMax = 0
        for i in range(len(prices) - 1):
            currentMax = max(0, currentMax + prices[i+1] - prices[i])
            result = max(result, currentMax)
        return result


class LongestSubstringWihtoutRepeatingCharacters:

    # Two Pointer: we keep track fo start and end, and as we progress end incrementally we also add those
    # in hashMap marking as visited along with those char index
    # when ever the end encounters duplicate in hashMap, it updates the start to that (duplicates' index + 1)[ +1 because end is already on that duplicate char, we dont want start to also point on duplicate char]
    # TC: O(n)
    # SC: O(n)
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) <= 1:
            return len(s)

        #  this hashmap will be used to keep track of duplicates and its indexes
        hashMap = defaultdict(int)

        # min substring with no repitition will be 1
        result = 1
        start,end = 0,1
        
        hashMap[s[start]] = start

        while end < len(s):
            curChar = s[end]

            # if char founded in hashMap means duplicate encountered
            # also the duplicate we encountered should be between start and end,
            # duplicates which are not between start and end doesnt matter,
            # cause we already evaluated those string (out of range from start and end) 
            # and now no use to us
            # this 'if' make sense with ex: "abba"
            if curChar in hashMap and hashMap[curChar] >= start:
                result = max(result, end - start)
                start = hashMap[curChar] + 1 # update the start with index next to duplicate elemtent we encountered
            hashMap[curChar] = end
            end +=1

        # if loop exited before encountering any duplicates then need to update the result
        result = max(result, end - start)
        return result

    # same Approach but cleaner code
    def lengthOfLongestSubstring1(self, s: str) -> int:
        if len(s) <= 1:
            return len(s)

        # Use a regular dictionary to store characters and their indices
        hashMap = {}

        result = 0
        start = 0

        # Iterate over characters and their indices directly
        for end, curChar in enumerate(s):
            if curChar in hashMap and hashMap[curChar] >= start:
                result = max(result, end - start)
                start = hashMap[curChar] + 1
            hashMap[curChar] = end

        # If loop exited without encountering duplicates
        result = max(result, end - start + 1)
        return result

class LongestRepeatingCharacterReplacement:

    def characterReplacement(self, s: str, k: int) -> int:
        
        if len(s) <= k:
            return len(s)

        end = 0
        result = k + 1
        temp_k = k

        for idx,char in enumerate(s):
            curRes = 0
            while temp_k > 0 and end < len(s):
                if char != s[end]:
                    temp_k -= 1
                curRes += 1
                end += 1
            result = max(result,curRes)
            end = idx + 1
            temp_k = k
        return result

obj = LongestRepeatingCharacterReplacement()
obj.characterReplacement("AABABBA", 1)
