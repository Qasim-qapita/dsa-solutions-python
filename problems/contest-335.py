



from collections import defaultdict, deque
import heapq
import math


def passThePillow(n: int, time: int) -> int:
    idx=0
    cur_time=0
    is_reverse=False
    while cur_time<time:
        if idx<n-1 and not is_reverse:
            idx+=1
        else:
            idx-=1
            if idx==0:
                is_reverse=False
            else:
                is_reverse=True


        cur_time+=1
    return idx+1

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def kthLargestLevelSum(self, root:TreeNode, k: int) -> int:
        
        def get_tree_depth(node):
            if node==None:
                return 0
            return 1+max(get_tree_depth(node.left),get_tree_depth(node.right))
        
        if k>get_tree_depth(root):
            return -1
              
        
        def get_level_sum(tree):
            maxHeap=[]
            queue=deque()
            queue.append(tree)
            
            
            while queue:
                cur_sum=0
                n=len(queue)
                for i in range(n):
                    node=queue.popleft()
                    cur_sum+=node.val
                    if node.left:
                        queue.append(node.left)
                    if node.right:
                        queue.append(node.right)    
                heapq.heappush(maxHeap,-cur_sum)
                
            return maxHeap
        
        max_sums=get_level_sum(root)
        
        i=1
        while i<=k:
            maxVal=heapq.heappop(max_sums)
            if i==k:
                return -maxVal
            i+=1
        return -1
    

# obj=Solution()
# tree_node=TreeNode(val=5,left=TreeNode(val=8,left=TreeNode(val=2,left=TreeNode(val=4),right=TreeNode(val=6)),right=TreeNode(val=1)),right=TreeNode(val=9,left=TreeNode(val=3),right=TreeNode(val=7)))
# obj.kthLargestLevelSum(tree_node,2)


#2584. Split the Array to Make Coprime Products
#watch: https://www.youtube.com/watch?v=lyJgJasrLVc
#Question in Lay-man terms: we have to find split in such a way that prime factors of left elements should not present in prime factors of right element
    #for eg: [2,8,7] here split will be [2,8] and [7] why? cause prime factor of 2 is present in [2,8] but not in [7] so its valid split
#Intuition:
#Get prime factors of numbers
#Get last index of prime factors present in array 
#Now we need to loop the nums array and also update our result which will have max of index of whatever number getting iterated
#we will break from loop once our result < idx, why? cause this is the point where we have found the split 
    # (meaning all the factors we encounters so far on left side , and no common factor present in right side of array)
def findValidSplit1(nums)->int:

    #IMP the factors should be prime fators not normal factors, cause we need to check for common PRIME factors not normal factors
    def getPrimeFactors(n):
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

    last_idx_of_factors=defaultdict(int)

    for i in range(len(nums)-1,-1,-1):
        facts=getPrimeFactors(nums[i])
        for f in facts:
            last_idx_of_factors[f]=max(last_idx_of_factors[f],i)
    
    idx=0
    res=0

    while idx<len(nums) and idx<=res:
        for f in getPrimeFactors(nums[idx]):
            res=max(res,last_idx_of_factors[f])
        idx+=1
    return -1 if res==len(nums)-1 else res   #if split is at end of array element, 
                                            #just return -1 cause split cant be made in array as we exhasuted all the elements in array








def findValidSplit(nums) -> int:
    prefix=1
    suffix=1
    total=1
    
    for i in nums:
        total*=i
        
    suffix=total
    
    for i in range(len(nums)):
        prefix*=nums[i]
        suffix= suffix//nums[i]
        if math.gcd(prefix,suffix)==1:
            return i
    return -1

findValidSplit1([4,7,8,15,3,5])