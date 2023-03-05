from ast import List


def maxNumOfMarkedIndices(nums) -> int:
    marked=set()
    nums.sort()
    for i in range(len(nums)):
        if i in marked:
            continue
        for j in range(i+1,len(nums)):
            if j in marked:
                continue
            if 2*nums[i]<=nums[j]:
                marked.add(i)
                marked.add(j)
                break
    return len(marked)


maxNumOfMarkedIndices([9,2,5,4])