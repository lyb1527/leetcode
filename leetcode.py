JIUZHANG:

BINARY SEARCH:

'''
1. Four Keys:

1). start + 1 < end:
two pointers from start and end, if they are next to each other or overlap , then exit the loop

2). start + (end - start) / 2: avoid overflow

3). A[mid] ==, end = mid

 <, >


4). A[start], A[end] ? target



2. Rotated Sorted ArrayList
* Find minimum
* find target
* with duplicates? O(n)


3. Find Median in two Sorted Array


4. Reverse in three Steps
'''

TEMPLATE:

    def findPosition(self, A, target):
        # Write your code here
        start = 0
        end = len(A) - 1


        # exit when start and end are next to each other(only two numbers left) or overlap
        while start + 1 < end:
            mid = start + (end - start) / 2
            if target == A[mid]:
                # when there is duplciates
                # move end, might have same number before mid
                end = mid
                #OR
                # when there is no duplicates, just return mid
                return mid
            elif target > A[mid]:
                start = mid
            else:
                end = mid


        # check where start or end is the target
        if target == A[start]:
            return start
        elif target == A[end]:
            return end
        else:
            return -1

# Closest Number in Sorted Array


# Last Position Of Target
    def findPosition(self, A, target):
        # Write your code here
        if len(A) == 0 or A == None:
            return -1

        start = 0
        end = len(A) - 1

        if target < A[start] or target > A[end]:
            return -1

        while start + 1 < end:
            mid = start + (end - start) / 2
            if target == A[mid]:
                end = mid
            elif target > A[mid]:
                start = mid
            else:
                end = mid

        # want to find the last Position, so check end fisrt
        if target == A[end]:
            return end
        elif target == A[start]:
            return start
        else:
            return -1

# Search a 2D matrix
'''
numbers in first row is bigger than numbers in the second row.
each row is ascending
Given target, return True if present in the matrix

DO duplicates
'''

# two binary Search
#do a BS on row then bs on column

# one binary search
# TC: Log(n*m) = logn + logm
class Solution:

    def searchMatrix(self, matrix, target):
        if len(matrix) == 0:
            return False

        row, col = len(matrix), len(matrix[0])
        # treat as whole list , total of  m * n elements
        start, end = 0, row * col - 1
        while start + 1 < end:
            mid = (start + end) / 2
            number = matrix[mid / col[][mid % col[]
            if number == target:
                return True
            elif: number < target:
                start = mid
            else:
                end = mid

        if matrix[start / col][start % col] == target:
            return True

        if matrix[end / col][end % col] == target:
            return True

        return False


# Maximum Number In Mountain Sequence

# Search In a Big Sorted Array

# Find Minimum In Rotated Sorted Array
'''
[0, 1, 2, 4, 5, 6 7] - > [4, 5, 6, 7, 0, 1, 2]


'''
# Approach: pick the LAST element from the list, compare with mi
#if mid > last element, move start. If < last element, move end

class Solution:
    # @param nums: a rotated sorted array
    # @return: the minimum number in the array
    def findMin(self, nums):
        if len(nums) == 0:
            return 0

        start, end = 0, len(nums) - 1
        target = nums[-1]
        while start + 1 < end:
            mid = (start + end) / 2
            if nums[mid] <= target:
                end = mid
            else:
                start = mid
        return min(nums[start], nums[end])


# Find Peak Element
'''
[1, 2, 1, 3, 4, 5, 7, 6]
peak if i < i -1 an i > i + 1
If many peaks, return any
'''

# O(n), linear pass

# O(logn)
class Solution:
    #@param A: An integers list.
    #@return: return any of peek positions.
    def findPeak(self, A):
        # write your code here
        start, end = 1, len(A) - 2
        while start + 1 <  end:
            mid = (start + end) / 2
            if A[mid] < A[mid - 1]:
                end = mid
            elif A[mid] < A[mid + 1]:
                start = mid
            else:
                end = mid

        if A[start] < A[end]:
            return end
        else:
            return start
# First Bad Version
'''
Versin NUmber : 1 - n
'''
  def findFirstBadVersion(self, n):
        start, end = 1, n
        while start + 1 < end:
            mid = (start + end) / 2
            if SVNRepo.isBadVersion(mid):
                end = mid
            else:
                start = mid

        if SVNRepo.isBadVersion(start):
            return start
        return end

# Search Insert Position
'''
find the first position that  >= target
'''

class Solution:
    """
    @param A : a list of integers
    @param target : an integer to be inserted
    @return : an integer
    """
    def searchInsert(self, A, target):
        if len(A) == 0:
            return 0

        start, end = 0, len(A) - 1
        # first position >= target
        while start + 1 < end:
            mid = (start + end) / 2
            if A[mid] == target:
                # do duplicates, if found, return
                return mid
            elif A[mid] < target:
                start = mid
            else:
                end  = mid

        if A[start] >= target:
            return start
        if A[end] >= target:
            return end
        # position not found
        return len(A)

###### Search In Rotated Sorted Array #######
'''

two rising segments, middle could be either segement.

'''

# 1. O(n), for loop



# 2. Binary Serach (logn)
    def search(self, A, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if A is None or len(A) == 0:
            return -1
        start, end = 0, len(A) - 1

        while start + 1 < end:
            mid = start + (end - start) / 2

            if A[mid] == target:
                return mid

            #middle in first rising segement, if T is between start and mid, move end to mid
            if A[start] < A[mid]:
                if A[start] <= target and target <= A[mid]:
                    end = mid
                else:
                    start = mid

            # mid in second rising segement.
            else:
                # if target is between mid and End, move start to mid
                if A[mid] <= target and target <= A[end]:
                    start = mid
                else:
                    end = mid

        if A[start] == target:
            return start
        if A[end] == target:
            return end
        return -1

# Smallest Rectangle Enclosing Black Pixels

# Total Occurrence Of Target

# Drop Eggs

# First Position Of target

# K Closest Numbers in Sorted Array

# Divide Two Integers

# Search for a range
'''
First target's first position, and last position
'''
    def searchRange(self, A, target):
        if len(A) == 0:
            return [-1, -1]

        start, end = 0, len(A) - 1
        while start + 1 < end:
            mid = (start + end) / 2
            if A[mid] < target:
                start = mid
            else:
                end = mid
        if A[start] == target:
            leftBound = start
        elif A[end] == target:
            leftBound = end
        else:
            return [-1, -1]


        start, end = leftBound, len(A) - 1
        while start + 1 < end:
            mid = (start + end) / 2
            if A[mid] <= target:
                start = mid
            else:
                end = mid
        # check end first for last position
        if A[end] == target:
            rightBound = end
        else:
            rightBound = start
        return [leftBound, rightBound]

# Search a 2D matrix II

'''
HAS duplicates. return number of occurrence of target
'''
#O(n + m), O(1)


class Solution:
    """
    @param matrix: An list of lists of integers
    @param target: An integer you want to search in matrix
    @return: An integer indicates the total occurrence of target in the given matrix
    """
    def searchMatrix(self, matrix, target):
        if matrix == [] or matrix[0] == []:
            return 0

        row, column = len(matrix), len(matrix[0])
        # from bottom left
        i, j = row - 1, 0
        count = 0
        while i >= 0 and j < column:
            if matrix[i][j] == target:
                count += 1
                i -= 1
                j += 1
            elif matrix[i][j] < target:
                j += 1
            elif matrix[i][j] > target:
                i -= 1
        return count

# NOT keeping count
def searchMatrix(self, matrix, target):
    m = len(matrix)
    n = len(matrix[0])

    # from top-right
    i, j = 0, n - 1
    while i < m and j >= 0:
        if matrix[i][j] == target:
            return True
        elif matrix[i][j] < target:
            i += 1
        else: # matrix[i][j] > target:
            j -= 1

    return False


# Binary Search (template)
class Solution:
    # @param {int[]} A an integer array sorted in ascending order
    # @param {int} target an integer
    # @return {int} an integer
    def findPosition(self, A, target):
        # Write your code here
        if len(A) == 0 or A == None:
            return -1

        start = 0
        end = len(A) - 1

        if target < A[start] or target > A[end]:
            return -1

        while start + 1 < end:
            mid = start + (end - start) / 2
            if target == A[mid]:
                return mid
            elif target > A[mid]:
                start = mid
            else:
                end = mid

        # want to find the first Position, so check start fisrt
        if target == A[start]:
            return start
        elif target == A[end]:
            return end
        else:
            return -1



# Sqrt(x)

# Maximum Average Subarray

# Sqrt(x) II

# Find Minimum In Rotated Sorted Array II
'''
Contains duplciates, worst case run-time is O(n )

black box testing: has to go through all n elements to find the
smallest number, which gives O(n)
'''


# Search in Rotated Sorted Array II

# Copy Books

# Wood Cut

# Merge Sorted Array
"""
merge B into A

FOLLOWUP: if A is 10^8, b is 10!
do binary search for 10 numbers to find location in A, then copy the
whole array to a new


"""
def merge(self, A, m, B, n):
    # comparing the bigger values from the back of the list
    indexA = m-1;
    indexB = n-1;
    while indexA >=0 and indexB>=0:

        #senario1: move elements at front of A to the back of A
        if A[indexA] > B[indexB]:
            A[indexA+indexB+1] = A[indexA]
            indexA -= 1

        #senario2. two lists in ascending
        else:
            A[indexA+indexB+1] = B[indexB]
            indexB -= 1
    #senario1 continued: copy B to the front of A
    while indexB >= 0:
         A[indexB] = B[indexB]
         indexB -= 1


class Solution(object):
    def merge(self, nums1, m, nums2, n):
        i = m -1
        j = n - 1
        k = m + n - 1
        while k >= 0 and i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1

        if j >= 0:
            nums1[:j+1] = nums2[:j+1]

# Merge sorted Array II
'''
A + B => C
'''
class Solution:
    def mergeSortedArrayII(self, a , b):
        i = 0
        b = 0
        c = []

        while i <len(a) and j < len(b):
            if a[i] < a[j]:
                a.append(a[i])
                i += 1
            else:
                c.append(b[i])
                j += 1

        # [1, 2, 3, 4], [5, 6, 7, 8]
        # add elements to c that were not added in the while
        if i < len(a):
            while i < len(a):
                c.append(a[i])
                i += 1

        if j < len(b):
            while j < len(b):
                c.append(b[j])
                j += 1
        return c


# Remove Duplicates for sorted array
def removeDuplicates(self, nums):
    if nums == None: return len(nums)

    tail = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[tail - 1]:
            nums[tail] = nums[i]
            tail += 1
    return tail

# Remove Duplicates for sorted array II

def removeDuplicatesII(self, nums):
    if nums == None:
        return len(nums)

    tail = 2
    for i in range(2, len(nums)):
        if nums[i] != nums[tail - 1] or nums[i] != nums[tail]:
            nums[tail] = nums[ni]
            tail += 1

    return tail

# Recover Rotated Sorted Array - REVERSE
'''
Change the rotated array to ascending

EX: 4, 5, 1, 2, 3, -> 1, 2, 3, 4, 5

O(N), hard for O(1)space, do it in-place!!!

THREE STEPS REVERSE: in-place by reverse
1 -> 5, 4, 1, 2, 3

2 -> 5, 4, 3, 2, 1

3 -> 1, 2, 3, 4, 5

'''
class Solution:

    def recoverRotatedSortedArray(self, nums):
        # [4, 5, 1, 2, 3]
        # range(5-1) = > [0, 1, 2, 3], so 3 + 1 will not go out of index
         for i in range(len(nums) - 1):
            if nums[i] > nums[i+1]:
                self.reverse(nums, 0, i)
                self.reverse(nums, i + 1, len(nums) - 1)
                self.reverse(nums, 0, len(nums) - 1)
                return

    def reverse(self, nums, start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1



# Rotate String- REVERSE
'''
Given a string and an offset, rotate string by offset. (rotate from left to right)

offset=0 => "abcdefg"
offset=1 => "gabcdef"
offset=2 => "fgabcde"
offset=3 => "efgabcd"
'''


class Solution:
    # @param s: a list of char
    # @param offset: an integer
    # @return: nothing
    def rotateString(self, s, offset):
        if s is None or len(s) == 0:
            return 0
        n =len(s)
        offset = offset % n
        # has to -1 at the end to get the last element in the list
        self.reverse(s, 0, n - offset - 1)
        self.reverse(s, n - offset, n - 1)
        self.reverse(s, 0, n - 1)

    def reverse(self, s, start, end):
        while start < end:
            s[start], s[end] = s[end], s[start]
            start += 1
            end -= 1

# REverse Wors in a String-REVERSE



# Median Of two Sorted Arrays == Find kth in in two sorted Arrays
'''
find the k/2-th elements for both A and B, if A's k/2 smaller than B's, then
remove the numbers before k/2 in A. Remaining is K - K/2
'''
# TC: log(k)   SC: O(1)
class Solution:
    """
    @param A: An integer array.
    @param B: An integer array.
    @return: a double whose format is *.5 or *.0
    """
    def findMedianSortedArrays(self, A, B):
        n = len(A) + len(B)
        if n % 2 == 1:
            return self.findKth(A, B, n / 2 + 1)
        else:
            smaller = self.findKth(A, B, n / 2)
            bigger = self.findKth(A, B, n / 2 + 1)
            return (smaller + bigger) / 2.0

    def findKth(self, A, B, k):
        if len(A) == 0:
            return B[k - 1]
        if len(B) == 0:
            return A[k - 1]
        if k == 1:
            return min(A[0], B[0])

        a = A[k / 2 - 1] if len(A) >= k / 2 else None
        b = B[k / 2 - 1] if len(B) >= k / 2 else None

        if b is None or (a is not None and a < b):
            return self.findKth(A[k / 2:], B, k - k / 2)
        return self.findKth(A, B[k / 2:], k - k / 2)

# Find kth in in two sorted Arrays
#(A.length + B.length)/2


"""
TWO POINTERS
"""

# Window Sum
'''
an array of integer, a moving window of size k, return the sum of each winow
'''
class Solution:
    def winSum(self, nums, k):
        # Write your code here
        n = len(nums)
        if n < k or k <= 0:
            return []
        sums = [0] * (n - k + 1)
        for i in xrange(k):
            sums[0] += nums[i];

        for i in xrange(1, n - k + 1):
            sums[i] = sums[i - 1] - nums[i - 1] + nums[i + k - 1]

        return sums

# maxiumum window sum


# mover zeroes

# Remove Duplicate Numers in Array




def removeDuplicates(self, nums):
    if nums == None or len(nums) == 0:
        return 0
    nums.sort()
    tail = 1
    for i in range(len(nums):
        if nums[i] != nums[tail]:
            nums[tail] = nums[i]
            tail += 1
    return tail


# valid palindrome

'''
def palindrome(s):
    start, end = 0, len(s) - 1
    while start < end :
        if s[start] != s[end]:
            return False
        start += 1
        end += 1

    return True

'''

# rotate String



# Recover Rotated Sorted Array




# TWO SUM

# two sum - Data Structure Design (can only use hashmap)

# two sum - input arary is sorted

#O(N)
def twoSum(numbers, target):
    if nums == None:
        return []

    l, r = 0, len(nums) - 1
    while l < r:
        sum = nums[l] + nums[r]
        if sum == target:
            return  [l + 1, r + 1] # not zero based
        elif sum < target:
            l += 1
        else:
            r -= 1
    return []


# Two Sum - Unique pairs
'''
how many unique paris such that their sum == target
'''
public class Solution {
    /**
     * @param nums an array of integer
     * @param target an integer
     * @return an integer
     */
    public int twoSum6(int[] nums, int target) {
        // Write your code here
        if (nums == null || nums.length < 2)
            return 0;

        Arrays.sort(nums);
        int cnt = 0;
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int v = nums[left] + nums[right];
            if (v == target) {
                cnt ++;
                left ++;
                right --;
                # until find
                while (left < right && nums[right] == nums[right + 1])
                    right --;
                # first different
                while (left < right && nums[left] == nums[left - 1])
                    left ++;
            } else if (v > target) {
                right --;
            } else {
                left ++;
            }
        }
        return cnt;
    }
}

# 3Sum
'''
all triplets sum to 0
'''

Hash: O(n^2) + O(n)

#two Pointers: O(n^2) + O(1)
class Solution(object):
    '''
        题意：求数列中三个数之和为0的三元组有多少个，需去重
        暴力枚举三个数复杂度为O(N^3)
        先考虑2Sum的做法，假设升序数列a，对于一组解ai,aj, 另一组解ak,al
        必然满足 i<k j>l 或 i>k j<l, 因此我们可以用两个指针，初始时指向数列两端
        指向数之和大于目标值时，右指针向左移使得总和减小，反之左指针向右移
        由此可以用O(N)的复杂度解决2Sum问题，3Sum则枚举第一个数O(N^2)
        使用有序数列的好处是，在枚举和移动指针时值相等的数可以跳过，省去去重部分
    '''
    def threeSum(self, nums):
        nums.sort()
        res = []
        length = len(nums)
        for i in range(0, length - 2):
            if i and nums[i] == nums[i - 1]:
                continue
            target = nums[i] * -1
            left, right = i + 1, length - 1
            while left < right:
                if nums[left] + nums[right] == target:
                    res.append([nums[i], nums[left], nums[right]])
                    right -= 1
                    left += 1
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
                elif nums[left] + nums[right] > target:
                    right -= 1
                else:
                    left += 1
        return res



# Triangle Count
'''
array of ints, how many triangle can we have

A < b < c : if a + b > c, must be a triangle
'''

# Brute force: O(n^3)

# TC: O(n^2) becasue only counting numbers
def triangleCount(S):
    s.sort()
    ans = 0
    for i in range(len(S)):
        left, right = 0, i - 1
        while left < right:
            if S[left] + S[right] > S[i]:
                # +=, not =
                ans  += right - left
                right -= 1
            else:
                left += 1
    return ans




# two sum - less than or equal to target

def twoSumSmaller(nums, target):
    if nums is None or len(nums) < 2:
        return 0

    nums.sort()
    left, right = 0, len(nums) - 1
    count = 0
    while left < right:
        if nums[left] + nums[right] <= target:
            count += right - left
            left += 1
        else:
            right -= 1
    return count

# two sum - greater than target


# two sum closest
def twoSumClosest(nums, target):
    nums.sort()

    left, right = 0, len(nums) - 1
    best = float('inf')
    while left < right:
        diff = abs(nums[left] + nums[right] - target)
        best= min(best, diff)
        if nums[left] + nums[right] < target:
            left += 1
        else:
            right -= 1
    return best


# 3Sum closest


# 4sum

# two sum - difference equals to target




# partition array

def partitionArray(nums, k):
    if nums is None or len(nums) == 0:
        return 0
    left, right = 0, len(nums) - 1

    while left < right:
        # find first one that should NOT on left
        while left < right and nums[left] < k:
            left += 1
        # find first that shoul not on right, from right to left
        while left < right and nums[right] >= k:
            right -= 1

        if left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1


    if nums[left] < k:
        # 0 ... left
        return left + 1
    # 0 ... left - 1
    return left


class Solution:
    """
    @param nums: The integer array you should partition
    @param k: As description
    @return: The index after partition
    """
    def partitionArray(self, nums, k):
        start, end = 0, len(nums) - 1
        while start <= end:
            while start <= end and nums[start] < k:
                start += 1
            while start <= end and nums[end] >= k:
                end -= 1
            if start <= end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1
        return start


# kth smallest numbers in unsorted array
 # Kth largets element



''' divie into left and right'''
# Partition Array By Odd and Even


# Interleaving Positive and negative numbers()
'''
alternating position an negative


'''



# Sort Letters by Case(lower to left, upper to right)


############Partition to 3 parts ###########


# sort Colors
'''array of n objects'''
def sortColors(a):
    if a is None or len(a) == 0:
        return
    left = 0
    right = len(a) - 1
    i = 0
    while i <= right:
        if a[i] == 0:
            # move 0 to left
            a[i], a[left] = a[left], a[i]
            left += 1
            i += 1
        elif a[i] == 1:
            i += 1
        else:
            # swap i and right
            a[i], a[right] = a[right], a[i]
            right -= 1

# Rainbow Sort == sort colors
'''
array of n object with k different colors, sort them so
same colors are adjacent. 1, 2,2, 3, k

MUST do in this IN-PLACE , cannot count
'''
#counting sort, use hash map to count , then expand

TC: O(nlogk)


# pancake sort(possible)


# topological Sort


#--------------------------------------------------------------------
'''linked list'''

# reverse nodes in K-group


# reverse linked list

def reverse(head):
    prev = None
    curr = head
    while curr:
        # store pointer 1 -2
        # then  cur.next -> prev
        tmp = cur.next
        cur.next = prev

        # move two nodes to right, None becomes cur,
        # cur = tmp
        prev = curr
        curr = tmp

    return prev




#########Questions with dummy node

# partition list

# merge two sorted lists

# reverse linked list II

# swap two nodes in linked list

# reorder list

# rotate list

# copy list with random pointer

method1: hashmap

method2: use next


# linked list Cycle

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head is None or head.next is None:
            return False
        slow = head
        fast = head.next
        while slow != fast:
            if fast is None or fast.next is None:
                return False
            slow = slow.next
            fast = fast.next.next
        return True



# linked list cycle II

'''where  the cycle begins'''
  def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None or head.next is None:
            return None
        slow = head
        fast = head.next
        while fast != slow:
            if fast is None or fast.next is None:
                return None
            slow = slow.next
            fast = fast.next.next

        while head != slow.next:
            head = head.next
            slow = slow.next
        return head



# Inetersection of two linked lists
'''essentially just finding where the cycle begins  '''



# Sort List
'''MUST know merge sort version, better know quick sort '''


#####Arrays

# Merge two sorted arrays
def mergeSortedArray(self, a, b )
        i = 0
        j = 0
        c = []

        while i <len(a) and j < len(b):
            if a[i] < a[j]:
                a.append(a[i])
                i += 1
            else:
                c.append(b[i])
                j += 1

        # [1, 2, 3, 4], [5, 6, 7, 8]
        # add elements to c that were not added in the while

        # check to see if one array is not empty
        if i < len(a):
    		while i < len(a):
    			c.append(a[i])
    			i += 1
    	if j < len(b):
    		while j < len(b):
        		c.append(b[j])
    			j += 1

        return c

# merge sorted arrays - B int A
def merge(self, num1, m, num2, n):
    i = m -1
    j = n - 1
    k = m + n - 1
    while k >= 0 and i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            #senario1: move elements at front of A to the back of A
            nums1[k] = nums1[i]
            i -= 1
        else:
            #senario2. two lists in ascending
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
    #senario1 continued: copy B to the front of A
    if j >= 0:
        nums1[:j+1] = nums2[:j+1]


# Intersection of two Arrays
'''3 common ways to solve:
1. hash, put one array into hash, for loop the second array

2. sorting an then compare the first elements for both a and b keep popping until same

3. binary search on one aray

'''

##########Subarray

# Maxium Subarray


#------------------------------------------------------------------------------
#-----------------------------------------------------------------------------
LEETCODE



###############################################################################
'''Tree Questions'''
#################################################################################






# 1. Merge two binary trees

'''
If two nodes overlap, sum is new node. If not , not NULL node is new

'''

class Solution(object):
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        if not t1 or not t2:
            return None

        t1.val += t2.val
        t1.left = self.mergeTrees(t1.left, t2.left)
        t1.right = self.mergeTrees(t1.right, t2.right)

        return t1

class Solution(object):
    def mergeTrees(selfs, t1, t2):
        if not t1: return t2
        st = []
        st.append((t1, t2))
        while st:
            a, b = st.pop()
            if not a or not b:
                continue
            a.val += b.val
            if not a.left:
                a.left = b.left
            else:
                st.append((a.left, b.left))
            if not a.right:
                a.right = b.right
            else:
                st.append((a.right, b.right))
        return t1


# Average Of Levels in Binary Tree
'''
return the average of nodes on each level in the form of an array
'''
class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        if not root:    return []
        res = []
        level = [root]
        while level:
            n = 0.0
            count = 0
            temp = []
            for node in level:
                n += node.val
                count += 1
                if node.left:
                    temp.append(node.left)
                if node.right:
                    temp.append(node.right)
            res.append(n / count)
            level = temp
        return res


class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        queue = [root]
        res = []
        while queue:
            length = len(queue)
            sum = 0
            for i in xrange(length):
                node = queue.pop(0)
                sum += node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(sum/float(length))

        return res





# Find Bottom left Tree Value

'''
Given a BT, find the leftmost value in the last row of a tree
'''
class Solution(object):
    def findBottomLeftValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        candidates = [root]
        while candidates:
            new_candidates = []
            for root in candidates:
                if root.left:
                    new_candidates.append(root.left)
                if root.right:
                    new_candidates.append(root.right)
            if not new_candidates:
                return candidates[0].val
            candidates = new_candidates



class Solution(object):
    def findBottomLeftValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        queue = []
        queue.append(root)
        last = None
        while queue:
            s = queue.pop(0)
            last = s
            if s.right:
                queue.append(s.right)
            if s.left:
                queue.append(s.left)

        return s.val




# Find max in each level


class Solution(object):
    def largestValues(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root is None:
            return []
        result = []
        pq = [root]
        while pq:
            new_pq = []
            res = max(node.val for node in pq)
            result.append(res)
            for node in pq:
                if node.left is not None:
                    new_pq.append(node.left)
                if node.right is not None:
                    new_pq.append(node.right)
            pq = new_pq

        return result

# Maximum Depth Of Bianry Tree


class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int

        if not root:
            return 0

        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
        """

        if not root:
            return 0

        depth = 0
        queue = [root]
        while queue:
            level = []
            for node in queue:
                if node.left:
                    level.append(node.left)
                if node.right:
                    level.append(node.right)
            queue = level
            depth += 1
        return depth



class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0

        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))



# Most Frequent Subtree Sum
'''
Given root of a tree, find the most frequent substree sum.

The subtree sum of a node is efined as the sum of all the node values forme by the subtree rooted at that node.
including the noe itself

It there is a tie, return all the values with the highest frequency in any orer
'''

'''
给定一棵二叉树，求其最频繁子树和。即所有子树的和中，出现次数最多的数字。如果存在多个次数一样的子树和，则全部返回。

注意：你可以假设任意子树和均为32位带符号整数。

解题思路：
树的遍历 + 计数
'''


class Solution(object):
    def findFrequentTreeSum(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        dic = {}
        self.subSum(root, dic)
        m = 0
        res = set()
        #print dic
        for k in dic:
            if dic[k] > m:
                m = dic[k]
                res = set()
                res.add(k)
            if dic[k] == m:
                res.add(k)
        return list(res)



    def subSum(self, root, dic):
        if not root:
            return 0
        s = 0
        if root.left:
            s += self.subSum(root.left, dic)
        if root.right:
            s += self.subSum(root.right, dic)
        s += root.val
        if s not in dic:
            dic[s] = 1
        else:
            dic[s] += 1
        return s


class Solution(object):
    def findFrequentTreeSum(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []

        self._map = {}
        self.max_count = 0

        self.findFrequentTreeSumHelper(root)

        res = []
        for k in self._map:
            if self._map[k] == self.max_count:
                res.append(k)

        return res


    def findFrequentTreeSumHelper(self, root):
        if not root:
            return 0

        left = self.findFrequentTreeSumHelper(root.left)
        right = self.findFrequentTreeSumHelper(root.right)

        _sum = root.val + left + right
        count = self._map.get(_sum, 0) + 1
        self._map[_sum] = count

        self.max_count = max(count, self.max_count)

        return _sum


# Constructing String from binary Tree
'''Construct a string consistis of parenthesis an integers from a binary tree with preorder traversing way
'''

class Solution(object):
    def tree2str(self, t):
        """
        :type t: TreeNode
        :rtype: str
        """
        if t == None:
            return ''
        if t.left == None and t.right == None:
            return str(t.val)
        if t.right == None:
            return str(t.val) + '(' + self.tree2str(t.left) + ')'
        return str(t.val) + '(' + self.tree2str(t.left) + ')' + '(' + self.tree2str(t.right) + ')'


# Construct Binary Tree From String
class Solution(object):
    def str2tree(self, s):


        if not s: return None
        stack, builder = [], ""
        for c in s:
            if c == '-' or c.isdigit():
                builder += c
            else:
                if builder:
                    stack.append(TreeNode(builder))
                    builder = ""
                if c == ')':
                    sub = stack.pop()
                    if not stack[-1].left:
                        stack[-1].left = sub
                    else:
                        stack[-1].right = sub
        if builder:
            stack.append(TreeNode(builder))
        return stack[0]




#Convert BST to Greater Tree


@recursive


@Iterative: reverse inorder traverse


# Invert  Binary Tree
'''

翻转一棵二叉树。
'''

'''

Complexity Analysis

Since each node in the tree is visited only once, the time complexity is O(n)O(n),
 where n is the number of nodes in the tree. We cannot do better than that,
 since at the very least we have to visit each node to invert it.

Because of recursion, O(h)O(h) function calls will be placed on the stack in the worst case,
 where hh is the height of the tree. Because h\in O(n)h∈O(n), the space complexity is O(n)O(n).


'''

class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None

        left = self.invertTree(root.left)
        right = self.invertTree(root.right)

        root.left = right
        root.right = left

        return root

class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return root
        elif not (root.left or root.right):
            return root
        layer = [root]
        while layer:
            nxt = []
            for node in layer:
                node.left,node.right = node.right,node.left
                if node.left:
                    nxt.append(node.left)
                if node.right:
                    nxt.append(node.right)
            layer= nxt
        return root

'''

Since each node in the tree is visited / added to the queue only once, the time complexity is O(n)O(n), where nn is the number of nodes in the tree.

Space complexity is O(n)O(n), since in the worst case, the queue will contain all nodes in one level of the binary tree. For a full binary tree, the leaf level has \lceil \frac{n}{2}\rceil=O(n)⌈
​2
​
​n
​​ ⌉=O(n) leaves.
'''


class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root: return root
        queue = [root]
        while len(queue) > 0:
            tmp = []
            for node in queue:
                node.left, node.right = node.right, node.left
                if node.left:
                    tmp.append(node.left)
                if node.right:
                    tmp.append(node.right)
            queue = tmp
        return root



# Kill Process
'''
给定n个进程，进程ID为PID，父进程ID为PPID。

当杀死一个进程时，其子进程也会被杀死。

给定进程列表和其对应的父进程列表，以及被杀死的进程ID，求所有被杀死的进程ID。

注意：

给定被杀死的进程ID一定在进程列表之中
n >= 1
解题思路：
树的层次遍历

利用孩子表示法建立进程树

然后从被杀死的进程号开始，执行层次遍历。

'''
class Solution(object):
    def killProcess(self, pid, ppid, kill):
        """
        :type pid: List[int]
        :type ppid: List[int]
        :type kill: int
        :rtype: List[int]
        """
        dic = collections.defaultdict(set)
        for child, parent in zip(pid, ppid):
            dic[parent].add(child)
        queue = [kill]
        victims = []
        while queue:
            first = queue.pop(0)
            victims.append(first)
            for child in dic[first]:
                queue.append(child)
        return victims


# Add One Row to Tree




# Binary Tree Tilt

'''
the tilt of a tree node is defined as the absolute difference between the sum of all left subtree noe and
the sum of all right subtree node values. Null node has tilt 0.

The tilt of the whole tree is defined as the sum of all nodes' tilt
'''


#R
'''

recursion

给定二叉树，计算二叉树的“倾斜值”（tilt）

二叉树节点的倾斜值是指其左右子树和的差的绝对值。空节点的倾斜值为0。

注意：

节点和不超过32位整数范围
倾斜值不超过32位整数范围
解题思路：
遍历二叉树 + 递归求二叉树子树和

'''


public class Solution {
    int tilt=0;
    public int findTilt(TreeNode root) {
        traverse(root);
        return tilt;
    }
    public int traverse(TreeNode root)
    {
        if(root==null )
            return 0;
        int left=traverse(root.left);
        int right=traverse(root.right);
        tilt+=Math.abs(left-right);
        return left+right+root.val;
    }
}

'''
Time complexity : O(n)O(n). where nn is the number of nodes. Each node is visited once.
Space complexity : O(n)O(n). In worst case when the tree is skewed depth of tree will be nn. In average case depth will be lognlogn
'''
class Solution(object):
    def findTilt(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        return abs(self.subSum(root.left)-self.subSum(root.right)) + self.findTilt(root.left) + self.findTilt(root.right)

    def subSum(self, node):
        if not node: return 0
        return node.val + self.subSum(node.left) + self.subSum(node.right)




# Sum Of Left Leaves

'''
Find the sum of all left leaves
'''

class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        stack = [root]
        res =0
        while stack:
            curr = stack.pop()


            if not curr.left :
                res += 0
            if curr.left:
                if not curr.left.right and not curr.left.left:
                    res += curr.left.val
                stack+=[curr.left]
            if curr.right:
                stack +=[curr.right]


        return res


class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        """
        """
        if not root:
            return 0

        nodes = [root]
        result = 0

        while nodes:
            next_nodes = []

            for node in nodes:
                if node.left:
                    next_nodes.append(node.left)
                    # node not null
                    if not node.left.left and not node.left.right:
                        result += node.left.val

                if node.right:
                    next_nodes.append(node.right)

            nodes = next_nodes

        return result


# Same Tree
'''
write a func to check if they are the same

SAME: when have structure the same and nodes same values

'''

class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p == q:
            return True

        if p is None or q is None:
            return False

        return p.val == q.val \
            and self.isSameTree(p.left, q.left) \
            and self.isSameTree(p.right, q.right)


class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        queue = [(p, q)]
        while queue:
            node1, node2 = queue.pop(0)
            if not node1 and not node2:
                continue
            elif None in [node1, node2]:
                return False
            else:
                if node1.val != node2.val:
                    return False
                queue.append((node1.left, node2.left))
                queue.append((node1.right, node2.right))
        return True


# Binary Tree Inorder Traversal
Recursive:

class Solution:
    """
    @param root: The root of binary tree.
    @return: Inorder in ArrayList which contains node values.
    """
    def inorderTraversal(self, root):
        # write your code here
        list = []
        self.traverse(root, list)
        return list

    def traverse(self, root, list):
        if root:
            self.traverse(root.left, list)
            list.append(root.val)
            self.traverse(root.right, list)



# Inorder Traversal
# left --  root -- right
def inorderTraversal(self, root):
    if not root:
        return None
    result = []
    stack = []
    curr = root
    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)
        current = curr.right
    return result







# BInary Tree Preorder Traversal

Recursive :

class Solution:

    """
    @param root: The root of binary tree.
    @return: Preorder in ArrayList which contains node values.
    """
    def preorderTraversal(self, root):
        # write your code here
        list = []
        self.recursive_preorder(root, list)
        return list


    def recursive_preorder(self, root, list):
        if root:
            list.append(root.val)
            self.recursive_preorder(root.left,list)
            self.recursive_preorder(root.right,list)




# Binary Tree Upside Down
class Solution:
    # @param {TreeNode} root the root of binary tree
    # @return {TreeNode} the new root
    def upsideDownBinaryTree(self, root):
        # Write your code here
        if root is None or root.left is None:
            return root

        new_root = self.upsideDownBinaryTree(root.left)
        root.left.left = root.right
        root.left.right = root
        root.right = None
        root.left = None
        return new_root


# Kth Smallest element in a BST
'''
find the kth smallest element in BST


What if the BST is modified (insert/delete operations) often and you need to
 find the kth smallest frequently? How would you optimize the kthSmallest routine?
左子树中所有元素的值均小于根节点的值

右子树中所有元素的值均大于根节点的值

因此采用中序遍历（左 -> 根 -> 右）即可以递增顺序访问BST中的节点，从而得到第k小的元素，时间复杂度O(k)

'''
# Binary Search

class Solution:
    def kthSmallest(self, root, k):

        n = self.countNodes(root.left)
        if n + 1 == k:
            return root.val
        elif n + 1 < k:
            # k - n - 1 is the nth node in the right subtree
            return self. kthSmallest(root.right, k - n - 1)
        else:
            # k is the same as the whole tree
            return self.kthSmallest(root.left, k)

    def countNodes(self, node):
        if node == None:
            return 0
        return self.countNodes(node.left) = self.countNodes(node.left) + 1







class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        stack = []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left

            root = stack.pop()
            k -= 1
            if k==0:
                return root.val
            root = root.right

        return -1

class Solution(object):
    def kthSmallest(self, root, k):
        res = []
        while k:
            while root:
                res.append(root)
                root = root.left
            node = res.pop()
            if k == 1:
                return node.val
            k -= 1
            root = node.right
        return -1

'''
For the follow up question, I think we could add a variable to the TreeNode to
 record the size of the left subtree. When insert or delete a node in the left
 subtree, we increase or decrease it by 1. So we could know whether the kth
  smallest element is in the left subtree or in the right subtree by compare
   the size with k.
'''







# Diameter of binary tree
'''
compute the length of the diameter of the tree

the diameter of BT is the lenght of the longest path between any two nodes in a tree
the path may or may not pass thorugh the root

NOTE :the length of path is represented by number of EDGES betwen them

给定一棵二叉树，计算任意两节点之间的边数的最大值。

给定一棵二叉树，计算任意两节点之间的边数的最大值。


解题思路：
解法I 计算子树深度
'''

class Solution(object):

    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.ans = 0
        self.traverse(root)
        return self.ans


    def traverse(self, root):
        if not root: return 0
        left = self.traverse(root.left)
        right = self.traverse(root.right)
        self.ans = max(self.ans, left + right)
        return max(left, right) + 1



from collections import defaultdict

class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        table, stack = defaultdict(set), [root]
        while stack:
            cur = stack.pop()
            if cur not in table:
                table[cur] = set()
            if cur.left:
                table[cur].add(cur.left)
                table[cur.left].add(cur)
                stack.append(cur.left)
            if cur.right:
                table[cur].add(cur.right)
                table[cur.right].add(cur)
                stack.append(cur.right)
        cnt = 0
        while len(table) > 2:
            leaves = [node for node in table if len(table[node]) == 1]
            for leaf in leaves:
                table[table.pop(leaf).pop()].remove(leaf)
            cnt += 2
        return cnt + len(table) - 1

#解法II 遍历二叉树 + 计算子树深度

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def depth(self, root):
        if not root: return 0
        return 1 + max(self.depth(root.left), self.depth(root.right))

    def traverse(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        return max(self.depth(root.left) + 1 + self.depth(root.right), \
                           self.traverse(root.left), \
                           self.traverse(root.right))

    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return max(self.traverse(root) - 1, 0)



# House Robber III
'''
如果两个有边直接相连的房间在同一晚上都失窃，就会自动联络警察。

判断盗贼在不惊动警察的情况下最多可以偷到的金钱数目。

测试用例如题目描述。

解题思路：
解法I 深度优先搜索（DFS）

深度优先遍历二叉树，每次遍历返回两个值：分别表示偷窃或者不偷窃当前节点可以获得的最大收益。
'''

class Solution(object):
    def rob(self, root):
        res = self.helper(root)
        return max(res)
    def helper(self, root):
        res =[0,0]
        if not root: return res
        left = self.helper(root.left)
        right = self.helper(root.right)
        res[0] = root.val + left[1] + right[1]
        res[1] = max(left) + max(right)
        return res



# Serialize and Deserialize binary tree
'''
 a binary tree can be serialized to a string and this string can be deserialized
  to the original tree structure.
'''

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        serialize_str = ''
        if not root:
            return serialize_str

        queue = collections.deque([root])
        while queue:
            front = queue.popleft()
            if serialize_str:
                serialize_str += ','
            if front:
                serialize_str += str(front.val)
                queue.extend([front.left, front.right])
            else:
                serialize_str += '#'

        return serialize_str

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return

        node_vals = data.split(',')
        idx = 0
        root = TreeNode(int(node_vals[idx]))
        queue = collections.deque([root])
        while queue:
            front = queue.popleft()
            idx += 1
            if node_vals[idx] != '#':
                front.left = TreeNode(int(node_vals[idx]))
                queue.append(front.left)

            idx += 1
            if node_vals[idx] != '#':
                front.right = TreeNode(int(node_vals[idx]))
                queue.append(front.right)

        return root






class Codec:
    def serialize(self, root):

        if not root:
            return ''

        queue = [root]
        arr = [str(root.val)]
        while queue:
            node = queue.pop(0)
            if node.left:
                arr.append(str(node.left.val))
                queue.append(node.left)
            else:
                arr.append('x')
            if node.right:
                arr.append(str(node.right.val))
                queue.append(node.right)
            else:
                arr.append('x')

        return '#'.join(arr)


    def deserialize(self, data):

        if not data:
            return None
        arr = data.split('#')
        root = TreeNode(int(arr.pop(0)))
        queue = [root]
        while arr and queue:
            node = queue.pop(0)
            l = arr.pop(0)
            if l != 'x':
                node.left = TreeNode(int(l))
                queue.append(node.left)
            r = arr.pop(0)
            if r != 'x':
                node.right = TreeNode(int(r))
                queue.append(node.right)
        return root


class Codec:
    # BFS
    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if root is None:
            return '[]'
        res = [root.val]
        q = [root]
        while q:
            node = q.pop(0)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            res.append(node.left.val if node.left else '#')
            res.append(node.right.val if node.right else '#')
        while res and res[-1] == '#':
            res.pop()
        return '[' + ','.join(map(str, res)) + ']'



    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if data == '[]':
            return None
        # make new TreeNode for each item in data
        nodes = [TreeNode(o) if o !='#' else None for o in data[1:-1].split(',')] # data[1:-1], '[]' are excluded
        #nodes=[[TreeNode(o), None][o == '#'] for o in data[1:-1].split(',')]
        q = [nodes.pop(0)]
        root = q[0] if q else None # return this in the end

        while q:
            parent = q.pop(0)
            left = nodes.pop(0) if nodes else None
            right = nodes.pop(0) if nodes else None
            parent.left, parent.right = left, right

            if left:
                q.append(left)
            if right:
                q.append(right)
        return root



# Convert Sorted Array to BST

class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return None

        mid = len(nums) // 2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])

        return root




# Convert Univalue Subtree




# Binary Serach Tree Iterator
class BSTIterator(object):
    def __init__(self, root):
        self.stack = []
        self.pushLeft(root)

    # @return a boolean, whether we have a next smallest number
    def hasNext(self):
        return self.stack

    # @return an integer, the next smallest number
    def next(self):
        top = self.stack.pop()
        self.pushLeft(top.right)
        return top.val

    def pushLeft(self, node):
        while node:
            self.stack.append(node)
            node = node.left


# Subtree of Another Tree
'''
Given two non-empty BST s and t, check whether t has the same structure and node values with a subtree of s.
Given tree s:

     3
    / \
   4   5
  / \
 1   2
Given tree t:
   4
  / \
 1   2
'''

'''
Complexity Analysis: preorder Traversal


Time complexity : O(m^2+n^2+m*n). A total of nn nodes of the tree ss and mm nodes of tree tt are traversed. Assuming string concatenation takes O(k)O(k) time for strings of length kk and indexOf takes O(m*n)O(m∗n).

Space complexity : O(max(m,n)). The depth of the recursion tree can go upto nn for tree tt and mm for tree ss in worst case.
'''
class Solution(object):
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        preorder_s = self.preorderString(s)
        preorder_t = self.preorderString(t)
        print(preorder_s)

        return preorder_s.find(preorder_t) != -1


    def preorderString(self, tree):
        out = ""
        stack = [tree]
        while stack:
            curr = stack.pop()
            if curr:
                out = ",".join([out, str(curr.val)])
                stack.append(curr.left)
                stack.append(curr.right)
            else:
                out = out + "!"
        return out



class Solution(object):
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        if self.sameTree(s, t):
            return True
        if not s:
            return False
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)

    def sameTree(self, root1, root2):
        if root1 and root2:
            return root1.val == root2.val and self.sameTree(root1.left, root2.left) and self.sameTree(root1.right, root2.right)
        return root1 is root2


'''
Complexity Analysis

Time complexity : O(m*n). In worst case(skewed tree) traverse function takes O(m*n)O(m∗n) time.

Space complexity : O(n). The depth of the recursion tree can go upto nn. nn refers to the number of nodes in ss.



'''
class Solution(object):
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        if not s or not t:
            return not s and not t
        if self.check(s, t): return True
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)

    def check(self, s, t):
        if not s or not t: return not s and not t
        if s.val != t.val: return False
        return self.check(s.left, t.left) and self.check(s.right, t.right)


# Unique Binary Search Tree
'''
Given n, how many structurally unique BSTs that sotre values 1...n?

Given n = 3, there are a total of 5 unique BST's.

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3

'''
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [1, 1, 2]
        if n <= 2:
            return dp[n]
        else:
            dp += [0 for i in range(n-2)]
            for i in range(3, n + 1):
                for j in range(1, i+1):
                    dp[i] += dp[j-1] * dp[i-j]
            return dp[n]


# Binary Tree Lonest Consecutive Sequence


# Binary Tree Right Sie View
'''
给定一棵二叉树，假设你站在它的右侧，自顶向下地返回你可以观察到的节点的值。

例如，给定上面的二叉树，你应该返回[1, 3, 4]。

解题思路：
二叉树的层次遍历，每层按照从右向左的顺序依次访问节点
'''
class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        ans = []
        if root is None:
            return ans
        queue = [root]
        while queue:
            size = len(queue)
            for r in range(size):
                top = queue.pop(0)
                if r == 0:
                    ans.append(top.val)

                # append right first, so pop will get the first!
                if top.right:
                    queue.append(top.right)
                if top.left:
                    queue.append(top.left)
        return ans



class Solution(object):
    def rightSideView(self, root):
        if not root: return []
        ans = []
        self.rightView(root, ans, 0)
        return ans

    def rightView(self, root, ans, level):
        if not root: return
        if level == len(ans):
            ans.append(root.val)
        self.rightView(root.right, ans, level+1)
        self.rightView(root.left, ans, level+1)


# Binary Tree PostOrder Traversal

# Left, Right, Root
class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        res = []
        stack = []
        stack.append(root)
        while stack:
            curr = stack[-1]
            while curr.left:
                tmp = curr.left
                stack.append(curr.left)
                curr.left = None
                curr = tmp
            if curr.right:
                stack.append(curr.right)
                curr.right = None
            else:
                res.append(curr.val)
                stack.pop()
        return res

class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root: return []

        output = []
        stack = [root]

        while stack:
            tmp = stack.pop()
            if tmp.left:
                stack.append(tmp.left)

            if tmp.right:
                stack.append(tmp.right)

            output.insert(0,tmp.val)

        return output

@similar : two stacks
class Solution:
    def postorderTraversal(self, root):
        if not root:
            return []
        stack = [root]
        ans = []
        while stack:
            top = stack.pop()
            ans.append(top.val)

            if top.left:
                stack.append(top.left)
            if top.right:
                stack.appen(top.right)

        return ans[::-1]



# keep visited flags while traversing, can be solved more directly.


class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        self.stack, result = [], []
        rightchildVisited = set()
        self.gotoLeaf(root) # visit the root's left children
        while self.stack:
            root = self.stack[-1]
            if root in rightchildVisited: # the right children are visited, we can return the root
                result.append(root.val)
                self.stack.pop()
            else:
                rightchildVisited.add(root) # the right children are not visited
                self.gotoLeaf(root.right)
        return result
    def gotoLeaf(self, node):
        while node:
            self.stack.append(node)
            node = node.left



use a prev variable to keep track of the trevisouly-traverse node. Curr is the current node that
is on top of the stack. When prev is curr's parent, we travseing down the Tree
. In this case, we try to traverse to curr's left child if avaible(put left chil to stack)

if not available, look at curr's right chil. If both not exists(curr is leaf), we print curr's value
and pop it off the stack

If prev is curr's left, we are traversing up the tree from left. We look at the right fchild,
if it is available, traverse down the right child(push to stack).

If prev is curr's right chil, traverseing up the tree from the right, in this case,
we print curr's value and pop it off the stack

# Binary Tree level Order Traversal II

'''
Given a BT, return the bottom-up level order traversal of its nodes' values

from left to right, level by level from leaf to root
'''

class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []

        result = []
        queue = [(root, 0)]

        while queue:
            node, dep = queue.pop(0)

            if len(result) == dep:
                result.append([])
            result[dep].append(node.val)

            if node.left:
                queue.append((node.left, dep + 1))
            if node.right:
                queue.append((node.right, dep + 1))

        return result[::-1]


class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root == None:
            return []
        queue = [root]
        result = []
        while queue:
            newq = []
            newr = []
            for item in queue:
                newr.append(item.val)
                if item.left:
                    newq.append(item.left)
                if item.right:
                    newq.append(item.right)
            queue = newq
            result.append(newr)
        return result[::-1]



# Path Sum III
'''
Given a BT in which each node contains an integer value

Find the number of paths that sum to a given value

path does NOT need start or end at root or a leaf, but it must go gownwards
(travelling only from parent nodes to child nodes)
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

Return 3. The paths that sum to 8 are:

1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11

'''

# 78 ms , DFS
class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        sumDict = dict()
        sumDict[0] = 1
        return self.dfs(root, 0, sum, sumDict)

    def dfs(self, root, sum, target, sumDict):
        if not root:
            return 0
        sum += root.val
        count = sumDict.get(sum - target, 0)
        sumDict[sum] = sumDict.get(sum, 0) + 1
        count += self.dfs(root.left, sum, target, sumDict)
        count += self.dfs(root.right, sum, target, sumDict)
        sumDict[sum] -= 1
        return count


class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        self.sum = sum
        self.result = 0
        if not root:
            return 0
        self.dfs(root, [])
        return self.result

    def dfs(self, node, vl):
        if node:
            vl = [i+node.val for i in vl] + [node.val]
            self.result += vl.count(self.sum)
            self.dfs(node.left, vl)
            self.dfs(node.right, vl)


class Solution(object):
    def _sum(self, root, target):
        if not root:
            return 0
        result = 0
        if root.val == target:
            result += 1
        result += self._sum(root.left, target - root.val)
        result += self._sum(root.right, target - root.val)
        return result


    def pathSum(self, root, sum):
        if not root:
            return 0
        result = self._sum(root, sum)
        result += self.pathSum(root.left, sum)
        result += self.pathSum(root.right, sum)
        return result






# Binary Tree Longest consecutive Sequence II



# Binary Tree level order Traversal
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        self.results = []
        if not root:
            return self.results
        q = [root]
        while q:
            new_q = []
            self.results.append([n.val for n in q])
            for node in q:
                if node.left:
                    new_q.append(node.left)
                if node.right:
                    new_q.append(node.right)
            q = new_q
        return self.results

# Closet Binary Seasrch Teee Value
???????????????


# Lowest Common Ancestor of a binary search tree
'''
find LCA of two given nodes in BST

 ___2__          ___8__
   /      \        /      \
   0      _4       7       9
         /  \
         3   5
For example, the lowest common ancestor (LCA) of nodes 2 and 8 is 6. Another
example is LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.


记当前节点为node，从根节点root出发

若p与q分别位于node的两侧，或其中之一的值与node相同，则node为LCA

否则，若p的值＜node的值，则LCA位于node的左子树

否则，LCA位于node的右子树
'''
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """

        current = root
        if not current:
            return current
        cmin = min(p.val, q.val)
        cmax = max(p.val, q.val)
        while current.val < cmin or current.val > cmax:
            if current.val < cmin:
                current = current.right
            else:
                current = current.left
        return current

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        while root:
            if root.val > p.val and root.val > q.val:
                root = root.left
            elif root.val < p.val and root.val < q.val:
                root = root.right
            else:
                return root

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root

# closest Binary Search Tree value II


# Symmetic Tree
'''
Given BT, check whether it is a a mirror of itself.
For example, this binary tree [1,2,2,3,4,4,3] is symmetric:

    1
   / \
  2   2
 / \ / \
3  4 4  3
'''

"""
Two trees are a mirror reflection of each other if:

1. Their two roots have the same value.
2. The right subtree of each tree is a mirror reflection of the left subtree of the other tree.
"""

# Approach 1: recursive
'''
TC: Because we traverse the entire input tree once, the total run time is
 O(n), where nn is the total number of nodes in the tree.


SC: ee. In the worst case, the tree is linear and the height is in O(n)O(n).
 Therefore, space complexity due to recursive calls on the stack is O(n)O(n) in the worst cas
'''

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.isMir(root, root)

    def isMir(self, node1, node2):
        if node1 is None and node2 is None:
            return True
        if node1 is None or node2 is None:
            return False
        return (node1.val == node2.val) and self.isMir(node1.left, node2.right) and self.isMir(node1.right, node2.left)



# Approach 2: iterative

'''
e total run time is O(n)O(n), where nn is the total number of nodes in the tree.

There is additional space required for the search queue. In the worst case, we
 have to insert O(n)O(n) nodes in the queue. Therefore, space complexity is O(n)O(n).
'''

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        queue = collections.deque()
        queue.append(root)
        queue.append(root)

        while queue:
            t1 = queue.popleft()
            t2 = queue.popleft()
            if t1 == None and t2 == None:
                continue
            if t1 == None or t2 == None:
                return False
            if t1.val != t2.val:
                return False

            # outter mirror
            queue.append(t1.left)
            queue.append(t2.right)
            # INNTER MIRROR
            queue.append(t1.right)
            queue.append(t2.left)

        return True


class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root is None:
            return True
        stack = [(root.left, root.right)]
        while stack:
            left, right = stack.pop()
            if left is None and right is None:
                continue
            if left is None or right is None:
                return False
            if left.val == right.val:
                stack.append((left.left, right.right))
                stack.append((left.right, right.left))
            else:
                return False
        return True

# Find mode In Binary Search Teee
'''
给定一棵包含重复元素的二叉树。寻找其中的所有众数。

注意：二叉树可能包含多个众数，以任意顺序返回即可。

'''

class Solution(object):
    def findMode(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        from collections import Counter

        if not root:
            return []
        stack = [root]
        c = Counter()

        while stack:
            tmp = stack.pop()
            c[tmp.val] += 1

            if tmp.right:
                stack.append(tmp.right)
            if tmp.left:
                stack.append(tmp.left)

        mode = max(c.values())

        return [key for key in c if c[key] == mode]


# Binary Tree Paths
'''
GIven a BT, return all root to leaf paths

给定一棵二叉树，返回所有“根到叶子”的路径。
'''
class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """

        if not root:
            return []

        res = []
        stack = [[root, ""]]

        while stack:
            node, lstr = stack.pop()

            if not node.left and not node.right:
                res.append(lstr + str(node.val))
            if node.left:
                stack.append([node.left, lstr + str(node.val) + "->"])
            if node.right:
                stack.append([node.right, lstr + str(node.val) + "->"])
        return res

class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        if not root:
            return []

        self.result = []
        self.helper(root, '')

        return self.result

    def helper(self, root, cur):
        cur += str(root.val)
        if not root.left and not root.right:
            self.result.append(cur)
            return

        if root.left:
            self.helper(root.left, cur + '->')

        if root.right:
            self.helper(root.right, cur + '->')


# Balanced Binary Tree
'''

Given a BT, etermine if it is hgiehgt balanced.

Is balanced if depth of two substrees of every node never differ by more than 1
'''
class Solution:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """
    def isBalanced(self, root):
        balanced, _ = self.validate(root)
        return balanced

    def validate(self, root):
        if root is None:
            return True, 0

        balanced, leftHeight = self.validate(root.left)
        if not balanced:
            return False, 0
        balanced, rightHeight = self.validate(root.right)
        if not balanced:
            return False, 0

        return abs(leftHeight - rightHeight) <= 1, max(leftHeight, rightHeight) + 1


public class Solution {
    public boolean isBalanced(TreeNode root) {
        return maxDepth(root) != -1;
    }

    private int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }

        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        if (left == -1 || right == -1 || Math.abs(left-right) > 1) {
            return -1;
        }
        return Math.max(left, right) + 1;
    }
}





# Populating next Right Pointers in Each Node

        1 -> Null
    2   ->   3 -> Null
 3  -> 4 ->  5 ->  6 -> NULL

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):

        if not root:
            return None
        queue = collections.deque()
        queue.append(root)
        while queue:

            currLevel = []
            for i in range(len(queue)):
                node = queue.popleft()
                currLevel.append(node)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            for i in range(len(currLevel)-1):
                currLevel[i].next  = currLevel[i+1]

            currLevel[len(currLevel) - 1].next = None




# Populating Next Right Pointers In Each Node II
I is perfect binary tree, will that solution work
for any binary tree?


# Delete node in a BSET
'''

Basically, the deletion can be divided into two stages:

1. Search for a node to remove.
2. If the node is found, delete the node.

Note: Time complexity should be O(height of tree).
'''


class Solution(object):
    def deleteNode(self, root, key):
        """
        :type root: TreeNode
        :type key: int
        :rtype: TreeNode
        """
        pre, cur = None, root
        while cur and cur.val != key:
            pre = cur
            if key < cur.val:
                cur = cur.left
            elif key > cur.val:
                cur = cur.right
        if not cur: return root

        ncur = cur.right
        if cur.left:
            ncur = cur.left
            self.maxChild(cur.left).right = cur.right

        if not pre: return ncur

        if pre.left == cur:
            pre.left = ncur
        else:
            pre.right = ncur
        return root

    def maxChild(self, root):
        while root.right:
            root = root.right
        return root




class Solution(object):
    def deleteNode(self, root, key):
        """
        :type root: TreeNode
        :type key: int
        :rtype: TreeNode
        """
        if not root:
            return None
        if root.val == key:
            root = self.update_node(root)
        else:
            if root.val > key:
                root.left = self.deleteNode(root.left, key)
            else:
                root.right = self.deleteNode(root.right, key)
        return root

    def update_node(self, node):
        # if right node, return left
        if node.right == None:
            return node.left
        else:
            tempNode = node.right
            while tempNode.left:
                tempNode = tempNode.left
            tempNode.left = node.left
            return node.right



# Sum Root to Leaf Numbers
'''

sum for all numbers from paths from root to leaves

'''

class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        self.ans = 0
        self.helper(root, 0)
        return self.ans

    def helper(self, root, sumAbove):
        tmp = root.val + 10 * sumAbove
        if not root.left and not root.right:
            self.ans += tmp
            return
        if root.left: left = self.helper(root.left, tmp)
        if root.right: right = self.helper(root.right, tmp)


class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0

        return self.helper(root,0)

    def helper(self,root,s):
        if not root: return 0

        if root and not root.left and not root.right:
            return s*10 + root.val

        return self.helper(root.left,s*10+root.val) + \
        self.helper(root.right,s*10+root.val)



class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.sum = 0
        self.recur(root, 0)
        return self.sum

    def recur(self, node, val):
        if not node:
            return
        val = val * 10 + node.val
        if not node.left and not node.right:
            self.sum += val
        self.recur(node.left, val)
        self.recur(node.right, val)


# Flatten Binary Tree To Linked List

'''
 each node's right child points to the next node of a pre-order traversal.

Given

         1
        / \
       2   5
      / \   \
     3   4   6
The flattened tree should look like:
   1
    \
     2
      \
       3
        \
         4
          \
           5
            \
             6
'''
class Solution(object):
	def __init__(self):
	    self.prev = None

	def flatten(self, root):
	    if not root:
	        return None
	    self.flatten(root.right)
	    self.flatten(root.left)

	    root.right = self.prev
	    root.left = None
	    self.prev = root

class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """

        stack = [root]
        prev = None
        while stack:
            node = stack.pop()
            if not node:
                continue
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
            if prev:
                prev.right = node
                prev.left = None
            prev = node

class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        if not root:
            return
        right = root.right
        if root.left:
            # Flatten left subtree
            self.flatten(root.left)
            # Find the tail of left subtree
            tail = root.left
            while tail.right:
                tail = tail.right
            # left <-- None, right <-- left, tail's right <- right
            root.left, root.right, tail.right = None, root.left, right
        # Flatten right subtree
        self.flatten(right)


# Binary Tree zigzag level order travesral
'''
return zigzag level order travesral of nodes values
from left to right, then right to left for the next level
'''

class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root == None:
            return []
        result = []
        curLevel = [root]
        direction = "L"
        while(curLevel):
            nextLevel = []
            curR = []
            for node in curLevel:
                curR.append(node.val)
                if node.left:
                    nextLevel.append(node.left)
                if node.right:
                    nextLevel.append(node.right)
            if direction == "L":
                result.append(curR)
                direction = "R"
            else:
                result.append(curR[::-1])
                direction = "L"
            curLevel = nextLevel
        return result

class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return []
        res, queue, temp,flag = [],[root],[],1
        while queue:
            for _ in range(len(queue)):
                stack = queue.pop(0)
                temp +=[stack.val]
                if stack.left:
                    queue.append(stack.left)
                if stack.right:
                    queue.append(stack.right)
            res += [temp[::flag]]
            temp =[]
            flag *= -1
        return res


class Solution:
    """
    @param root: The root of binary tree.
    @return: A list of list of integer include
             the zig zag level order traversal of its nodes' values
    """
    def zigzagLevelOrder(self, root):
        self.results = []
        self.preorder(root, 0, self.results)
        return self.results

    def preorder(self, root, level, res):
        if root:
            if len(res) < level+1: res.append([])
            if level % 2 == 0:
                res[level].append(root.val)
            else:
                res[level].insert(0, root.val)
            self.preorder(root.left, level+1, res)
            self.preorder(root.right, level+1, res)


# Path Sum
'''
Given a BT and sum, values along root to leaf == target

'''
class Solution(object):
    def hasPathSum(self, root, sum):

        if not root:
            return False
        if root.left is None and root.right is None:
            return sum == root.val
        return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)


class Solution(object):
    def hasPathSum(self, root, sum):

        #DFS深度搜索
        if not root:
            return False
        stack = []
        stack.append((root,root.val))
        while stack:
            curNode,curSum = stack.pop()
            if not curNode.left and not curNode.right and curSum == sum:
                return True
            if curNode.left:
                stack.append((curNode.left,curSum+curNode.left.val))
            if curNode.right:
                stack.append((curNode.right,curSum+curNode.right.val))
        return False


class Solution(object):
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if root is None:
            return False
        if sum == root.val and root.left is None and root.right is None:
            return True
        left = self.hasPathSum(root.left, sum - root.val)
        right = self.hasPathSum(root.right, sum - root.val)
        return left or right

# Populating Next Right Pointers in Each Node II



# Path Sum II
'''
find all root-to-leaf paths where each path equal target
'''

class Solution(object):
    def pathSum(self, root, target):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        res = []
        if not root:
            return res
        self.backtrack(root, [], target, res)
        return res

    def backtrack(self, root, path, target, res):
        if not root.left and not root.right:
            if root.val + sum(path) == target:
                res.append(path+[root.val])
            return

        if root.left:
            self.backtrack(root.left, path+[root.val], target, res)
        if root.right:
            self.backtrack(root.right, path+[root.val], target, res)

class Solution(object):
    def dfs(self, root, target, path, res):
        if root.left is None and root.right is None:
            if root.val == target:
                res.append(path + [target])
        else:
            path.append(root.val)
            if root.left:
                self.dfs(root.left, target - root.val, path, res)
            if root.right:
                self.dfs(root.right, target - root.val, path, res)
            path.pop()

    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        res = []
        if root != None:
            self.dfs(root, sum, [], res)
        return res


# Minimum Depth of Binary Tree
'''


'''

class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        Queue = [root]

        depth = 0
        while Queue:
            depth += 1
            for i in xrange(len(Queue)):
                node = Queue.pop(0)
                if not node.left and not node.right:
                    return depth
                if node.left:
                    Queue.append(node.left)
                if node.right:
                    Queue.append(node.right)
        return depth


class Solution:
    # @param root, a tree node
    # @return an integer
    def minDepth(self, root):
        if root == None:
            return 0
        if root.left==None or root.right==None:
            # one of the subtree is lenght 0
            return self.minDepth(root.left)+self.minDepth(root.right)+1
        return min(self.minDepth(root.right),self.minDepth(root.left))+1




# Construct Binary Tree from Inorder and postorder traversal

class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        i,j=0,0
        stack=[]
        cur=None
        while j<len(postorder):
            if stack and stack[-1].val==postorder[j]:
                stack[-1].right=cur
                cur=stack.pop()
                j+=1
            else:
                stack.append(TreeNode(inorder[i]))
                stack[-1].left=cur
                cur=None
                i+=1
        return cur



class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if len(inorder) == 0:
            return None
        mid = inorder.index(postorder.pop(-1))
        root = TreeNode(inorder[mid])
        root.right = self.buildTree(inorder[mid+1:], postorder)
        root.left = self.buildTree(inorder[:mid], postorder)
        return root




# Construct Binary tree from preoder an inorder traversal


class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if len(inorder) == 0:
            return None
        ind = inorder.index(preorder.pop(0))
        root = TreeNode(inorder[ind])
        root.left = self.buildTree(preorder, inorder[0:ind])
        root.right = self.buildTree(preorder, inorder[ind+1:])
        return root

class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """

        if len(preorder) == 0:
            return None

        head = TreeNode(preorder[0])
        stack = [head]
        i = 1
        j = 0

        while i < len(preorder):
            temp = None
            t = TreeNode(preorder[i])
            while stack and stack[-1].val == inorder[j]:
                temp = stack.pop()
                j += 1
            if temp:
                temp.right = t
            else:
                stack[-1].left = t
            stack.append(t)
            i += 1

        return head


# Unique Binary Search Trees II
'''
Unique BST I: how many are there?

This Questions:  return all  unique BST's shown below.

'''


class Solution(object):
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        #res = []
        if n == 0:
            return []
        return self.buildBST(1,n)

    def buildBST(self,lo,hi):
        if lo > hi:
            return [None]
        res = []
        for i in range(lo,hi+1):
            #root = TreeNode(i)
            left = self.buildBST(lo,i-1)
            right = self.buildBST(i+1,hi)
            for j in left:
                for k in right:
                    root = TreeNode(i)
                    root.left = j
                    root.right = k
                    res.append(root)
        return res


# Largest BST Subtree

# BOundary of Binary Tree

class Solution(object):
    def boundaryOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root: return []
        if not root.left and not root.right: return [root.val]
        leftBoundary = self.getLeft(root)
        leaves = []
        self.getLeaves(root, leaves)
        rightBoundary = self.getRight(root)

        return leftBoundary + leaves + rightBoundary

    def getLeft(self, root):
        result = [root.val]
        root = root.left
        while root:
            if root.left or root.right:
                result.append(root.val)
            if root.left:
                root = root.left
            else:
                root = root.right
        return result

    def getLeaves(self, root, leaves):
        if not root: return
        if not root.left and not root.right:
            leaves.append(root.val)
            return
        self.getLeaves(root.left, leaves)
        self.getLeaves(root.right, leaves)

    def getRight(self, root):
        result = []
        root = root.right
        while root:
            if root.left or root.right:
                result.append(root.val)
            if root.right:
                root = root.right
            else:
                root = root.left
        result.reverse()
        return result

# Recover Binary Search Tree
'''
Two elements of a BST are swapped by mistake

Recover the tree without changing its structure

#A solution using O(n) space is pretty straight forward.
Could you devise a constant space solution?
'''
class Solution(object):
    def recoverTree(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        swap = []
        pre, cur = None, root
        stack = []
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.left

            cur = stack.pop()
            if pre and pre.val >= cur.val:
                if not swap:
                    swap.append(pre)
                    swap.append(cur)
                else:
                    swap[1] = cur
            pre = cur
            cur = cur.right
        if len(swap) == 2:
            swap[0].val, swap[1].val = swap[1].val, swap[0].val






# Lowest Common Ancestor of a binary Tree
'''
Given binary tree, find LCA of two given noes in the tree

allow a node to be a descendant of itself

'''
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        if root == p or root == q:
            return root

        # divide
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        # conquer
        if left != None and right != None:
            return root
        if left != None:
            return left
        else:
            return right

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """

        stack = [root]
        parent = {root: None}
        while p not in parent or q not in parent:
            node = stack.pop()
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)
        ancestors = set()
        while p:
            ancestors.add(p)
            p = parent[p]
        while q not in ancestors:
            q = parent[q]
        return q





# Count Complete Tree Nodes
'''
Given a complete binary tree, count the number of nodes

Complete BT: each level, except the last, is completely filled.
all the nodes in the last level are as far left as possible.

can have between 1 and 2^h nodes inclusive at the last level h
'''
class Solution():
    def countNodes(self, root):
        if not root:
            return 0
        if self.depth(root.left, True) == self.depth(root.right, False):
            return 2 ** (self.depth(root.left, True) + 1) - 1
        else:
            return self.countNodes(root.left) + self.countNodes(root.right) + 1

    def depth(self, root, isLeft):
        ans = 0
        while root:
            if isLeft:
                root = root.left
            else:
                root = root.right
            ans += 1
        return ans



class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        leftDepth = self.getDepth(root.left)
        rightDepth = self.getDepth(root.right)
        if leftDepth == rightDepth:
            return pow(2, leftDepth) + self.countNodes(root.right)
        else:
            return pow(2, rightDepth) + self.countNodes(root.left)

    def getDepth(self, root):
        if not root:
            return 0
        return 1 + self.getDepth(root.left)


class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        h = self.height(root)
        nodes = 0
        while root:
            if self.height(root.right) == h - 1:
                nodes += 2 ** h  # left half (2 ** h - 1) and the root (1)
                root = root.right
            else:
                nodes += 2 ** (h - 1)
                root = root.left
            h -= 1
        return nodes

    def height(self, root):
        return -1 if not root else 1 + self.height(root.left)



# Binary Tree Maximum Path Sum - HARD
'''
Find the maximum path sum

A path is defined as any sequence of of nodes from some starting
to any node int he tree. Path must have at least one node and does not need to go through the root

'''

class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.maxm = root.val
        self.solve(root)
        return self.maxm

    def solve(self, root):
        if not root: return 0
        le = self.solve(root.left)
        ri = self.solve(root.right)
        lmax = max(0, le) + max(0, ri) + root.val
        if lmax > self.maxm:
            self.maxm = lmax
        return root.val + max(0, le,ri)


class Solution(object):
    def __init__(self):
        self.path_sum = []

    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0

        self._max_path_sum(root)
        return max(self.path_sum)

    def _max_path_sum(self, root):
        if not root:
            return False
        left = self._max_path_sum(root.left)
        right = self._max_path_sum(root.right)

        if not left or left + root.val < root.val:
            left = 0
        if not right or right + root.val < root.val:
            right = 0

        path_sum = root.val + left +right
        self.path_sum.append(path_sum)
        return left + root.val if left > right else right + root.val





# Validate Binary Search Tree
'''

'''

# divide and conquer
import sys
class Solution:
    # @param {TreeNode} root
    # @return {boolean}
    def isValidBST(self, root):
        return self._isValidBST(root, -sys.maxint, sys.maxint)

    def _isValidBST(self, root, lb, ub):
        if not root:
            return True
        if root.val >= ub or root.val <= lb:
            return False
        return self._isValidBST(root.left, lb, root.val) and self._isValidBST(root.right, root.val, ub)


class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root is None:
            return True

        pre = None
        stack = []
        cur = root
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            if cur.val <= pre:
                return False
            pre = cur.val
            cur = cur.right
        return True




# 662 Maximum Width Of Binary Tree

# 129 Sum root to leaf NUmbers
    1
2       3
12 + 13 = 25
return 25

@Recursive


class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return self.rootSum(root, 0)

    def rootSum(self, root, preSum):
        if not root:
            return 0
        preSum = 10 * preSum + root.val
        if root.left == None and root.right == None:
            return preSum
        left = self.rootSum(root.left, preSum)
        right = self.rootSum(root.right, preSum)
        return left + right



@Iterative
class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        paths = []
        stack = [(root, str(root.val))]
        while stack:
            node, path = stack.pop()
            if not node.left and not node.right:
                paths.append(path)
            if node.left:
                stack.append((node.left, path + str(node.left.val)))
            if node.right:
                stack.append((node.right, path + str(node.right.val)))
        res = sum([int(path) for path in paths])
        return res

# 255 verify preorer sequence in binary search tree
'''
Kinda simulate the traversal, keeping a stack of nodes (just their values) of
 which we're still in the left subtree. If the next number is smaller than the
 last stack value, then we're still in the left subtree of all stack nodes, so
 just push the new one onto the stack. But before that, pop all smaller ancestor
  values, as we must now be in their right subtrees (or even further, in the right
   subtree of an ancestor). Also, use the popped values as a lower bound, since
   being in their right subtree means we must never come across a smaller number anymore.
'''

{10, 5, 1, 7, 40, 50}
@O(n) and O(n)
class Solution(object):
    def verifyPreorder(self, preorder):
        """
        :type preorder: List[int]
        :rtype: bool
        """
        low = -1
        stack = []
        for val in preorder:
            if val < low:
                return False
            while stack and val > stack[-1]:
                low = stack.pop()
            stack.append(val)
        return True

'''
we realize that the preorder array can be reused as the stack thus achieve O(1)
 extra space, since the scanned items of preorder array is always more than or equal to the length of the stack.
'''
@ O(n) an O(1)
    # stack = preorder[:i], reuse preorder as stack
class Solution(object):
    def verifyPreorder(self, preorder):
        i = -1 # 保持一个降序数列
        low = float('-inf')
        for p in preorder:
            if p < low: # 如果后面进来的小于左面最大的，则不成立。
                return False
            while i >= 0 and p > preorder[i]:
                low = preorder[i]
                i -= 1
            i += 1
            preorder[i] = p
        return True
# 536 construct binary tree from string

# 606 construct string from binary tree

class Solution:
    def tree2str(self, t):
        if t is None:
            return ''
        currVal = str(t.val)

        # case 1
        if t.left == None an t.right == None:
            return currVal
        #******* case 2: CANNOT omit
        if t.left == None and t.right != None:
            return currVal + '()' + '(' + self.tree2str(t.right) + ')'

        if t.right == None and t.left == None:
            return currVal + '(' + self.tree2str(t.left) + ')'

        return currVal + '(' + self.tree2str(t.left) + ')' + '(' + self.tree2str(t.right) + ')'
###############################################################################
                                '''Array Questions '''
###############################################################################

# Array Partition I
'''
给定一个长度为2n的整数数组，将数组分成n组，求每组数的最小值之和的最大值。

注意：

n是正整数，范围[1, 10000]
所有整数范围为[-10000, 10000]

'''
'''
Sorting based solution

For an optimized solution, begin with an example arr = [4,3,1,2]
Sort this array. arr = [1,2,3,4]
Now note that 1 needs to be a paired with a larger number. What is the number
 we would like to sacrifice? Clearly the smallest possible.
This gives the insight: sort and pair adjacent numbers.
Sorting takes Nlg(N) and space lg(N).
'''
class Solution(object):
    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        s_so_far = 0
        for i in range(0, len(nums)-1, 2):
            s_so_far += nums[i]
        return s_so_far

# Reshape the Matrix
'''
给定二维矩阵nums，将其转化为r行c列的新矩阵。若无法完成转化，返回原矩阵。

Input:
nums =
[[1,2],
 [3,4]]
r = 1, c = 4
Output:
[[1,2,3,4]]
Explanation:
The row-traversing of nums is [1,2,3,4]. The new reshaped matrix is a 1 * 4 matrix,
 fill it row by row by using the previous list.

 https://leetcode.com/problems/reshape-the-matrix/#/solution
'''


'''

'''
class Solution(object):
    def matrixReshape(self, nums, r, c):
        """
        :type nums: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        if r*c != len(nums)*len(nums[0]):
            return nums

        data = []
        for nn in nums:
            for n in nn:
                data.append(n)

        res = []
        index = 0
        for ri in range(r):
            item = []
            for ci in range(c):
                item.append(data[index])
                index += 1
            res.append(item)
        return res


'''
Time complexity : O(m*n). We traverse the entire matrix of size m*nm∗n once
 only. Here, mm and nn refers to the number of rows and columns in the given matrix.

Space complexity : O(m*n). The resultant matrix of size m*nm∗n is used.

'''
class Solution(object):
    def matrixReshape(self, nums, r, c):
        """
        :type nums: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        numRows = len(nums)
        numColumns = len(nums[0])

        if r * c != numRows * numColumns:
            return nums
        ret = []
        n = 0
        for i in range(r):
            ret += [[]]
            for j in range(c):
                ret[i] += [nums[n / numColumns][n % numColumns]]
                n += 1

        return ret
"""
The element nums[i][j]nums[i][j] of numsnums array is represented in the form of
a one dimensional array by using the index in the form: nums[n*i + j]nums[n∗i+j],
where mm is the number of columns in the given matrix. Looking at the same in the
 reverse order, while putting the elements in the elements in the resultant matrix,
  we can make use of a countcount variable which gets incremented for every element
  traversed as if we are putting the elements in a 1-D resultant array. But, to convert
   the countcount back into 2-D matrix indices with a column count of cc, we can obtain
    the indices as res[count/c][count\%c]res[count/c][count%c] where count/ccount/c is
     the row number and count\%ccount%c is the coloumn number. Thus, we can save the
      extra checking required at each step.
"""


class Solution(object):
    def matrixReshape(self, nums, r, c):
        """ass
        :type nums: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        h, w = len(nums), len(nums[0])
        if h * w != r * c: return nums
        ans = []
        for x in range(r):
            row = []
            for y in range(c):
                row.append(nums[(x * c + y) / w][(x * c + y) % w])
            ans.append(row)
        return ans


# Kth Largest Element in an array

class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        return self.quickSelect(nums, 0, len(nums) - 1, k)


    def quickSelect(self, nums, start, end, k):
        if start == end:
            return nums[start]

        i = start
        j = end
        pivot = nums[(i + j) / 2]

        while i <= j:
            while i <= j and nums[i] > pivot:
                i += 1
            while i <= j and nums[j] < pivot:
                j -= 1


            if i <= j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1

        # kth, k - 1 is a offset relative to start
        # kth is on the left
        if (start + k - 1 <= j):
            return self.quickSelect(nums, start, j, k)
        # kth is on the rihgt
        if start + k -1 >= i:
            return self.quickSelect(nums, i, end, k - (i - start))
        # kth is middle
        return nums[j + 1]



# Wiggle Sort
'''
given an unsorted array, reorder it in place such at
nums[0] <= nums[1] >= nums[2] <= nums[3]....
'''

class Solution(object):
    """
    @param {int[]} nums a list of integer
    @return nothing, modify nums in-place instead
    """
    def wiggleSort(self, nums):
        # Write your code here
        n = len(nums)
        for i in xrange(1, n):
            #                                             first number,
            if i % 2 == 1 and nums[i] < nums[i - 1] or i % 2 == 0 and nums[i] > nums[i - 1]:
                nums[i], nums[i - 1] = nums[i- 1], nums[i]


# Find all duoplicates in an array

#Find all the elements that appear twice in this array.
'''
O(1) space not including the input and output variables

The idea is we do a linear pass using the input array itself as a hash to store
 which numbers have been seen before. We do this by making elements at certain
 indexes negative. See the full explanation here

http://www.geeksforgeeks.org/find-duplicates-in-on-time-and-constant-extra-space/
'''
class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = []
        for x in nums:
            # -1 , zero-based indexing
            if nums[abs(x)-1] < 0:
                res.append(abs(x))
            else:
                nums[abs(x)-1] *= -1
        return res


class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        seen = set()
        res = []
        for num in nums:
            if num in seen:
                res.append(num)
            else:
                seen.add(num)
        return res



# Range Addition



# Max Consecutive Ones
'''
Input: [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s.
    The maximum number of consecutive 1s is 3.

'''

class Solution(object):
    def findMaxConsecutiveOnes(self, nums):

        curLength = 0
        maxLength = 0
        for n in nums:
            if n == 1:
                curLength += 1
                if curLength > maxLength:
                    maxLength = curLength
            else:
                curLength = 0

        return maxLength



# Lonely Pixel I
'''
给定一个包含字符'W'（白色）和'B'（黑色）的像素矩阵picture。

求所有同行同列有且仅有一个'B'像素的像素个数。
Input:
[['W', 'W', 'B'],
 ['W', 'B', 'W'],
 ['B', 'W', 'W']]

Output: 3
Explanation: All the three 'B's are black lonely pixels.
'''


'''
利用数组rows，cols分别记录某行、某列'B'像素的个数。

然后遍历一次picture即可。
'''
class Solution(object):
    def findLonelyPixel(self, picture):
        """
        :type picture: List[List[str]]
        :rtype: int
        """
        w, h = len(picture), len(picture[0])
        rows, cols = [0] * w, [0] * h
        for x in range(w):
            for y in range(h):
                if picture[x][y] == 'B':
                    rows[x] += 1
                    cols[y] += 1
        ans = 0
        for x in range(w):
            for y in range(h):
                if picture[x][y] == 'B':
                    if rows[x] == 1:
                        if cols[y] == 1:
                            ans += 1
        return ans

# Shortest Word Distance


# Find all Numbers Disappered in an array
'''
interger where  1<= a[i] < n, n is size of array
'''
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        nums = [0] + nums
        for i in range(len(nums)):
            index = abs(nums[i])
            nums[index] = -abs(nums[index])

        return [i for i in range(len(nums)) if nums[i] > 0]


class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        for num in nums:
            idx = abs(num) - 1
            nums[idx] = -abs(nums[idx])

        return [i+1 for i in range(len(nums)) if nums[i] > 0]


# Teemo Attacking
'''
给定一组递增的时间起点timeSeries，以及一个时间段duration，timeSeries中的每个起点st对应的终点ed = st + duration。

求各时间段覆盖的时间总长度。

'''

#一趟遍历即可，

class Solution(object):
    def findPoisonedDuration(self, timeSeries, duration):
        """
        :type timeSeries: List[int]
        :type duration: int
        :rtype: int
        """
        now = ans = 0
        for st in timeSeries:
            ans += min(duration, st + duration - now)
            now = st + duration
        return ans




# Shortest Word Distance III


# Array nesting
'''

'''


# Move zeroes
'''
给定一个数组nums，编写函数将数组内所有0元素移至数组末尾，并保持非0元素相对顺序不变。

例如，给定nums = [0, 1, 0, 3, 12]，调用函数完毕后， nums应该是 [1, 3, 12, 0, 0]。

注意：

你应该“就地”完成此操作，不要复制数组。
最小化操作总数。

'''


'''
算法步骤：

使用两个"指针"x和y，初始令y = 0

利用x遍历数组nums：

若nums[x]非0，则交换nums[x]与nums[y]，并令y+1
'''
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        tail = 0
        for x in range(len(nums)):
            if nums[x] != 0 :
                nums[x], nums[tail] = nums[tail], nums[x]
                tail += 1




# Product of Array Except self
'''

'''

'''
首先想到的思路是计算全部数字的乘积，然后分别除以num数组中的每一个数（需要排除数字0）。然而，题目要求不能使用除法。

下面的解法非常巧妙，参考LeetCode Dicuss

链接地址：https://leetcode.com/discuss/46104/simple-java-solution-in-o-n-without-extra-space

由于output[i] = (x0 * x1 * ... * xi-1) * (xi+1 * .... * xn-1)

因此执行两趟循环：

第一趟正向遍历数组，计算x0 ~ xi-1的乘积

第二趟反向遍历数组，计算xi+1 ~ xn-1的乘积
'''
class Solution:
    # @param {integer[]} nums
    # @return {integer[]}
    def productExceptSelf(self, nums):
        size = len(nums)
        output = [1] * size
        left = 1
        for x in range(size - 1):
            left *= nums[x]
            output[x + 1] *= left
        right = 1
        for x in range(size - 1, 0, -1):
            right *= nums[x]
            output[x - 1] *= right
        return output


# Two Sum II - input array is sorted
'''
Given an array of integers that is already sorted in ascending order,
 find two numbers such that they add up to a specific target number.
Please note that your returned answers (both index1 and index2) are not zero-based.

'''

# Binary Serach

# O(NlogN)
def twoSum(self, nums, target):
    if nums == None or len(numbers) == 0:
        return []

    for i in range(len(nums)):
        l, r = i + 1, len(nums) - 1
        while l <= r:
            mid = l + (l - r) /2
            if nums[mid] == target - numbers[i]:
                return [i + 1, mid + 1]
            elif nums[mid] < target - nums[i]:
                l = mid + 1
            else:
                r = mid - 1
    return []




# TWo pointer solution
#O(N)
def twoSum(numbers, target):
    if nums == None:
        return []

    l, r = 0, len(nums) - 1
    while l < r:
        sum = nums[l] + nums[r]
        if sum == target:
            return  [l + 1, r + 1] # not zero based
        elif sum < target:
            l += 1
        else:
            r -= 1
    return []



class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        l, r = 0, len(numbers)-1
        while l < r:
            s = numbers[l] + numbers[r]
            if s == target:
                return [l+1, r+1]
            elif s < target:
                l += 1
            else:
                r -= 1

# dictionary
def twoSum2(self, numbers, target):
    dic = {}
    for i, num in enumerate(numbers):
        if target-num in dic:
            return [dic[target-num]+1, i+1]
        dic[num] = i

# binary search
def twoSum(self, numbers, target):
    for i in xrange(len(numbers)):
        l, r = i+1, len(numbers)-1
        tmp = target - numbers[i]
        while l <= r:
            mid = l + (r-l)//2
            if numbers[mid] == tmp:
                return [i+1, mid+1]
            elif numbers[mid] < tmp:

                l = mid+1
            else:
                r = mid-1


# Best time to Buy and Sell Stock II
'''
FInd max profit, may complete as many as transactions as you like
ie. buy one and sell one share multiple times a day.
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/#/solution
'''


'''
directly keep on adding the difference between the consecutive numbers of the
array if the second number is larger than the first one, and at the total sum we obtain will be the maximum profit.
'''
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        total = 0
        for i in range(len(prices)-1):
            if prices[i+1] > prices[i]:
                total+= prices[i+1] - prices[i]

        return total

# Majority Element
'''
majority is the lement more than half
'''
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        dic = {}
        for num in nums:
            if num not in dic:
                dic[num] = 1
            if dic[num] > len(nums)//2:
                return num
            else:
                dic[num] += 1



# Contains Duplicates
'''
 return true if any value appears at least twice in the array,
  and it should return false if every element is distinct.
'''
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        dic = {}
        for i in nums:
            if i in dic:
                dic[i] += 1
            else:
                dic[i] = 1

        for i in nums:
            if dic[i] >= 2:
                return True

        return False

class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        return len(nums) != len(set(nums))


# maximum product of three numbers

def maximumProduct(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    max1 = -1000
    max2 = -1000
    max3 = -1000
    min1 = 1000
    min2 = 1000
    for n in nums:
        if n >max1:
            max1,max2,max3 = n,max1,max2
        elif n >max2:
            max2,max3 = n,max2
        elif n >max3:
            max3 = n
        if n < min1:
            min1,min2 = n,min1
        elif n < min2:
            min2 = n
    #print(max1, max2, max3, min1, min2)
    return max(max1*max2*max3, max1*min1*min2)


# Combination Sum  III
'''
find all possible combinations of K numbers that add up to a number n. Given only
numbers from 1 to 9 can be used. Each combination should be unique
寻找所有满足k个数之和等于n的组合，只允许使用数字1-9，并且每一种组合中的数字应该是唯一的。

确保组合中的数字以递增顺序排列。
'''
class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        if n > sum([i for i in range(1, 11)]):
            return []

        res = []
        self.sum_help(k, n, 1, [], res)
        return res


    def sum_help(self, k, n, curr, arr, res):
        if len(arr) == k:
            if sum(arr) == n:
                res.append(list(arr))
            return

        if len(arr) > k or curr > 9:
            return

        for i in range(curr, 10):
            arr.append(i)
            self.sum_help(k, n, i + 1, arr, res)
            arr.pop()


class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        if n < 0 or k>9 or k<1:
            return
        nums = [1,2,3,4,5,6,7,8,9]
        paths = []
        self.dfs(paths, [], nums, k, n, 0)
        return paths

    def dfs(self, paths, curr_path, nums,  k, n, start):
        if n == 0 and len(curr_path) == k:
            paths.append(curr_path[:])
        else:
            for i in range(start, len(nums)):

                self.dfs(paths, curr_path + [nums[i]], nums,  k, n - nums[i], i + 1)


# Missing Number
'''
n distinct numbers from 0, 1, 2, ...n find the one that is missing from the aaray

For example,
Given nums = [0, 1, 3] return 2.

our algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?
'''

#解法I：等差数列前n项和 - 数组之和


class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        return n * (n + 1) / 2 - sum(nums)


class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        # method1, use sum
        n = len(nums)
        index_sum = (0 + n - 1) * n /2 + n
        for i in range(n):
            index_sum = index_sum - nums[i]
        return index_sum

        # method2, use XOR
        val = len(nums)
        for i in range(len(nums)):
            val = val ^ nums[i] ^ i
        return val


# Find the Duplicate Number
'''

'''

"""
根据鸽笼原理，给定n + 1个范围[1, n]的整数，其中一定存在数字出现至少两次。

假设枚举的数字为 n / 2：

遍历数组，若数组中不大于n / 2的数字个数超过n / 2，则可以确定[1, n /2]范围内一定有解，

否则可以确定解落在(n / 2, n]范围内。
"""
#O(n * log n)
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        low, high = 1, len(nums) - 1
        while low <= high:
            mid = (low + high) >> 1
            cnt = sum(x <= mid for x in nums)
            if cnt > mid:
                high = mid - 1
            else:
                low = mid + 1
        return low


#O(n)
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        slow = nums[0]
        fast = nums[nums[0]]
        while (slow != fast):
            slow = nums[slow]
            fast = nums[nums[fast]]

        fast = 0;
        while (fast != slow):
            fast = nums[fast]
            slow = nums[slow]

        return slow


# Lonely Pixel II

'''

'''
# Task Scheduler
'''
CPU执行任务调度，任务用字符数组tasks给出，每两个相同任务之间必须执行n个不同的其他任务或者空闲。

求最优调度策略下的CPU运行周期数。
'''


# 3Sum Smaller

'''
return  the number of triplets that nums[i] + nums[j] + nums[k] < target

different from 3sum:
if temp < target:
    res += r - l
    l += 1

'''
class Solution(object):
    def threeSumSmaller(self, nums, target):
        if nums == None or len(nums) < 3:
            return 0

        nums.sort()
        res = 0

        for i in range(len(nums)):
            l, r = i + 1, len(nums) - 1

            while l < r:
                tmp = nums[i] +nums[l] + nums[r]

                # accumulate the count
                if temp < target:
                    res += r - l
                    l += 1
                else:
                    r -= 1
        return res




# Best time to buy and sell stock
'''
#https://leetcode.com/articles/best-time-buy-and-sell-stock/

Approach -1: Brute Force


approach-2: one pass
TC: O(n)
SC: O(1)

Goal: find peak and valley in the given graph. Largest peak following the
smallest valley. Maintain two variables,
 minprice and maxprofit corresponding to smallest valley and max profit(the Maximumdifference between selling price
 and minprice)

'''

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        low = float('inf')
        profit = 0
        for i in prices:
            profit = max(profit, i-low)
            low = min(low, i)
        return profit

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices: return 0
        lowest = prices[0]
        res = 0
        for i in range(1, len(prices)):
            lowest = min(prices[i], lowest)
            res = max(res, prices[i]-lowest)
        return res


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """

        if len(prices) <= 1:
            return 0

        lo,hi = prices[0],prices[0]
        profit = 0

        for i in xrange(len(prices)):
            if prices[i] < lo:
                lo = prices[i]
            else:
                profit = max(profit,prices[i]-lo)


        return profit

# Unique Paths

# Subarray Sum Equals K
'''
给定整数数组nums和整数k，寻找和等于k的连续子数组的个数。

利用字典cnt统计前N项和出现的个数

遍历数组nums：

    在cnt中将sums的计数+1

    累加前N项和为sums

    将cnt[sums - k]累加至答案https://leetcode.com/articles/subarray-sum-equals-k/
'''

class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        count = {0:1}
        out=0
        runsum=0
        for x in nums:
            runsum +=x
            if runsum-k in count:
             out +=count[runsum-k]
            if runsum in count:
             count[runsum] +=1
            else:
                count[runsum]=1
        return out


class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        d = {0: 1}
        count, cur_sum = 0, 0
        for x in nums:
            cur_sum += x
            if cur_sum - k in d:
                count += d[cur_sum-k]
            if cur_sum in d:
                d[cur_sum] += 1
            else:
                d[cur_sum] = 1
        return count

class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        ans = sums = 0
        cnt = collections.Counter()
        for num in nums:
            cnt[sums] += 1
            sums += num
            ans += cnt[sums - k]
        return ans
#78	Subsets

#611 Valid Triangle Number
'''
Given array , to cunt the number of triplets chosen from the array
that can make triangles if take them as side lenghths.
Input: [2,2,3,4]
Output: 3
Explanation:
Valid combinations are:
2,3,4 (using the first 2)
2,3,4 (using the second 2)
2,2,3
https://leetcode.com/problems/valid-triangle-number/#/solution
'''
"""
Time complexity : O(n^2). Loop of kk and jj will be executed O(n^2) times in total, because, we do not reinitialize the value of kk for a new value of jj chosen(for the same ii). Thus the complexity will be O(n^2+n^2)=O(n^2).

Space complexity : O(logn). Sorting takes O(logn) space.
"""
class Solution(object):
    def triangleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        nums = nums[::-1]
        sol = 0
        for i in range(len(nums) - 2):
            j = i + 1
            k = len(nums) - 1
            while j < k:
                diff = nums[i] - nums[j]
                while nums[k] <= diff and k > j:
                    k -= 1
                sol += (k - j)
                j += 1
        return sol




#
# 153	Find Minimum in Rotated Sorted Array	39.7%	Medium
'''
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

Find the minimum element.
'''
class Solution:
    # @param nums: a rotated sorted array
    # @return: the minimum number in the array
    def findMin(self, nums):
        if len(nums) == 0 or nums is None:
            return 0

        start, end = 0, len(nums) - 1
        target = nums[-1]

        while start + 1 < end:
            mid = (start + end) / 2
            # if mid <= target, on second rising segement, smaller elements
            # exist on the left of mid, move end to mid
            if nums[mid] <= target:
                end = mid
            else: # if mid > end, move start move to mid
                start = mid
        return min(nums[start], nums[end])



# Search Insert Position
'''
KEY: find the first position >= target
'''
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # find the first position >= target
        if not nums or len(nums) == 0:
            return 0
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start)/2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                start = mid
            else:
                end = mid
        if nums[start] >= target:
            return start
        elif nums[end] >= target:
            return end
        else:
            return end + 1


# First Bad Version
'''
Versin NUmber : 1 - n
'''
  def findFirstBadVersion(self, n):
        start, end = 1, n
        while start + 1 < end:
            mid = (start + end) / 2
            if SVNRepo.isBadVersion(mid):
                end = mid
            else:
                start = mid

        if SVNRepo.isBadVersion(start):
            return start
        return end


# Count Of Smaller Number
'''

for each query, return the number of element in that array that are smaller than
the given integer

Array : [1, 2, 7, 8, 5]
Query: [1, 8, 5]
return [0, 4, 2]

'''

# Sort and binary search

class Solution:
    def countOfSmallerNumber(self, A, queries):
        A.sort()
        res = []

        for q in queries:
            l, r = 0 len(A)
            while l < r:
                mid = l + (r - l) / 2
                # the first pos that is greater than q
                if A[mid] >= q:
                    r = mid
                else:
                    l = mid + 1
            res.append(r)

        return res


class Solution:
    """
    @param A: A list of integer
    @return: The number of element in the array that
             are smaller that the given integer
    """
    def countOfSmallerNumber(self, A, queries):
        A = sorted(A)

        results = []
        for q in queries:
            results.append(self.countSmaller(A, q))
        return results

    def countSmaller(self, A, q):
        # find the first number in A >= q
        if len(A) == 0 or A[-1] < q:
            return len(A)

        start, end = 0, len(A) - 1
        while start + 1 < end:
            mid = (start + end) / 2
            if A[mid] < q:
                start = mid
            else:
                end = mid
        if A[start] >= q:
            return start
        if A[end] >= q:
            return end
        return end + 1





# 53	Maximum Subarray	39.5%	Easy
'''
Find the contiguous subarray within an array (containing at least one number) which has the largest sum.

For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
the contiguous subarray [4,-1,2,1] has the largest sum = 6.
'''
class Solution(object):
    def maxSubArray(self, A):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not A:
            return 0

        curSum = maxSum = A[0]
        for num in A[1:]:
            curSum = max(num, curSum + num)
            maxSum = max(maxSum, curSum)

        return maxSum






#643	Maximum Average Subarray I	39.3%	Easy
'''
给定包含n个整数的数组，寻找长度为k的连续子数组的平均值的最大值
https://leetcode.com/articles/maximum-average-subarray/
'''
class Solution(object):
    def findMaxAverage(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: float
        """
        ans = None
        sums = 0
        for x in range(len(nums)):
            sums += nums[x]
            if x >= k:
                sums -= nums[x - k]
            if x >= k - 1:
                ans = max(ans, 1.0 * sums / k)
        return ans


#59	Spiral Matrix II	39.4%	Medium
'''
SPIRAL INSIDE OUT, square matrix fileld with 1 - n^2
'''

def generateMatrix(self, n):
    A, lo = [], n*n+1
    while lo > 1:
        lo, hi = lo - len(A), lo
        A = [range(lo, hi)] + zip(*A[::-1])
    return A


class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        A = [[0] * n for _ in range(n)]
        i, j, di, dj = 0, 0, 0, 1
        for k in xrange(n*n):
            A[i][j] = k + 1
            if A[(i+di)%n][(j+dj)%n]:
                di, dj = dj, -di
            i += di
            j += dj
        return A

class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        if n == 1:
            return [[1]]

        num_layers = n - 1
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        count = 1

        for layer in range(num_layers):
            for i in range(layer, n - layer):
                matrix[layer][i] = count
                count += 1

            for i in range(1 + layer, n - layer):
                matrix[i][-1 - layer] = count
                count += 1

            for i in range(n - 2 - layer, -1 + layer, -1):
                matrix[-1 - layer][i] = count
                count += 1

            for i in range(n - 2 - layer, layer, -1):
                matrix[i][0 + layer] = count
                count += 1

        return matrix

#562	Longest Line of Consecutive One in Matrix 	39.0%	Medium
'''
给定01矩阵M，计算矩阵中一条线上连续1的最大长度。一条线可以为横向、纵向、主对角线、反对角线。

提示：给定矩阵元素个数不超过10,000

解题思路：
动态规划（Dynamic Programming）

分表用二维数组h[x][y], v[x][y], d[x][y], a[x][y]表示以元素M[x][y]结尾，横向、纵向、主对角线、反对角线连续1的最大长度

状态转移方程如下：

h[x][y] = M[x][y] * (h[x - 1][y]  + 1)

v[x][y] = M[x][y] * (v[x][y - 1]  + 1)

d[x][y] = M[x][y] * (d[x - 1][y - 1]  + 1)

a[x][y] = M[x][y] * (a[x + 1][y - 1]  + 1)
'''
class Solution(object):
    def longestLine(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        h, w = len(M), len(M) and len(M[0]) or 0
        ans = 0

        #horizontal & diagonal
        diag = [[0] * w for r in range(h)]
        for x in range(h):
            cnt = 0
            for y in range(w):
                cnt = M[x][y] * (cnt + 1)
                diag[x][y] = M[x][y]
                if x > 0 and y > 0 and M[x][y] and diag[x - 1][y - 1]:
                    diag[x][y] += diag[x - 1][y - 1]
                ans = max(ans, cnt, diag[x][y])

        #vertical & anti-diagonal
        adiag = [[0] * w for r in range(h)]
        for x in range(w):
            cnt = 0
            for y in range(h):
                cnt = M[y][x] * (cnt + 1)
                adiag[y][x] = M[y][x]
                if y < h - 1 and x > 0 and M[y][x] and adiag[y + 1][x - 1]:
                    adiag[y][x] += adiag[y + 1][x - 1]
                ans = max(ans, cnt, adiag[y][x])

        return ans



#380	Insert Delete GetRandom O(1)	39.0%	Medium
'''
design all the following ops in average O(1) time for SET

insert(val): Inserts an item val to the set if not already present.
remove(val): Removes an item val from the set if present.
getRandom: Returns a random element from current set of elements. Each element must have the same probability of being returned.
'''

"""
哈希表 + 数组 （HashMap + Array）

利用数组存储元素，利用哈希表维护元素在数组中的下标

由于哈希表的新增/删除操作是O(1)，而数组的随机访问操作开销也是O(1)，因此满足题设要求

记数组为dataList，哈希表为dataMap

insert(val): 将val添至dataList末尾，并在dataMap中保存val的下标idx

remove(val): 记val的下标为idx，dataList末尾元素为tail，弹出tail并将其替换至idx处，在dataMap中更新tail的下标为idx，最后从dataMap中移除val

getRandom: 从dataList中随机选取元素返回
"""
import random
class RandomizedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dataMap = {}
        self.dataList = []

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.dataMap:
            return False
        self.dataMap[val] = len(self.dataList)
        self.dataList.append(val)
        return True

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.dataMap:
            return False
        idx = self.dataMap[val]
        tail = self.dataList.pop()
        if idx < len(self.dataList):
            self.dataList[idx] = tail
            self.dataMap[tail] = idx
        del self.dataMap[val]
        return True

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        return random.choice(self.dataList)

#27	Remove Element	38.8%	Easy
'''

Given a array and a value, remove all instances of that value in place and return new length

MUST in place with constant memory
Given input array nums = [3,2,2,3], val = 3

Your function should return length = 2, with the first two elements of nums being 2.
'''
#Two Pointers
@TC: O(n), array has total of n elements, both i and j traverse at most 2n steps
@ SC: O(1)
def removeElement(nums, target):
    j = 0
    for i in range(len(nums)):
        if nums[i] != target:
            nums[j] = nums[i]
            j += 1
    return i

# Two Pointers - when elemensts to remove are rare

@TC: O(n), both i and n traverse at most n steps. The number of assginment operation is euqal to the number
of elements to remove. More efficient is elements are rareself.
@SC; O(1)


def removeElementRare(nums, target):
    i = 0
    n = len(nums)
    while i < n:
        if nums[i] == target:
            nums[i] = nums[n-1]
            n -= 1 # reduce array size by 1
        else:
            i += 1

    return n

48	Rotate Image	38.4%	Medium
66	Plus One	38.4%	Easy
64	Minimum Path Sum	38.3%	Medium
#118	Pascal's Triangle	38.3%	Easy
# Pascal's Triangle
'''
Given numRows, generate the first numRows of Pascal's triangle

For exameple, given numRows = 5

[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
'''
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if not numRows:
            return []

        if numRows == 1:
            return [[1]]

        result = [[1]]
        for i in range(1, numRows):
            result.append([1])
            for j in range(1, i):
                result[i].append(result[i - 1][j - 1] + result[i - 1][j])
            result[i].append(1)

        return result


39	Combination Sum	38.2%	Medium
75	Sort Colors	37.8%	Medium
#162	Find Peak Element	37.2%	Medium
'''

'''
class Solution:
    #@param A: An integers list.
    #@return: return any of peek positions.
    def findPeak(self, A):
        # write your code here
        start, end = 1, len(A) - 2
        while start + 1 <  end:
            mid = (start + end) / 2
            if A[mid] < A[mid - 1]:
                end = mid
            elif A[mid] < A[mid + 1]:
                start = mid
            else:
                end = mid

        if A[start] < A[end]:
            return end
        else:
            return start

# Sqrt(x)
'''return the square root of x'''
class Solution:

    def sqrt(self, x):
        start, end = 1, x
        while start + 1 < end:
            mid = (start + end) / 2
            if mid * mid == x:
                return mid
            elif mid * mid < x:
                start = mid
            else:
                end = mid
        if end * end <= x:
            return end
        return start


# Wood Cut
'''
pieces of wood with length[123, 345, 231], cut them into k pieces of same length

'''
class Solution:
    def woodCut(self, L, k):
        if sum(L) < k:
            return 0

        maxLen = max(L)
        # from 1 to max len
        start, end = 1, maxLen
        while start + 1 < end:
            mid = (start + end) / 2
            pieces = sum([l / mid for l in L])
            if pieces >= k:
                start = mid
            else:
                end = mid

        if sum([l / end for l in L]) >= k:
            return end
        return start

154	Find Minimum in Rotated Sorted Array II	37.0%	Hard
289	Game of Life	36.8%	Medium
128	Longest Consecutive Sequence	36.6%	Hard
#119	Pascal's Triangle II	36.6%	Easy
'''
Given an index k, return the kth row of the pascal's Triangle

given k return [1, 3, 3, 1]

NOTE: optimize using only O(k) extra space?

'''
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        if rowIndex < 0:
            return []
        row = [0] * (rowIndex + 1)
        row[0] = 1
        for i in xrange(1, rowIndex + 1):
            for j in xrange(i, 0, -1):
                row[j] = row[j] + row[j - 1]
        return row


11	Container With Most Water	36.5%	Medium
'''

'''
def maxArea(self, height):
    l, r = 0, len(height) - 1
    res = 0

    while l < r:
        res = max(res, min(height[l], height[r]) * (r - l))
        if height[l] < height[r]:
            l += 1
        else:
            r -= 1
    return res
42	Trapping Rain Water	36.5%	Hard
#Remove Duplicates from Sorted Array II	35.8%	Medium
'''
allowed at most twice in result

'''

def removeDuplicatesII(self, nums):
    if nums == None:
        return len(nums)

    tail = 2
    for i in range(2, len(nums)):
        if nums[i] != nums[tail - 1] or nums[i] != nums[tail]:
            nums[tail] = nums[ni]
            tail += 1

    return tail

73	Set Matrix Zeroes	35.8%	Medium
90	Subsets II	35.8%	Medium
#26	Remove Duplicates from Sorted Array	35.5%	Easy
'''
given [1, 1, 2], retrun 2. with the nums being 1 an 2
'''

def removeDuplicates(self, nums):
    if nums == None:
        return len(nums)
    tail = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[tail - 1]:
            nums[tail] = nums[i]
            tail += 1
    return tail  # return tail's length

# Partition Array - Lintcode
'''
given nums an an int k, smaller than k and bigger than k
return the Partitioning index, first nums[i] >= k

'''
def partitionArray(self, nums, k):
    if nums == None:
        return 0
    for i in range(len(nums)):
        # move the ones smaller than k
        if nums[i] < k:
            nums[tail], nums[i] = nums[i], nums[tail]
            tail += 1
    return tail

# Partition Array By Odd and Even
'''
Parition an integer array into odd number first then even number seonc

'''
def partitionArrayByOddAndEven(self, nums):

    tail = 0
    for i in range(len(nums)):
        if nums[i] % 2 == 1:
            nums[tail], nums[i] = nums[i], nums[tail]
            tail += 1

# Sort Letters by Case
'''
A stirng only containing letters, sort it by lower case first
and upper case second

EX:

'abAcD', a reasonable answer is 'acbA'. Not necessarily keep the orginal
orer of lower case or uppper case.

DO IT in-place and in one pass
'''

def sortLetters(self, chars):
    tail = 0
    for i in range(len(chars)):
        # if lower cases, put in front
        for ord(chars[i]) >= ord('a' and ord(chars[i])) <= ord('z'):
            chars[tail], chars[i] = chars[i], chars[tail]
            tail += 1

277	Find the Celebrity 	35.2%	Medium
74	Search a 2D Matrix	35.1%	Medium
'''
'''
class Solution:
    def searchMatrix(self, matrix, target):
        m = len(matrix)
        n = len(matrix[0])

        l, r = m * n -1

        while l <= r:
            mid = l + (r - l) / 2
            # n is the number of column
            row = mid / n
            col = mid % n

            if matrix[row][col] == target:
                return True
            elif matrix[row][col] < target:
                l = mid + 1
            else:
                r = mid - 1
        return False

# Search a 2D matrix II
'''
each row is sorted, each column is sorted
'''

def searchMatrix(self, matrix, target):
    m = len(matrix)
    n = len(matrix[0])

    i, j = 0, n - 1
    while i < m and j >= 0:
        if matrix[i][j] == target:
            return True
        elif matrix[i][j] < target:
            i += 1
        else: # matrix[i][j] > target:
            j -= 1

    return False


# H-Index


# H-Index II

def hIndex(self, citations):
    n = len(citations)
    l, r = 0, n - 1

    while l <= r:
        mid = l + (r -l) / 2

        if citations[mid] == n - mid:
            return n - mid
        elif citations[mid] < n - mid:
            l = mid + 1
        else:
            r = mid - 1
    return n - l


548	Split Array with Equal Sum 	34.0%	Medium
1	Two Sum	34.0%	Easy
120	Triangle	33.6%	Medium
40	Combination Sum II	33.4%	Medium
624	Maximum Distance in Arrays 	33.0%	Easy
81	Search in Rotated Sorted Array II	32.8%	Medium
219	Contains Duplicate II	32.2%	Easy
33	Search in Rotated Sorted Array	32.1%	Medium
105	Construct Binary Tree from Preorder and Inorder Traversal	32.0%	Medium
106	Construct Binary Tree from Inorder and Postorder Traversal	32.0%	Medium
#88	Merge Sorted Array	31.9%	Easy
'''
given two sorted arrays nums1 an nums2, merge nums2 into nums1

# A has m elements, but size of A is m + n
# B has n elements

# has to be two ascending ordered
'''

def merge(self, num1, m, num2, n):
    i = m -1
    j = n - 1
    k = m + n - 1
    while k >= 0 and i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            #senario1: move elements at front of A to the back of A
            nums1[k] = nums1[i]
            i -= 1
        else:
            #senario2. two lists in ascending
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
    #senario1 continued: copy B to the front of A
    if j >= 0:
        nums1[:j+1] = nums2[:j+1]


# Merge Soted List(A + B => C)
##VARIATION: Merge two given sorted integer array A and B into a new sorted integer array.
# TC: O(n1 + n2)

def mergeSortedArray(Aa, b):

	i = 0
	j = 0
	c =  []
	while i < len(a) and j < len(b):
		if a[i] < b[j]:
			c.append(a[i])
			i += 1
		else:
			c.append(b[j])
			j += 1

	if i < len(a):
		while i < len(a):
			c.append(a[i])
			i += 1
	if j < len(b):
		while j < len(b):
			c.append(b[j])
			j += 1
	return c

# The Smallest Difference
'''
given two arrays, A and B. Find an element in A and B to make the
smallest difference

A =[3, 6, 7, 4]
B = [2, 8, 9, 3]
return 0 (3-3)
'''
class Solution:
    def smallestDifference(self, A, B):
        A.sort()
        B.sort()

        i = j = 0
        diff = 2147483647

        while i < len(A) an j < len(B):
            if A[i] > B[j]:
                diff = min(A[i] - B[j], diff)
                # because i > j, so increment j to get closer to i
                j += 1
            else:
                diff = min(B[j] - A[i], diff)
                i += 1
        return diff



63	Unique Paths II	31.6%	Medium
34	Search for a Range	31.3%	Medium
#16	3Sum Closest	31.0%	Medium
'''
return the sum of the three integers closest to a given number
'''
class Solution():
    def threeSumClosest(self, nums, target):
        if nums == None or len(nums) < 3:
            return 0

        nums.sort()
        res = nums[0] + nums[1] + nums[2]

        for i in range(len(nums)):
            l, r = i + 1, len(nums) - 1

            while l < r:
                tmp = nums[i] + nums[l] + nums[r]

                if abs(res - target) > abs(tmp - target):
                    res = tmp

                if tmp == target:
                    return tmp
                elif tmp < target:
                    l += 1
                else:
                    r -= 1
        return res


#209	Minimum Size Subarray Sum	30.3%	Medium
'''
给定一个包含n个正整数的数组和一个正整数s，找出其满足和sum ≥ s的子数组的最小长度。如果不存在这样的子数组，返回0

例如，给定数组 [2,3,1,2,4,3]与s = 7，
子数组[4,3]具有满足题设条件的最小长度。

进一步练习：
如果你已经找到了O(n)的解法，尝试使用时间复杂度为O(n log n)的解法完成此题
解题思路：
O(n)解法：滑动窗口法，使用两个下标start和end标识窗口（子数组）的左右边界

O(nlogn)解法：二分枚举答案，每次判断的时间复杂度为O(n)

'''
# TWO POINTERS
def minSubArrayLen(nums, s):
    sum = 0
    res = float('inf')
    head = 0

    for i in range(len(nums)):
        sum += nums[i]

        # inside while loop, - head to see if the sum still >= 7 to get shorter
        while sum >= s:
            res = min(res, i - head + 1)
            sum -= nums[head]
            head += 1

    return res if res <= len(nums) else 0


# Binary Search
# O(nlogn)
class Solution:
    def minSubArrayLen(self, s, nums):
        if nums == None:
            return 0

        n = len(nums)
        res = float('inf')
        sumList = [0] * (n + 1)

        for i in range(1, n+1):
            sumList[i] = nums[i - 1] + sumList[i - 1]

        for i in range(n + 1):
            l, r = i + 1, n + 1
            while l < r:
                mid = l + (r - l) / 2
                if sumList[mid] >= sumList[i] + s:
                    r = mid
                else:
                    l = mid + 1
            if r < n + 1:
                res = min(res, r - i)

        return res if res <= len(nums) else 0





# Subarray Sum II
'''
Given [1, 2, 3, 4], and interval = [1, 3]. return 4

[0, 0], [0, 1], [1, 1], [2, 2]
  1,     1 + 2,    2,     3
'''
#[0, 1, 3, 6, 10]
#10 - 3 = 7 ([3, 4])

class Solution:
    def subarraySumII(self, A, start, end):
        if A is None:
            return 0

        res = 0
        n = len(A)
        S = [0] * (n + 1)
        for i in range(n):
            S[i + 1] = S[i] + A[i]

        for i in range(n):
            for j in range(i + 1, n + 1):
                iff = S[j] - S[i]
                if diff >= start and diff <= end:
                    res += 1
        return res


# Longest Substring without repeating chars
"""abcabcbb is 'abc', bbbbb is 'b', with length 1"""

def lengthOfLongestSubstring(self, s):
    if s == None:
        return 0
    head = 0
    dic = dict()
    res = 0

    for i in range(len(s)):
        # conition check false, then move head
        if s[i] in dict and head <= dic[s[i]]:
            head = dic[s[i]] + 1

        # condition met
        # put index of every digit into dic
        dic[s[i]] = i
        res = max(res, i - head + 1)

    return res


class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        start = maxLength = 0
        usedChar = {}

        for i in range(len(s)):
            if s[i] in usedChar and start <= usedChar[s[i]]:
                start = usedChar[s[i]] + 1
            else:
                maxLength = max(maxLength, i - start + 1)

            usedChar[s[i]] = i

        return maxLength

# Longest Substring with at Most two Distinct characters
'''
find the length of the longest substring T that contains at most 2 istinct chars

'''

# two pointers - window
def lengthOfLongestSubstringTwoString(self, s):
    if s is None:
        return 0

    res = 0
    head = 0
    dic = colelctions.defaultdict(int)

    for i in range(len(s)):
        dic[s[i]] += 1

        # condition broken, move window's head until condiion is met
        while len(dic) > 2:
            dic[s[head]] -= 1
                if dic[s[head]] == 0:
                    del dic[s[head]]
                head += 1
        res = max(res, i -head + 1)

    return res

# Minimum window Substring
'''
Given string S and T, find the min widow in S which will all chars in T in O(n)

S = 'ADDBECODEBANC'
T = 'ABC'

Min winow is 'BANC'

'''
class Solution:
    def minWinow(self, s, t):
        counter = collections.Counter(t)
        window = []
        res = ''
        # s string's value and their indices
        dic = collections.defaultdict(list)

        for i, c in filter(lambda x: x[1] in t, enumerate(s)):
            dic[c].append(i)
            window.append(i)

            # not satisfies condition
            if len(dic[c]) > counter[c]:
                window.remove(dic[c].pop(0))


            if len(window) == len(t) and (res == '' or window[-1] - window[0] < len(res)):
                res = s[window[0]:window[-1]+1]

        return res

window TEMPLATE:

for():
    elements appended to window

    while(or if) condition not met:
        shift window

    if condition met:
        update result



# Substring with concatenation of all words
'''
Given a string s and a list of words of the same lenght.
Find all starting indices of substring in s that is a concatenation of each word in words exactly once


EX :

s : 'barfoothefoobarman'
words : ['foo', 'bar']
return [0, 9]

'''
class Solution:
    def findSubstring(self, s, words):



        counter = collections.Counter(words)
        res = []
        x = len(words[0])

        for n in range(x):
            dic = colelctions.defaultdict(int)
            count = 0
            head = n
            for  i in range(n, len(s) -x + 1, x):
                tmp = s[i:i+x]

                if tmp in counter:
                    dic[tmp] += 1
                    count += 1

                    while dic[tmp] > counter[tmp]:
                        dic[s[head:head+x]] -= 1
                        count -= 1
                        head += x
                    if count == len(words):
                        res.append(head)

                else:
                    head = i + x
                    dic = colelctions.defaultdict(int)
                    count = 0
        return res

605	Can Place Flowers	30.0%	Easy
56	Merge Intervals	29.8%	Medium
'''

'''
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        if len(intervals) == 0:
            return []

        intervals.sort(key = lambda x : x.start)
        result = [intervals[0]]

        for interval in intervals:
            if interval.start > result[-1].end:
                result.append(interval)
            else:
                result[-1].end = max(result[-1].end, interval.end)

        return result



581	Shortest Unsorted Continuous Subarray	29.7%	Easy
228	Summary Ranges	29.5%	Medium
55	Jump Game	29.5%	Medium
123	Best Time to Buy and Sell Stock III	29.1%	Hard
31	Next Permutation	28.7%	Medium
381	Insert Delete GetRandom O(1) - Duplicates allowed	28.7%	Hard
229	Majority Element II	28.5%	Medium
532	K-diff Pairs in an Array	28.2%	Easy
414	Third Maximum Number	27.8%	Easy
85	Maximal Rectangle	27.6%	Hard
57	Insert Interval	27.4%	Hard
#18	4Sum	26.6%	Medium
'''
a + b + c +  = target

'''
# Hash table
# TC: O(n^2)
# SC: O(n)
def fourSum(self, nums, target):
    if nums == None or len(nums) < 4:
        return []

    nums.sort()
    ans = set()
    dic = colelctions.defaultdict(list)

    for i, m in enumerate(nums):
        for j, n in enumerate(nums[:i]):
            sum = m + n
            if target - sum in dic:
                for sub in dic[target - sum]:
                    ans.add(tuple(sub + [n, m]))

        for k, p in enumerate(nums[i+1:]):
            dic[m + p].append([m, p])

    return list(ans)





# Divie and conquer
# general solutionL-- Ksum
#TC:  O(n^(k-1)), in this problem , O(n^3),
#SC:O(1)
class Solution(object):
    def fourSum(self, nums, target):
        if nums == None or len(nums) < 4:
            return []
        nums.sort()
        return [list(t) for t in self.kSum(nums, target, 4)]

    def kSum(self, nums, target, k):
        res = set()

        if k == 2:
            l, r = 0, len(nums) -1
            while l < r:
                sum = nums=[l] + nums[r]

                if sum == target:
                    res.add((nums[l], nums[r]))
                    l += 1
                elif sum < target:
                    l += 1
                else:
                    r -= 1

        else:
            l = 0
            while l < len(nums) - k + 1:
                for n in self.kSum(nums[l+1:], target - nums[l], k -1):
                    res.add((nums[l], ) + tuple(n))
                l += 1

        return res




84	Largest Rectangle in Histogram	26.5%	Hard
79	Word Search	26.5%	Medium
'''
search if a word in within a 2D matrix
'''
class Solution(object):

    def exist(self, board, word):

        visited = [[False] * len(board[0]) for _ in range(len(board))]
        if not board:
            return False
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.dfs(board, visited, i, j, word):
                    return True
        return False

    # check whether can find word, start at (i,j) position
    def dfs(self, board, visited, i, j, word):
        if len(word) == 0: # all the characters are checked
            return True
        if i<0 or i>=len(board) or j<0 or j>=len(board[0]) or word[0]!=board[i][j]:
            return False
        if visited[i][j]:
            return False
        tmp = board[i][j]  # first character is found, check the remaining part
        visited[i][j] = True   # Mark this as visited
        # check whether can find "word" along one of the four possible directions
        if self.dfs(board, visited, i+1, j, word[1:]) or self.dfs(board, visited, i-1, j, word[1:]) \
        or self.dfs(board, visited, i, j+1, word[1:]) or self.dfs(board, visited, i, j-1, word[1:]):
            return True
        # If we cnnot find the word, then we must backtrack
        visited[i][j] = False
        return False

45	Jump Game II	26.2%	Hard
54	Spiral Matrix	25.8%	Medium
152	Maximum Product Subarray	25.5%	Medium
41	First Missing Positive	25.4%	Hard
163	Missing Ranges 	24.8%	Medium
189	Rotate Array	24.4%	Easy
#4	Median of Two Sorted Arrays	21.6%	Hard
'''


'''
#15	3Sum	21.6%	Medium
'''
a + b + c = 0, all unique triplets, NO Duplicates

For example, given array S = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
'''

# SILU: fix the first number, then two pointers the second and last number

class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        nums.sort()
        for i in range(len(nums)-2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l, r = i+1, len(nums)-1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s < 0:
                    l +=1
                elif s > 0:
                    r -= 1
                else:
                    res.append((nums[i], nums[l], nums[r]))
                    while l < r and nums[l] == nums[l+1]:
                        l += 1
                    while l < r and nums[r] == nums[r-1]:
                        r -= 1
                    l += 1; r -= 1
        return res


#TC: O(n^2)
class Solution(object):
    def threeSum(self, nums):
        if nums == None or len(nums) < 3:
            return []
        nums.sort()
        res = []

        for i in range(len(nums)):
            #if i > 0 and nums[i] == nums[i - 1]
            #    continue

            # avoid duplicates, if same, go to the next
            if i == 0 or nums[i] != nums[i-1]:
                l, r = i + 1, len(nums) - 1
                while l < r:
                    if nums[i] + nums[l] + nums[r] == 0:
                        res.append((nums[i], nums[l], nums[r]))
                        # avoid duplicates
                        while l < r and nums[l] == nums[l + 1]:
                            l += 1
                        while l < r and nums[r] == nums[r-1]:
                            r -= 1
                        l += 1
                        r -= 1
                    elif  nums[i] + nums[l] + nums[r] < 0:
                        l += 1
                    else:
                        r -= 1
        return res



644	Maximum Average Subarray II	16.8%	Hard
126	Word Ladder II	14.0%	Hard











################################################################################
                        """Linked List Questions """
################################################################################

# E: Linked List Cycle

# KEY: slow and fast runner, slow moves 1 and fast moves 2 each times
class Solution(object):
    def hasCycle(self, head):
        if head is None or head.next is None:
            return False
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False





# E: Delete Node in a Linked List
'''
Delete a node except the tail in a single LL
'''

def deleteNode(node):
    if node.next == None:
        return
    else:
        node.val = node.next.val
        node.next = node.next.next


# E: Remove Duplicates from Sorted List
'''
Delete all duplicates such that each element appear only once

'''
def deleteDuplicates(head):
    if head is None:
        return head
    curr = head
    while curr.next:
        if curr.next.val == curr.val:
            curr.next = curr.next.next
        else:
            curr = curr.next
    return head

# Inetersection of Two Linked List
'''
Find the ndoe at which the Inetersection of two singly linked ist begins
'''

#https://leetcode.com/articles/intersection-two-linked-lists/#approach-2-hash-table-accepted
'''

Approach #1 (Brute Force) [Time Limit Exceeded]

For each node ai in list A, traverse the entire list B and check if any node in list B coincides with ai.

Complexity Analysis

Time complexity : O(mn)O(mn).
Space complexity : O(1)O(1).


Approach #2 (Hash Table) [Accepted]
Traverse list A and store the address / reference to each node in a hash set. Then check every node bi in list B: if bi appears in the hash set, then bi is the intersection node.

Complexity Analysis

Time complexity : O(m+n)
Space complexity : O(m) or O(n).

'''
def getIntersectionNode(headA, headB):
    nodeList = set()
    curr = headA
    while(curr != None):
        nodeList.add(curr)
        curr = curr.next

    pointer = headB
    while (pointer != None):
        if pointer in nodeList:
            return pointer
        pointer = pointer.next
    return None


# Remove Linked List Elements
def removeELements(head, val):
    dummy = ListNode(0)
    dummy.next = head
    curr = dummy

    while curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next
        else:
            curr = curr.next
    return dummy.next

# Reverse Linked List

'''
Complexity analysis

Time complexity : O(n)O(n). Assume that nn is the list's length, the time complexity is O(n)O(n).

Space complexity : O(1)O(1).
'''
def reverseLinkedList(head):
    prev = None
    curr = head
    while curr is not None:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next

    return prev

# E. Palindrome Linked List
"

class Solution(object):
    def isPalindrome(self, head):
        if head == None:
            return True

        mid = self.getMid(head)
        right = mid.next
        mid.next = None
        return self.compare(head, self.rotate(right))

    def rotate(self, head):
        pre = None
        while head:
            temp = head.next
            head.next = pre
            pre = head
            head = temp
        return pre

    def compare(self, h1, h2):
        while h1 and h2:
            if h1.val != h2.val:
                return False
            h1 = h1.next
            h2 = h2.next
        return True

    def getMid(self, head):
        slow = head
        fast = head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow


class Solution(object):
    def isPalindrome(self, head):

        #if not head or not head.next:
        #    return True

        # move slow pointer to the middle and fast pointer to the end
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

        # odd number, move slow pointer bias to the right part
        if fast:
            slow = slow.next

        # reverse the right part
        tail = None
        while slow:
            carry = slow.next
            slow.next = tail
            tail = slow
            slow = carry

        # check if the left and right parts are equal
        fast = head
        while tail:
            if tail.val != fast.val:
                return False
            tail = tail.next
            fast = fast.next

        return True

# E: Merge Two Sorted List


    def mergeTwoLists(l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """


        new_dummy_head = curr = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next

        curr.next = l1 or l2

        return new_dummy_head.next



# rotate List
'''
rotate the list to the right by k places, where k is non-negative


'''
def rotateRight(head, k):
    """
    :type head: ListNode
    :type k: int
    :rtype: ListNode
    """
    if k == 0 or head == None:
        return head
    dummy = ListNode(0)
    dummy.next = head
    p = dummy
    count = 0
    while p.next:
        p = p.next
        count += 1
    p.next = dummy.next
    step = count - ( k % count )
    for i in range(0, step):
        p = p.next
    head = p.next
    p.next = None
    return head

class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """

        if head is None or head.next is None or k == 0:
            return head

        tail = head
        length = 1
        while tail.next is not None:
            tail = tail.next
            length += 1
        tail.next = head

        k = k % length
        for i in range(length - k):
            tail = tail.next
        head = tail.next

# Sort List
'''
sort a LL in O(nlogn0) using constant space complexity
'''
class Solution(object):
    def sortList(self, head):
        if head == None or head.next == None:
            return head

        # right: mid.next, left: head
        mid = self.midMiddle(head)
        right = self.sortList(mid.next)
        mid.next = None
        left = self.sortList(head)
        return self.merge(left, right)


    def midMiddle(self, head):
        slow = head
        fast = head.next
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def merge(self, left, right):
        dummy = ListNode(0)
        curr = dummy
        while left != None and right != None:
            if left.val < right.val:
                curr.next = left
                left = left.next
            else:
                curr.next = right
                right = right.next

            curr = curr.next

        if left != None:
            curr.next = left
        if right != None:
            curr.next = right
        return dummy.next


# Insertion Sort List
"""
Sort a linked list using insertion sort
"""
class Solution(object):
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)

        while head:
            temp = dummy
            next = head.next
            while temp.next and temp.next.val < head.val:
                temp = temp.next

            head.next = temp.next
            temp.next = head
            head = next

        return dummy.next

class Solution(object):

    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return None

        dummy = ListNode(0)
        dummy.next = head
        while head.next != None:
            if head.next.val >= head.val:
                head = head.next
            else:
                temp = head.next
                head.next = temp.next
                temp.next = None
                x = dummy
                while x.next.val < temp.val:
                    x = x.next
                temp.next = x.next
                x.next  = temp
        return dummy.next





# Reorder List
'''
Given a singly linked list L: L0→L1→…→Ln-1→Ln,
reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…

You must do this in-place without altering the nodes' values.

For example,
Given {1,2,3,4}, reorder it to {1,4,2,3}.
'''
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: void Do not return anything, modify head in-place instead.
        """
        if not head:
            return

        # find the mid point
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # reverse the second half in-place
        pre, node = None, slow
        while node:
            pre, node.next, node = node, pre, node.next

        # Merge in-place; Note : the last node of "first" and "second" are the same
        first, second = head, pre
        while second.next:
            first.next, first = second, first.next
            second.next, second = first, second.next
        return



# Linked List Cycle II

'''
Given a LL, return the node where the cycle begins.
If no cycle, return null
NOTE :do NOT modify the linked list

FOLLOW UP- Solve it without using extra space

'''
class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow, fast = head, head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow == fast:
                slow = head
                while slow != fast:
                    slow, fast = slow.next, fast.next
                return slow
        return None


class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow,fast=head,head
        while True:
            if fast==None or fast.next==None:
                return None
            slow=slow.next
            fast=fast.next.next
            #当快指针和慢指针相遇时，停止
            if slow==fast:
                break
        #头节点和相遇时候的快节点同时触发，如果相等，则那个节点就是环的起点
        while head!=fast:
            head=head.next
            fast=fast.next
        return head

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        node_set = set()
        node = head
        while node:
            if node in node_set:
                return node
            else:
                node_set.add(node)
            node = node.next

        return None

    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow, fast = head, head
        while True:
            if fast == None or fast.next == None: return None
            slow = slow.next; fast = fast.next.next
            if slow == fast: break
        while head != fast:
            head = head.next; fast = fast.next
        return head

# Copy List with Random Pointer
'''

'''

class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """

        m = n = head
        dict = {}

        while m:
            dict[m] = RandomListNode(m.label)
            m = m.next

        while n:
            dict[n].next = dict.get(n.next)
            dict[n].random = dict.get(n.random)
            n = n.next
        return dict.get(head)



class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        if not head: return
        map = {}
        cur = head
        while cur:
            map[cur] = RandomListNode(cur.label)
            cur = cur.next
        for oldNode in map:
            newNode = map[oldNode]
            newNode.next = map.get(oldNode.next, None)
            newNode.random = map.get(oldNode.random, None)

        return map.get(head)


# Convert Sorted List to Binary Search Tree
'''
LL with elements are sorte in ascending order, conver ti to a height balanced BST

#https://discuss.leetcode.com/topic/18935/python-solutions-convert-to-array-first-top-down-approach-bottom-up-approach/2
'''

class Solution(object):

    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if not head: return None
        if not head.next: return TreeNode(head.val)
        slow = head
        fast = head.next.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        temp = slow.next
        slow.next = None
        root = TreeNode(temp.val)
        root.left = self.sortedListToBST(head)
        root.right = self.sortedListToBST(temp.next)
        return root


# Reverse Linked List II
'''
Reverse a linked list from position m to n. Do it in-place and in one pass
For example:
Given 1->2->3->4->5->NULL, m = 2 and n = 4,

return 1->4->3->2->5->NULL.

'''
def reverseBetween(self, head, m, n):

        dummy = middle = ListNode(0)
        dummy.next = head

        for i in range(m-1):
            dummy = dummy.next

        tail = None
        cur = dummy.next
        for i in range(n - m + 1):
            tmp = cur.next
            cur.next = tail
            tail = cur
            cur = tmp

        dummy.next.next = cur
        dummy.next = tail

        return middle.next








class Solution(object):
    def reverseBetween(self, head, m, n):
        dummy = ListNode(0)
        dummy.next = head
        prev, curr = dummy, head
        while m > 1:
            prev, curr = curr, curr.next
            m -= 1
            n -= 1
        rev, tail = self.reverse(curr, n-m+1)
        prev.next = rev
        curr.next = tail
        return dummy.next

    def reverse(self, head, count):
        prev, curr = None, head
        while count > 0:
            curr.next, prev, curr = prev, curr, curr.next
            count -= 1
        return prev, curr



# Partition List
'''
Given a LL and a value x, Partition it such that
all nodes less than x come before nodes greater than
 or euqla to x

 Should preserve the original relative order of node in two partitions

 For example,
Given 1->4->3->2->5->2 and x = 3,
return 1->2->2->4->3->5.
'''

class Solution:
    def partition(self, head, x):
        h1 = cur1 = ListNode(0)
        h2 = cur2 = ListNode(0)

        while head:
            tmp = head.next
            if head.val < x:
                cur1.next = head
                head.next = None
                cur1 = cur1.next
            else:
                cur2.next = head
                head.next = None
                cur2 = cur2.next

            head = tmp
        # 1's tail = 2's start
        cur1.next = h2.next
        return h1.next


# Rempve Duplicates from sorted list II
'''
Given a sorted LL, delete all the nodes that have duplicate numbers, leaving
 only distinct numbers, leaving only distinct numbers from the original
 list

For example,
Given 1->2->3->3->4->4->5, return 1->2->5.
Given 1->1->1->2->3, return 2->3.

'''

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        pre = dummy
        cur = head
        while cur!= None:
            while cur.next!=None and cur.next.val == cur.val:
                cur = cur.next
            if pre.next == cur:
                pre=pre.next
            else:
                pre.next = cur.next
            cur = cur.next
        return dummy.next


class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # key point: each node has different id, so when we do pre.next=cur we are actually comparing their ids
        # the general idea is to detect if pre.next is equal to cur, if not means there are duplicates, assign cur.next to pre.next
        # only move pre to the next position when pre.next is equal to cur

        #special case:[1,1,2,2] [1,1,2,3] [1,2,2,3] [1,2,3,3]

        dummy=ListNode(0)
        pre=dummy
        pre.next=head
        cur=head
        while cur!=None:
            while cur.next!=None and cur.val==cur.next.val:
                cur=cur.next
            if pre.next!=cur:
                pre.next=cur.next
            else:
                pre=pre.next
            cur=cur.next

        return dummy.next


# add two numbers I

'''
given two non-empty LL representing two non=negative integers. THe digits are stored in
Reverse
order and each of their ndoes contains a single digit.

Add the two numbers and return it as a LL
You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8


'''

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        dummy = ListNode(0)
        curr = dummy
        adv = 0
        while l1 is not None or l2 is not None or adv != 0:
            val = adv
            if l1 is not None:
                val += l1.val
                l1 = l1.next
            if l2 is not None:
                val += l2.val
                l2 = l2.next
            curr.next = ListNode(val % 10)
            curr = curr.next
            adv = val / 10
        return dummy.next


# Add two Numbers II
'''
Given two non-empty linked list representing two non-negatiefv integers.
The most significant digit comes first and each of their nodes contain a single digit. Add
the two numbers and return it as a linked list


ASSUME the two numbers do not contain leading zero.

FOLLOW-UP: what if reversing the lists is not allowed?
'''
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        s1, s2 = [], []
        while l1:
        	s1.append(l1.val)
        	l1 = l1.next
        while l2:
        	s2.append(l2.val)
        	l2 = l2.next
        r = 0
        res = None
        while s1 or s2:
        	t1 = t2 = 0
        	if s1: t1 = s1.pop()
        	if s2: t2 = s2.pop()
        	t = t1+t2+r

        	r = t//10
        	temp = res
        	res = ListNode(t%10)
        	res.next = temp
        if r != 0:
        	temp = res
        	res = ListNode(r)
        	res.next = temp
        return res


class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        n1 = n2 = 0
        while l1:
            n1 = n1*10 + l1.val
            l1 = l1.next
        while l2:
            n2 = n2*10 + l2.val
            l2 = l2.next
        total = str(n1 + n2)
        dummy = ListNode(0)
        tmp = dummy
        for char in total:
            tmp.next = ListNode(char)
            tmp = tmp.next
        return dummy.next



# Odd Even Linked List

'''
Given a singly LL, group all odd nodes together followed by the even nodes.
We are talking about the node number an not eh value in the nodes.

try to do it in-place.

The program should run in O(1) space complexity and O(n )time complexity

Example:
Given 1->2->3->4->5->NULL,
return 1->3->5->2->4->NULL.

'''
class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None: return
        odd = head
        even = head.next
        evenhead = even
        while even is not None and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next.next
            even = even.next
        odd.next = evenhead
        return head



class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """

        # 这道题甚至不用判断奇数还是偶数，通过测试简单样例可以得知，只是奇书位置的所有节点向前移动

        # 如果head为空，直接返回head
        if not head:
                return head
        # 设 isOdd 为 true
        isOdd = True
        # 设 cur 指向 head
        cur = head
        # 设 odd_head , even_head,odd_cur,even_cur
        odd_head,even_head = ListNode(0),ListNode(0)
        odd_cur = odd_head
        even_cur = even_head
        # 通过 cur 遍历 该链表：
        while cur:
        #         如果 isOdd:
                if isOdd:
        #                 将当前的节点接到odd_cur
                        odd_cur.next = cur
        #                 odd_cur后移一位
                        odd_cur = odd_cur.next
        #                 isOdd = false
                        isOdd = False
        #         否则：
                else:
        #                 将当前的节点接到even_cur
                        even_cur.next = cur
        #                 even_cur后移一位
                        even_cur = even_cur.next
        #                 isOdd = true
                        isOdd = True
                cur = cur.next
        # 将两个链表拼接起来
        even_cur.next = None
        # if even_head.next:
        if even_head.next:
        #         odd_cur.next 指向 even_head.next
                odd_cur.next = even_head.next
        # 返回 odd_head.next
        return odd_head.next



# Remove Nth NOde from end of List
'''
GIven a LL, remove the nth node from the end of list and return its head

For example,
   Given linked list: 1->2->3->4->5, and n = 2.
   After removing the second node from the end, the linked list becomes 1->2->3->5.

Note:
Given n will always be valid.
Try to do this in one pass

https://leetcode.com/articles/remove-nth-node-end-list/

Approach #1: two  pass

approach #2: one pass, using two pointers.

'''


# approach-2
class Solution(object):
    def removeNthFromEnd(self, head, n):
        slow,fast = head,head
        for _ in range(n):
            fast = fast.next
        if fast is None:
            return head.next
        while fast.next:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return head

# two pointers
class Solution(object):
    def removeNthFromEnd(self, head, n):

        # need a new head, if n is length of list
        dummy = cur = ListNode(0)
        dummy.next = head

        for i in range(n):
            head = head.next

        while head:
            cur = cur.next
            head = head.next

        cur.next = cur.next.next
        return dummy.next
# Swap Nodes in Paris
'''
Given a LL, swap every two adjacent noes an return its head

Should use only O(1) space. May NOT modify values in the list, only nodes itself can be changed.
'''

class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """


        dummy=ListNode(0)
        pre=dummy
        pre.next=head

        while pre.next and pre.next.next:
            a=pre.next
            b=a.next
            pre.next, a.next, b.next = b, b.next, a
            pre=a
        return dummy.next


class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """

        current=ListNode(-1)
        current.next=head
        # current=dummy
        while current.next and current.next.next:
            tmp=current.next.val
            current.next.val=current.next.next.val
            current.next.next.val=tmp
            current=current.next.next
        return head



# Reverse Noes in k-Group



# Merge k sorted lists




##############################################################################
''' '''
##############################################################################











'''
Array

'''


# Missing Number
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        for i in range(len(nums)):
            if i!=nums[i]:
                return i
        return len(nums)

class Solution(object):
        def missingNumber(self, nums):
            expected = len(nums) * (len(nums)+1)/2
            return int(expected - sum(nums))

class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        s = set(nums)
        for i in range(0, len(nums) + 1):
            if i not in s:
                return i
        return -1

# two Sum

# Move Zeroes
'''
Given an array nums, move all 0's to the end of array while matianting the
relative orer of the non-zero element

#do this In-Place without making a copy of the array
# minimize the total number of operations

https://leetcode.com/articles/move-zeroes/


'''
class Solution(object):
    def moveZeroes(self, a):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        i = 0
        for j in range(len(a)):
            if a[j] is not 0:
                a[i], a[j] = a[j], a[i]
                i += 1


# Two Sum II- input is sorted


# Majority Element
'''
Array of size n, find the majority element.
Majority element is the element that appear s more than n/2 times
'''

# Sotring

#python sort TC: o(nlogn)
def majorityElement4(self, nums):
    nums.sort()
    return nums[len(nums)//2]

# one pass + dictionary
def majorityElement2(self, nums):
    dic = {}
    for num in nums:
        if num not in dic:
            dic[num] = 1
        if dic[num] > len(nums)//2:
            return num
        else:
            dic[num] += 1



#Rotate array
'''
Rotate an array of n elements to the right by k steps

For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].

Note:
Try to come up as many solutions as you can, there are at least 3 different ways to solve this problem.

https://leetcode.com/articles/rotate-array/
Approach-1: Brute Force
Rotate all the elements of the array in k steps by the rotating the elements by 1 unit iin each steps

Approach-2: Extra Array
use an extra array in which we place every element of the array at its correct position.
i.e. the number at index i in the original array is placed at the index i+k
then we copy the new array to the origanl one

Approach-3: using Reverse
TC: O(n)
SC: O(1)

In this approach, we firstly reverse all the elements of the array. Then, reversing the first k elements followed by reversing the rest n-kn−k elements gives us the required result.

Let n=7n=7 and k=3k=3.

Original List                   : 1 2 3 4 5 6 7
After reversing all numbers     : 7 6 5 4 3 2 1
After reversing first k numbers : 5 6 7 4 3 2 1
After revering last n-k numbers : 5 6 7 1 2 3 4 --> Result



'''
# Approach-2
#TC: O(n)     SC: O(n)
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        this solution works, but it is not very pretty. try to come up with another way to do this more simply
        """

        nums2 = [0]*len(nums)

        for i in range(0, len(nums)):
            nums2[(i+k)%len(nums2)] = nums[i]

        for i in range(0, len(nums2)):
            nums[i] = nums2[i]


#Approach-3
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        self.reverse(nums, 0, len(nums) -1)
        self.reverse(nums, 0, k - 1)
        self.reverse(nums, k, len(nums) -1)


    def reverse(self, nums, start, end):
        while start < end:
            tmp = nums[start]
            nums[start] = nums[end]
            nums[end] = tmp
            start += 1
            end -= 1


# Remove duplicates from sorted array
'''
remove duplicates in place such that each element appear only once and return new length

Approach-1: two pointers

TC: O(n)    SC: O(1)

Since array already sorted, keep two pointers i and j, where i is the slower-runner and j is the fast -runner

as long as nums[i] = nums[j], we increment j to skip the duplicate

when we encounter nums[j] != nums[i],  the uplicate run has ened so we must copy its value to nums[i+1, i is then incremented
and we repeat the process again until j reaches the end of arrays]

https://leetcode.com/articles/remove-duplicates-sorted-array/
'''

class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0

        l1 = 0
        for l2 in range(1, len(nums)):
            if nums[l1] != nums[l2]:
                l1 += 1
                nums[l1] = nums[l2]
        return l1 + 1

class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums: return 0

        tail = 1
        for i in range(1, len(nums)):
            if nums[tail] != nums[i]:
                tail += 1
                nums[tail] = nums[i]

        return tail
# Remove Element
'''
Given an array and a value, remove all instances of that value in place and return new length

DO this in place

https://leetcode.com/articles/remove-element/



Approach #1 (Two Pointers) [Accepted]

Intuition

Since question asked us to remove all elements of the given value in-place,
 we have to handle it with O(1)O(1) extra space. How to solve it? We can keep
  two pointers ii and jj, where ii is the slow-runner while jj is the fast-runner.

Algorithm

When nums[j]nums[j] equals to the given value, skip this element by incrementing jj.
As long as nums[j] \neq valnums[j]≠val, we copy nums[j]nums[j] to nums[i]nums[i] and
increment both indexes at the same time. Repeat the process until jj reaches the end of the array and the new length is ii.

TC: O(n)   SC: O(1)


'''


class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        if not nums:
            return 0

        tail = 0
        for num in nums:
            if num != val:
                nums[tail] = num
                tail += 1

        return tail

# Array Partition I
'''
Given an array of 2n integers, task is to group ints into n pairs of integers,
say a1, b1), (a2, b2), ..., (an, bn) which makes sum of min(ai, bi) for all i from 1 to n as large as possible.


#Consider the smallest element x. It should be paired with the next smallest element,
 #because min(x, anything) = x, and having bigger elements only helps you have a
 #larger score. Thus, we should pair adjacent elements together in the sorted array.
'''


def arrayPairSum(self, A):
    return sum(sorted(A)[::2])




# Can Place Flowers
'''
Flowerbed in which some of the plots are planted and some not.
However, flowers cannot be planted in adjacent plots - the would compte for water and both Disappeared

Gveien a flower bed(represented as an array containing 0 and 1, where 0 means empty and 1 means not empty)
and a number n, return if n new flowers can be planted in it without violating the non-ajacent-flowers range-sum-query-immutable


https://leetcode.com/articles/can-place-flowers/

Approach - 1: Single Scan

Find out the extra maximum number of flowers, count, that can be
planted for the given flowerbed arrangement

we traverse over all the elements of the flowered and find out those elements which are 0, impying an empty
positin. For every such element, we check if its both ajacent positions are also empty. If so, we can planted
a flower at the current position without without violating the no-adjacent flowers rule.
For the first an last element, need NOT to check the previous and next ajacent positions

If the count obtained is greater than or equal to n, the required number of flowers to be planted, we
can plant n flowers in the empty spaces, otherwise not.






Approach-2: Optimized

Instead of find the maximum value of count that can be obtaine we can stop the process of checking the
positions for planting flowers as soon sas count becomes equal to n.

Doing this leads to an optimization of the first approache.
If count never becomes equal to n, n flowers can NOT be palnted a t the empty positions.

'''

# Approach 1

# Time complexity : O(n)O(n). A single scan of the flowerbedflowerbed array of size nn is done.
#Space complexity : O(1)O(1). Constant extra space is used.
class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        """
        :type flowerbed: List[int]
        :type n: int
        :rtype: bool
        """
        if not flowerbed or len(flowerbed) < n:
            return False

        count = 0

        for i, flower in enumerate(flowerbed):
            if flower == 0:
                if (i == 0 or flowerbed[i - 1] == 0) and (i == len(flowerbed) - 1 or flowerbed[i + 1] == 0) :
                    flowerbed[i] = 1
                    count += 1

        return count >= n



# Approach-2: optimized check n
# TC: O(n)    SC:O(1)
class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        """
        :type flowerbed: List[int]
        :type n: int
        :rtype: bool
        """
        if n == 0:
            return True
        for i in range(len(flowerbed)):
            if (i != 0 and flowerbed[i-1]) or flowerbed[i] or (i != len(flowerbed) - 1 and flowerbed[i+1]):
                continue
            flowerbed[i] = 1
            n -= 1
            if n == 0:
                return True
        return False

# Contains Duplicate
'''
Given an array of integers, find if the array contains any duplicates.

Function return to true if value appears at least twice, and return False if eeveyr element is istince

https://leetcode.com/articles/contains-duplicate/


Approach-1: Naive Linear Serach


Approach-2: Sorting
TC: O(nlogn)   SC: O(1)


Approach-3: Hash table
TC: O(n)  SC: O(n)
Time complexity : O(n)O(n). We do search() and insert() for nn times and each operation takes constant time.

Space complexity : O(n)O(n). The space used by a hash table is linear with the number of elements in it.


For certain test cases with not very large nn, the runtime of this method can
 be slower than Approach #2. The reason is hash table has some overhead in
 maintaining its property. One should keep in mind that real world performance
 can be different from what the Big-O notation says. The Big-O notation only tells
 us that for sufficiently large input, one will be faster than the other.
 Therefore, when nn is not sufficiently large, an O(n)O(n) algorithm can be slower than an O(n \log n)O(nlogn) algorithm.
'''


#Sort
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        nums.sort()

        for i in range(len(nums)-1):
            if nums[i] == nums[i+1]:
                return True

        return False


class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        duplicate = set()
        for n in nums:
            if n in duplicate:
                return True
            else:
                duplicate.add(n)

        return False

class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        seen = {}
        for x in nums:
            if x in seen:
                return True
            else:
                seen[x] = 1
        return False


#Third Maximum  number
'''
Given non-empty array of integer, return the third Maximum number in this array
If does not exist, return the maximum number

TC must be in O(n)


Input: [3, 2, 1]
Output: 1
Explanation: The third maximum is 1.]


Input: [1, 2]
Output: 2
Explanation: The third maximum does not exist, so the maximum (2) is returned instead.


Input: [2, 2, 3, 1]
Output: 1
Explanation: Note that the third maximum here means the third maximum distinct number.
Both numbers with value 2 are both considered as second maximum.

'''














# Search Insert Position
'''
KEY: find the first position >= target
'''
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # find the first position >= target
        if not nums or len(nums) == 0:
            return 0
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start)/2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                start = mid
            else:
                end = mid
        if nums[start] >= target:
            return start
        elif nums[end] >= target:
            return end
        else:
            return end + 1

# Find All Numbers Disappeared in an Array
'''
Given an array of ints, some elements appear twice or more

Find all the elements of [1, n] inclusive that o not appear in the array

Could you do it without extra space and in O(n) runtime? You may assume the
 returned list does not count as extra space.

EX:
Input:
[4,3,2,7,8,2,3,1]

Output:
[5,6]
'''
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        all_nums = set(nums)
        out = []
        for num in range(1, len(nums)+1):
            if not num in all_nums:
                out.append(num)
        return out

'''
For each number i in nums,
we mark the number that i points as negative.
Then we filter the list, get all the indexes
who points to a positive number.
Since those indexes are not visited.

'''
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # For each number i in nums,
        # we mark the number that i points as negative.
        # Then we filter the list, get all the indexes
        # who points to a positive number
        for i in xrange(len(nums)):
            index = abs(nums[i]) - 1
            nums[index] = - abs(nums[index])

        return [i + 1 for i in range(len(nums)) if nums[i] > 0]

# Plus One
'''

Given a non-negative int represented as a non-empty array of digits, plus one
to the integer.

Digits are sorted such that the most significant digit is at the head of the list


'''

class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """

        for i in range(len(digits)-1, -1, -1):
            if digits[i] < 9:
                digits[i] = digits[i] + 1
                return digits
            else:
                digits[i] = 0
        digits.insert(0, 1)
        return digits


class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        rem = 1
        for i in reversed(range(len(digits))):
            new = digits[i]+rem
            if new > 9:
                digits[i] = 0
                rem = 1
            else:
                digits[i] = new
                return digits

        return [1] + digits

# Max Consecutive Ones

'''
given a binary array, find the max number of consecutive 1s in this array

EX:
Input: [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s.
    The maximum number of consecutive 1s is 3.

NOTE:
1. input array will only contain 0 and 1

2. length of  the array is positive and will NOT excee 10, 000

'''

class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        maxLen = 0
        curLen = 0
        for num in nums:
            if num == 0:

                if maxLen < curLen:
                    maxLen = curLen
                curLen = 0

            else:
                curLen = curLen + 1
        if curLen > maxLen:
            maxLen = curLen
        return maxLen

class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        cnt = 0
        ans = 0
        for num in nums:
            if num == 1:
                cnt += 1
                ans = max(ans, cnt)
            else:
                cnt = 0

        return ans


class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        curLength = 0
        maxLength = 0
        for n in nums:
            if n == 1:
                curLength += 1
                if curLength > maxLength:
                    maxLength = curLength
            else:
                curLength = 0

        return maxLength


# Shortest Unsorted Continuous Subarray
'''
Given an integer array, need to find one Continuous Subarray that if you only
sort this subarray in ascending order, then the whole array will be sorted in
ascending order, too

Find the shorst such subarray and output its lenght

Input: [2, 6, 4, 8, 10, 9, 15]
Output: 5
Explanation: You need to sort [6, 4, 8, 10, 9] in ascending order to make the whole array sorted in ascending order.

Then length of the input array is in range [1, 10,000].
The input array may contain duplicates, so ascending order here means <=.
'''


# O(n)   O(1)
# java
public class Solution {
    public int findUnsortedSubarray(int[] nums) {
        int len=nums.length;
        int max=Integer.MIN_VALUE, min=Integer.MAX_VALUE;
        int start=-1, end=-1;

        for(int i=0; i<len; i++){
            max = Math.max(max, nums[i]); //from left to right, search the current max
            min = Math.min(min, nums[len-i-1]);  //from right to left, search the current min

            if(nums[i] < max)
                end = i;
            if(nums[len-i-1] > min)
                start = len-i-1;
        }

        if(start==-1) //the entire array is already sorted
            return 0;

        return end-start+1;
    }
}

# https://leetcode.com/articles/shortest-unsorted-continous-subarray/

class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        min_loc=len(nums)-1
        max_loc=0
        nums1=sorted(nums)
        for i in range(len(nums)):
            if nums[i]!=nums1[i]:
                if i<min_loc:
                    min_loc=i
                if i>max_loc:
                    max_loc=i
        if max_loc==0:
            return 0
        return max_loc-min_loc+1


class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        """
        2 pointers,
        nums on the left, should be smallest, ... going up
        nums on the right, should be largests, ... going down inside.
        """

        """
        nums_sorted = sorted(nums)

        # check non-overlap length
        # 1, 3, 5, 7, 2, 8, 9   # original
        # 1, 2, 3, 5, 6, 8, 9   # sorted

        i, j = 0, len(nums)-1
        while i < len(nums) and nums[i] == nums_sorted[i]:
            i += 1
        while i != len(nums) -1 and j > i and nums[j] == nums_sorted[j]:
            j -= 1

        if j == i:
            return 0
        return j - i + 1

        """

        # another approach, inspired by a solution  in the discussion.   SMART !!!
        """
        # java version
        public int findUnsortedSubarray(int[] A) {
             int n = A.length, beg = -1, end = -2, min = A[n-1], max = A[0];
             for (int i=1;i<n;i++) {
                 max = Math.max(max, A[i]);
                 min = Math.min(min, A[n-1-i]);
                 if (A[i] < max) end = i;
                 if (A[n-1-i] > min) beg = n-1-i;
            }
            return end - beg + 1;
        }

        # 1236477895 array
        # 0123456789 index
        # according to above code, end=9,  beg = 3.  return 7
        """

        l = len(nums)
        start, end = -1, -2
        current_max, current_min = nums[0], nums[l-1]

        for i in range(0, l):
            current_max = max(nums[i], current_max)
            current_min = min(nums[l-i-1], current_min)

            if nums[i] < current_max:
                end = i
            if nums[l-i-1] > current_min:
                start = l-i-1

        print start, end
        return end - start + 1


class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        sort = sorted(nums)
        l,r = 1,0
        for i in range(len(nums)):
		    if sort[i] != nums[i]:
		        l=i
		        break
        for j in range(len(nums)-1,l-1,-1):
			if sort[j]!=nums[j]:
				r=j
				break
        return r-l+1


class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        start = None
        end = None
        for i in range(len(nums)-1):
            if start == None and nums[i] > nums[i+1]:
                    start = i
                    #print start
            if start is not None and nums[i] > nums[i+1]:
                    end = i+1
                    #print end
        if start is None:
            return 0

        while end+1 <= len(nums)-1:
            if max(nums[start: end+1]) > nums[end+1]:
                end += 1
            else:
                break
        while start-1 >= 0:
            if min(nums[start: end+1]) < nums[start-1]:
                start -= 1
            else:
                break
        return (end - start + 1)




# Third Maximum Number

# Maximum Distance in arrays

# Shortest Word Distance

# best time to buy and sell stock Ii
'''

have an array for which the ith element is the price of a given stock on day i

Find the maximum profit. can complete as many transactions as you like, that is
buy one and sell one share of stock MULTIPLE TIMES!

MUST sell the stock before you buy again.

# https://leetcode.com/articles/best-time-buy-and-sell-stock-ii/
Solution
Approach #1 Brute Force [Time Limit Exceeded]
Time complexity : O(n^n) Recursive function is called n^n times.
Space complexity : O(n)O(n). Depth of recursion is nn.


Approach #2 (Peak Valley Approach) [Accepted]
Time complexity : O(n)O(n). Single pass.
Space complexity : O(1)O(1). Constant space required.

Approach #3 (Simple One Pass) [Accepted]
Time complexity : O(n)O(n). Single pass.
Space complexity: O(1)O(1). Constant space needed.

'''


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        total = 0
        for i in range(len(prices)-1):
            if prices[i+1] > prices[i]:
                total+= prices[i+1] - prices[i]

        return total




# Best time to Buy and Sell Stock
'''

Need to find out the maximum difference between two numbers in the given arrays
second number, selling price, must be larger than the first one, buying price

https://leetcode.com/articles/best-time-buy-and-sell-stock/

Approach #1 (Brute Force) [Time Limit Exceeded]

for (int i = 0; i < prices.length - 1; i++) {
            for (int j = i + 1; j < prices.length; j++) {
                int profit = prices[j] - prices[i];
                if (profit > maxprofit)
                    maxprofit = profit;

Approach #2 (One Pass) [Accepted]
. We need to find the largest peak following the smallest valley. We can maintain
two variables - minprice and maxprofit corresponding to the smallest valley and
 maximum profit (maximum difference between selling price and minprice) obtained
  so far respectively.

TC: O(n)
SC: O(1)
'''

def maxProfit(prices):
    low = float('inf')
    profit = 0
    for i in prices:
        profit = max(profit, i-low)
        low = min(low, i)
    return profit

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices or len(prices) == 1:
            return 0
        min_price = prices[0]
        max_profit = 0
        for num in prices:
            if num < min_price:
                min_price = num
            elif num - min_price > max_profit:
                max_profit = num - min_price
        return max_profit


# Reshape the matrix

# maximum Subarray



# Reshape the matrix
'''
reshape the matrix int a new one with different size but keep its original data



'''


# Pascal's Triangle Ii



# Maximum Product of three Numbers
'''
Given an integer array, find three numbers whose product is maxium and outpu the maxium

Input: [1,2,3]
Output: 6
Example 2:
Input: [1,2,3,4]
Output: 24

https://leetcode.com/articles/maximmum-product-of-three-numbers/
Solution
Approach #1 Brute Force [Time Limit Exceeded]
TC: O(N^3), consider every triplet from the nums of length n
SC: O(1), constant extra space used

Approach #2 Using Sorting [Accepted]
sort the given nums in ascending order and find out the product of last three numbers.
NOTE, this product will be maximum only if all numbers in numbers are position.
However, given the question, negative elements could also exist as well.
It also is possible that two negative numbers lying at the left extreme end ,
thrid number in the triplet being considered is the largest positiove number in the nums array.

Thus, either the product nums[0] * nums[1] * nums{1}
OR
nums[n - 3]*nums[n - 2]*nums[n-1]

TC: O(nlogn), sorting the nums takes O(nlogn)
SC: O(logn), sorting takes O(logn) space



Approach #3 Single Scan [Accepted]   O(n) & O(1)

NOt necessarily need to sort array to find max product

We could find the required 2 smallest values, min1 and majorityElement2
and the three largest values, max1, max2, max3 in the nums array, by iterating over the nums array only
once.
At the end, we can find out the larger value out of min*min2*max1
and max1*max2*max3 to find the max product

Time complexity : O(n). Only one iteration over the numsnums array of length nn is required.

Space complexity : O(1). Constant extra space is used.




'''

def maximumProduct(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    max1 = -1000
    max2 = -1000
    max3 = -1000
    min1 = 1000
    min2 = 1000
    for n in nums:
        if n >max1:
            max1,max2,max3 = n,max1,max2
        elif n >max2:
            max2,max3 = n,max2
        elif n >max3:
            max3 = n
        if n < min1:
            min1,min2 = n,min1
        elif n < min2:
            min2 = n
    #print(max1, max2, max3, min1, min2)
    return max(max1*max2*max3, max1*min1*min2)


nums = [1, 3, 4, 5, 6, 7, 8]
maximumProduct(nums)
# K-diff Pairs in an array

# Contains Duplciate Ii

# Merge Sorted Array


# Triangle
'''
Given a Triangle, find the minimum path sum from top to bottom. Each step
you may move to adjacent numbers on the row below

[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]

The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).


Bonus point if you are able to do this using only O(n) extra space, where n is the total number of rows in the triangle.


'''

# Modify the original triangle, top-down
def minimumTotal2(self, triangle):
    if not triangle:
        return
    for i in xrange(1, len(triangle)):
        for j in xrange(len(triangle[i])):
            if j == 0:
                triangle[i][j] += triangle[i-1][j]
            elif j == len(triangle[i])-1:
                triangle[i][j] += triangle[i-1][j-1]
            else:
                triangle[i][j] += min(triangle[i-1][j-1], triangle[i-1][j])
    return min(triangle[-1])

# Modify the original triangle, bottom-up
def minimumTotal3(self, triangle):
    if not triangle:
        return
    for i in xrange(len(triangle)-2, -1, -1):
        for j in xrange(len(triangle[i])):
            triangle[i][j] += min(triangle[i+1][j], triangle[i+1][j+1])
    return triangle[0][0]



class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        # use triangle itself for store, and bottom up to get the minimal value directly.

        row_len=len(triangle)
        for i in range(row_len-1,0,-1):
            for j in range(len(triangle[i])-1):
                triangle[i-1][j]+=min(triangle[i][j],triangle[i][j+1])

        return triangle[0][0]



# O(n) space
class Solution(object):
    def minimumTotal(self, triangle):
        if not triangle:
            return
        res = triangle[-1]
        for i in xrange(len(triangle)-2, -1, -1):
            for j in xrange(len(triangle[i])):
                res[j] = min(res[j], res[j+1]) + triangle[i][j]
        return res[0]



# Set Matrix Zeroes
'''
m * n matrix, if an element is 0, set its entire and column to 0.

Do it in place

FOLLOW-UP:
Did you use extra space?
A straight forward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?
'''

class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        row_index = []
        col_index = []

        # Record all the row/col indices where matrix element is a zero
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[i])):
                if matrix[i][j] == 0:
                    row_index.append(i)
                    col_index.append(j)

        for i in row_index:
            for j in range(0, len(matrix[i])):
                matrix[i][j] = 0

        for i in range(0, len(matrix)):
            for j in col_index:
                matrix[i][j] = 0


class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """

        i_set=set()
        j_set=set()
        for i, row in enumerate(matrix):
            for j,col in enumerate(row):
                if col ==0:
                    i_set.add(i)
                    j_set.add(j)


        for i, row in enumerate(matrix):
            for j,col in enumerate(row):
                if i in i_set or j in j_set:
                    matrix[i][j]=0


class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        zeroRows = [False] * len(matrix)
        zeroCols = [False] * len(matrix[0])
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    zeroRows[i] = True
                    zeroCols[j] = True
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if zeroRows[i] or zeroCols[j]:
                    matrix[i][j] = 0
# Serach a 2D matrix



class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if matrix == None or len(matrix) == 0 or len(matrix[0]) == 0: return False
        m, n = len(matrix), len(matrix[0])
        lo, hi = 0, m * n - 1
        while lo <= hi:
            mid = (lo + hi) / 2
            x = mid / n
            y = mid % n
            if matrix[x][y] == target:
                return True
            elif matrix[x][y] > target:
                hi = mid - 1
            else:
                lo = mid + 1

        return False



# Stair Search - O(n)
'''
# start from the top right,
if greater than matrix[0][columnLength-1], go down


'''


class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        n = len(matrix)
        if n == 0: return False
        m = len(matrix[0])
        if m == 0: return False

        i, j = 0, m - 1
        while i < n and j >= 0:
            if matrix[i][j] == target:
                return True
            elif  target > matrix[i][j]:
                i += 1
            else: # target < matrix[i][j]
                j -= 1

        return False






# Sort Colors
'''
Given an array with n objects colored red, white, blue, sort them so that objects of the same color are a
adjacent, with the colors in order, red white, an blue

use integers, 0, 1, 2 to represent red, white, blue rspectively.


'''

# O(n)       O(1)
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if not nums: return
        i, j = 0, 0
        for k in xrange(len(nums)):
            v = nums[k]
            nums[k] = 2
            if v < 2:
                nums[j] = 1
                j += 1
            if v == 0:
                nums[i] = 0
                i += 1


class Solution(object):
    def sortColors(self, nums):
        """

        """
        if not nums:
            return []
        n = len(nums)
        cur = 0
        lo,hi = 0,n-1

        while cur <= hi:
            if nums[cur] == 0:
                nums[cur],nums[lo] = nums[lo],nums[cur]
                lo += 1

            if nums[cur] == 2:
                nums[cur],nums[hi] = nums[hi],nums[cur]
                hi -= 1

            else: cur += 1




# Unique Paths
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """

        #   [[1] * n] * m
        board = [[1] * n for i in range(m)]

        for i in range(1, m):
            for j in range(1,n):
                board[i][j] = board[i-1][j] + board[i][j-1]
        return board[-1][-1]


# Unique Paths - II


class Solution(object):
    def uniquePathsWithObstacles(self, g):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        if len(g) == 0 or len(g[0]) == 0 or g[0][0] == 1:
            return 0
        m, n = len(g), len(g[0])
        f = [[0] * n for _ in range(m)]
        f[0][0] = 1
        for i in range(m):
            for j in range(n):
                if g[i][j] == 1:
                    continue
                if i != 0:
                    f[i][j] = f[i - 1][j]
                if j != 0 :
                    f[i][j] += f[i][j - 1]
        return f[m-1][n-1]


# optimize space
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [0] * (n + 1)
        dp[0] = 1

        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 0:
                    if j > 0:
                        dp[j] += dp[j- 1]
                else:
                    dp[j] = 0
        return dp[n - 1]


# Minimum Path Sum
"""
Given m*n grid filled with non-negative numbers, find a path from top left to
bottom right which mimimizes the sum of all numbers

NOTE: move one down or right
"""

class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m = len(grid)
        n = len(grid[0])
        for i in range(1, m):
            grid[i][0] += grid[i - 1][0]
        for j in range(1, n):
            grid[0][j] += grid[0][j - 1]
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
        return grid[-1][-1]


# 3sum
'''
Given an array S of n integers, find all unique triplets that gives the sum of zero
NOTE: no duplicate triplets

Note: The solution set must not contain duplicate triplets.

For example, given array S = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
'''

        nums.sort()
        res = []
        length = len(nums)
        for i in range(0, length - 2):
            if i and nums[i] == nums[i - 1]:
                continue
            target = nums[i] * -1
            left, right = i + 1, length - 1
            while left < right:
                if nums[left] + nums[right] == target:
                    res.append([nums[i], nums[left], nums[right]])
                    right -= 1
                    left += 1
                    while left < right and nums[left] == nums[left - 1]:
                           left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
                elif nums[left] + nums[right] > target:
                    right -= 1
                else:
                    left += 1
        return res



# find all duplicates in an array
'''
Given array of integers, some elements appear twice and others appear once
Find all elements that appear twice in this array

Do this in O(n) wihout using extra space
'''




class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        seen = set()
        res = []
        for num in nums:
            if num in seen:
                res.append(num)
            else:
                seen.add(num)
        return res


# O(n)    O(1)
 class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = []
        for x in nums:
            if nums[abs(x)-1] < 0:
                res.append(abs(x))
            else:
                nums[abs(x)-1] *= -1
        return res















################################################################################
"""


String


"""


# Rea N charaters given Read4
def read(self, buf, n):
    idx = 0
    while n > 0:
        # read file to buf4
        buf4 = [""]*4
        l = read4(buf4)
        # if no more char in file, return
        if not l:
            return idx
        # write buf4 into buf directly
        for i in range(min(l, n)):
            buf[idx] = buf4[i]
            idx += 1
            n -= 1
    return idx
# Multiple calls
### https://discuss.leetcode.com/topic/31965/python-solution-with-explainations-and-comments


# Implement StrStr
'''
Return the index of the first occurrence of needle in hastack. or -1 if not present


'''

class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if haystack is None or needle is None:
            return -1
        if needle == "":
            return 0
        for i in range(len(haystack)):
            for j in range(len(needle)):
                if i+j>=len(haystack):
                    return -1
                if haystack[i+j] != needle[j]:
                    break
                if j==len(needle)-1:
                    return i
        return -1

def strStr(self, haystack, needle):

    for i in range(len(haystack) - len(needle)+1):
        if haystack[i:i+len(needle)] == needle:
            return i
    return -1


# ddouble array, two pointers
def strStr(self, haystack, needle):
    m = len(haystack)
    n = len(needle)

    for i in range(m+1):
        for j in range(n+1):
            if j == n:
                return i

            # not found
            if i + j == m:
                return -1

            if haystack[i+j] != needle[j]:
                break


# KMP
def strStr(self, s, t):
    if len(t) > len(s): return -1
    kmp = [-1]
    for i in range(len(t)):
        j = kmp[i]
        while not (j == -1 or t[i] == t[j]):
            j = kmp[j]
        kmp.append(j + 1)
    i1 = i2 = 0
    while True:
        if i2 == len(t): return i1 - len(t)
        elif i1 == len(s): return -1
        elif i2 == -1 or s[i1] == t[i2]:
            i1 += 1
            i2 += 1
        else:
            i2 = kmp[i2]


'''
The time complexity is definitely not O(n), it is O(n*m).

Since checking haystack[i:i+len(needle)] == needle is O(m) done O(n) times.

n - length of haystack m - length of needle
'''


# Reverse words in a string -III
'''
Input: "Let's take LeetCode contest"
Output: "s'teL ekat edoCteeL tsetnoc"

 In the string, each word is separated by single space and there will not be any extra space in the string.

https://leetcode.com/articles/reverse-words-in-a-string/

'''
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        return ' '.join([i[::-1] for i in s.split()])



#Stuent Attenence Record I
'''
GIven a string representing an attendance record for a student. contaings following 3 chars
'A': Absent
"L": late
"P": present

rewarded if his record does not contain more than one A or more than two continuous "l"
'''

# so check that there are NOT two A's or three consecutifve 'L'

class Solution(object):
    def checkRecord(self, s):
        """
        :type s: str
        :rtype: bool
        """
        return s.count('A') <= 1 and s.count('LLL') == 0

class Solution(object):
    def checkRecord(self, s):
        """
        :type s: str
        :rtype: bool
        """
        n=len(s)
        if not n:
            return True
        countL,countA=0,0
        for i in range(n):
            if s[i]=="L":
                countL+=1
                if countL>2:
                    return False
            else:
                countL=0
                if s[i]=="A":
                    countA+=1
                    if countA>1:
                        return False
        return True



# Reverse String  II
'''
Given a string and an int k, reverse the first k chars for every 2k chars counting from the start
of the string
If less than k char left, reverse all of them.
'''

#For every block of 2k characters starting with position i, we want to replace S[i:i+k] with it's reverse.

def reverseStr(self, s, k):
    s = list(s)
    for i in xrange(0, len(s), 2*k):
        s[i:i+k] = reversed(s[i:i+k])
    return "".join(s)



# Longest uncommon Subsequence I
'''
given two stirngs, find longest uncommon subsequence of this groupfp of
two strings.

LUS is defined as the longest subsequence of one of these strings strings and
this subsequence should not be any subsequence of the other strings

A subsequence is a sequence that can be derived from one sequence by deleting some chars wihtout changing the order of the remaining elements.

'''

"""
Approach #1 Brute Force [Time Limit Exceeded]

In the brute force approach we will generate all the possible 2^n
​​ subsequences of both the strings and store their number of occurences in a
hashmap. Longest subsequence whose frequency is equal to 11 will be the required
 subsequence. And, if it is not found we will return -1−1.

Time complexity : O(2^x+2^y) where xx and yy are the lengths of strings aa and bb respectively .
Space complexity : O(2^x+2^y) 2^x+2^y subsequences will be generated.



These three cases are possible with string aa and bb:-

a=b. If both the strings are identical, it is obvious that no subsequence will
be uncommon. Hence, return -1.

length(a)=length(b) and a≠b. Example: abcabc and abdabd. In this case we can
consider any string i.e. abcabc or abdabd as a required subsequence, as out of
 these two strings one string will never be a subsequence of other string.
 Hence, return length(a)length(a) or length(b)length(b).

length(a)≠length(b). Example abcdabcd and abcabc. In this case we can consider
bigger string as a required subsequence because bigger string can't be a
subsequence of smaller string. Hence, return max(length(a),length(b)).

Complexity Analysis

Time complexity : O(min(x,y)) where xx and yy are the lengths of strings aa and bb respectively. Here equals method will take min(x,y)min(x,y) time .

Space complexity : O(1). No extra space required.

"""


class Solution(object):
    def findLUSlength(self, A, B):
        """
        :type a: str
        :type b: str
        :rtype: int
        """
        if A == B:
            return -1
        return max(len(A), len(B))


# Roman To Integer


class Solution:
    # @param {string} s
    # @return {integer}
    def romanToInt(self, s):
        ROMAN = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }

        if s == "":
            return 0

        index = len(s) - 2
        sum = ROMAN[s[-1]]
        while index >= 0:
            if ROMAN[s[index]] < ROMAN[s[index + 1]]:
                sum -= ROMAN[s[index]]
            else:
                sum += ROMAN[s[index]]
            index -= 1
        return sum



# Longest Common Prefix
'''
Write a fun to find the longest common prefix string amongst an array of strings

https://leetcode.com/articles/longest-common-prefix/
'''
class Solution:

    def lcp(self, str1, str2):
        i = 0
        while (i < len(str1) and i < len(str2)):
            if str1[i] == str2[i]:
                i = i+1
            else:
                break
        return str1[:i]

    # @return a string
    def longestCommonPrefix(self, strs):
        if not strs:
            return ''
        else:
            return reduce(self.lcp,strs)


class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ''
        p = strs[0]
        for s in strs[1:]:
            l = min(len(s), len(p))
            i = 0
            while i < l and s[i] == p[i]:
                i += 1
            p = p[:i]
        return p

class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if strs==[]:
            return ""
        first = min(strs)
        last = max(strs)
        r = ""
        for idx in range(min(len(first), len(last))):
            if first[idx] == last[idx]:
                r += first[idx]
            else:
                break
        return r


class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        #min (0, len(min))
        if not strs:
            return ''
        shortest_word = min(strs)
        if [shortest_word] == strs:
            return shortest_word

        result = ''
        for ch in shortest_word:
            for e in strs:
                if ch != e[len(result)]:
                    return result
            result += ch
        return result



# Detect Capital
'''
given a word, judge whether the usage of Capitals in it is right or not

right if:
1. All letters are capitals, like USA
2. All letters are not capitals. like leetcode
3. Only first letter is cap if it has more than one letter, like Google

otherwise, does not use capicals correctly


'''
public class Solution {
    public boolean detectCapitalUse(String word) {
        int cnt = 0;
        for(char c: word.toCharArray()) if('Z' - c >= 0) cnt++;
        return ((cnt==0 || cnt==word.length()) || (cnt==1 && 'Z' - word.charAt(0)>=0));
    }
}




# Valid Parentheses
def isValidParentheses(s):
    stack = []
    for char in s:
        if char == '(' or char == '[' or char == '}':
            stack.push(c)

        if c == ')':
            if stack or stack[-1] != ")":
                return False

        if c == ']':
            if stack or stack[-1] != '[':
                return False
        if c == '}':
            if stack or stack[-1] != '{':
                return False

        stack.pop()

    return  stack



# Repeated Substring Pattern
'''
Given non-empty string, check if it can be constructed by taking a substring of
it and appedning Multiple compies together
Assume string string consists of lowercase English letters only

Example 1:
Input: "abab"

Output: True

Explanation: It's the substring "ab" twice.
Example 2:
Input: "aba"

Output: False
Example 3:
Input: "abcabcabcabc"

Output: True

Explanation: It's the substring "abc" four times. (And the substring "abcabc" twice.)

'''

class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """

        l = len(s)
        next = [-1 for i in range(l)]
        j = -1
        for i in range(1, l):
            while j >= 0 and s[i] != s[j + 1]:
                j = next[j]
            if s[i] == s[j + 1]:
                j += 1
            next[i] = j
        lenSub = l - 1 - next[l - 1]
        return lenSub != l and l % lenSub ==0


class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """

        string =""
        lengthOfStr = len(s)
        for char in s[:int(lengthOfStr/2)]:
            string += char
            if lengthOfStr % len(string) == 0 and string * (lengthOfStr / len(string)) == s:
                return True
        return False


# Constructing string from Binary Tree
'''
Need to construct a string consisting of parentheses and inttegers from a binary
tree with preorder traversing way.

Null node represented by empty parenthesis pair (). Omit all the empty parentheses

Input: Binary tree: [1,2,3,4]
       1
     /   \
    2     3
   /
  4

Output: "1(2(4))(3)"

Explanation: Originallay it needs to be "1(2(4)())(3()())",
but you need to omit all the unnecessary empty parenthesis pairs.
And it will be "1(2(4))(3)".
Example 2:
Input: Binary tree: [1,2,3,null,4]
       1
     /   \
    2     3
     \
      4

Output: "1(2()(4))(3)"

Explanation: Almost the same as the first example,
except we can't omit the first parenthesis pair to break the one-to-one mapping
 relationship between the input and the output.

https://leetcode.com/articles/construct-string-from-binary-tree/#approach-1-using-recursion-accepted

'''
public class Solution {

    public String tree2str(TreeNode t) {
        if (t == null)
            return "";
        Stack < TreeNode > stack = new Stack < > ();
        stack.push(t);
        Set < TreeNode > visited = new HashSet < > ();
        String s = "";
        while (!stack.isEmpty()) {
            t = stack.peek();
            if (visited.contains(t)) {
                stack.pop();
                s += ")";
            } else {
                visited.add(t);
                s += "(" + t.val;
                if (t.left == null && t.right != null)
                    s += "()";
                if (t.right != null)
                    stack.push(t.right);
                if (t.left != null)
                    stack.push(t.left);
            }
        }
        return s.substring(1, s.length() - 1);
    }
}


class Solution(object):
    def tree2str(self, t):
        """
        :type t: TreeNode
        :rtype: str
        """
        if t == None:
            return ''
        if t.left == None and t.right == None:
            return str(t.val)
        if t.right == None:
            return str(t.val) + '(' + self.tree2str(t.left) + ')'
        return str(t.val) + '(' + self.tree2str(t.left) + ')' +
        '(' + self.tree2str(t.right) + ')'




# Number of Segments in a string
'''
Given number of segement in a string, where a segment is a contiguous sequence
of non-space chars.

Input: "Hello, my name is John"
Output: 5
'''
class Solution(object):
    def countSegments(self, s):
        """
        :type s: str
        :rtype: int
        """
        return len(s.split())

class Solution(object):
    def countSegments(self, s):
        """
        :type s: str
        :rtype: int
        """
        c = 0
        # at the end of string to get the last element
        s += ' '
        for i in xrange(len(s)):
            if s[i] != ' ' and s[i+1] == ' ':
                c += 1
        return c

def countSegments(self, s):
    return sum([s[i] != ' ' and (i == 0 or s[i - 1] == ' ') for i in range(len(s))])




public int countSegments(String s) {
    int res=0;
    for(int i=0; i<s.length(); i++)
        if(s.charAt(i)!=' ' && (i==0 || s.charAt(i-1)==' '))
            res++;
    return res;
}
'''
Time complexity:  O(n)
Space complexity: O(1)

'''


# Count and say - POORLY described
'''
count and say sequence is the sequence of intergers with the first five tersm as :

 1.     1
 2.     11
 3.     21
 4.     1211
 5.     111221
 6.     312211
 7.     13112221
 8.     1113213211
 9.     31131211131221
 10.   13211311123113112211


'''
class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        result = "1"
        for _ in xrange(2, n+1):
            count, length = 1, len(result)
            temp = ""
            result += "#"
            for i in xrange(length):
                if result[i+1] == result[i]:
                    count += 1
                else:
                    temp += str(count) + result[i]
                    count = 1
            result = temp
        return result


#  Ransom Note
'''s
Given an arbitrary ransoam note string and
'''




#revesrse vowels of a string
'''
takes a string as input and reverse only the vowels of a stirng

Example 1:
Given s = "hello", return "holle".

Example 2:
Given s = "leetcode", return "leotcede".

Note:

'''
class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        vowels = 'aeuioAEUIO'
        s = list(s)
        l, r = 0, len(s)-1
        while l < r:
            if s[l] not in vowels:
                l += 1
            elif s[r] not in vowels:
                r -= 1
            else:
                s[l], s[r] = s[r], s[l]
                l += 1
                r -= 1
        return ''.join(s)




# Reverse Strin

'''
takes a string and reverse it
'''

class SolutionClassic(object):
    def reverseString(self, s):
        r = list(s)
        i, j  = 0, len(r) - 1
        while i < j:
            r[i], r[j] = r[j], r[i]
            i += 1
            j -= 1

        return "".join(r)

   class Solution(object):
        def reverseString(self, s):
            """
            :type s: str
            :rtype: str
            """
            return s[::-1]


# Length of the last word
'''
Given string with uppper/lower cases letters and empty spaces.
return the length of the last word in the string.

If last does not exist, return 0

For example,
Given s = "Hello World",
return 5.

'''

class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        if(s == ""):
            return 0
        last = len(s)-1
        ans = 0
        while(last >= 0 and s[last] == " "):
            last -=1
        while(last >= 0 and s[last] != " "):
            last -=1
            ans +=1
        return ans


# Add Binary
'''
Given two binary strings, return their sum(also binary string)

For example,
a = "11"
b = "1"
Return "100".
'''
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        if a == "":
            return b
        if b == "":
            return a
        a_2 = int(a,2)
        b_2 = int(b,2)
        carry = 1
        while carry != 0:
            carry = (a_2 & b_2)<<1
            a_2 = a_2 ^ b_2
            b_2 = carry
        return bin(a_2)[2:]



# vali palindrome
'''
Given a string, determine if it s a palindrome.

For example,
"A man, a plan, a canal: Panama" is a palindrome.
"race a car" is not a palindrome.

Note:
Have you consider that the string might be empty? This is a good question to ask during an interview.

For the purpose of this problem, we define empty string as valid palindrome.

'''
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s = s.lower()
        l = 0
        r = len(s) - 1
        while l < r:
            while l < r and not s[l].isalnum():
                l += 1
            while l < r and not s[r].isalnum():
                r -= 1
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True



# Longest common subsequence
'''
Two strings, find the longest common subsequence

return the lenght of the LCS

For "ABCD" and "EDCA", the LCS is "A" (or "D", "C"), return 1.

For "ABCD" and "EACB", the LCS is "AC", return 2.
'''

class Solution:
    """
    @param A, B: Two strings.
    @return: The length of longest common subsequence of A and B.
    """
    def longestCommonSubsequence(self, A, B):
        # write your code here
        n, m = len(A), len(B)
        f = [[0] * (n + 1) for i in range(m + 1)]
        for i in range(n):
            for j in range(m):
                f[i + 1][j + 1] = max(f[i][j + 1], f[i + 1][j])
                if A[i] == B[j]:
                    f[i + 1][j + 1] = f[i][j] + 1
        return f[n][m]


# 583. Delete Operation for two Strings -  Longest common subsequence
'''
Given two words, find the min number of steps required to make word1 and wor2 the same
where each step you can delete one char in either string

nput: "sea", "eat"
Output: 2
Explanation: You need one step to make "sea" to "ea" and another step to make "eat" to "ea".


https://leetcode.com/articles/delete-operation-for-two-strings/
'''
class Solution(object):
    def minDistance(self, A, B):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        M, N = len(A), len(B)
        dp = [[0] * (N+1) for _ in xrange(M+1)]

        for i in xrange(M):
            dp[i][-1] = M-i
        for j in xrange(N):
            dp[-1][j] = N-j

        for i in xrange(M-1, -1, -1):
            for j in xrange(N-1, -1, -1):
                if A[i] == B[j]:
                    dp[i][j] = dp[i+1][j+1]
                else:
                    dp[i][j] = 1 + min(dp[i+1][j], dp[i][j+1])

        return dp[0][0]



#



##################################################################################
'''Basic Algorithms'''
########################################################################



class Solution():


    def sortInts(self, A):
        if A == None or len(A) == 0:
            return
        self.quickSort(A, 0, len(A) -1)

    def quickSort(self, A, start, end):
        if start >= end:
            return
        left = start, right = end
        # 1. pivot cannot be start or end, random number, could call random, but more calculation
        pivot = A[(start + end) / 2]

        # 2. left <= right
        # this while is partition algorithm
        while left <= right:
            while left <= right and A[left] < pivot:
                left += 1
            while left <= right and A[right] > pivot:
                right += 1

            if left <= right:
                A[left], A[right] = A[right], A[left]
                left += 1
                right -= 1
        # the while exits when left an right overlap

        self.quickSort(A, start, right)
        self.quickSort(A, left, end)






# Merge Sort

def mergeSort(alist):
    print("Splitting ",alist)
    if len(alist)>1:
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)

        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k]=lefthalf[i]
                i=i+1
            else:
                alist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            alist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            alist[k]=righthalf[j]
            j=j+1
            k=k+1
    print("Merging ",alist)


class Solution:
    # @param {int[]} A an integer array
    # @return nothing
    def sortIntegers2(self, A):
      self._mergesort(A, 0, len(A) - 1 )


    def _mergesort(self, aList, first, last ):
      # break problem into smaller structurally identical pieces
      mid = ( first + last ) / 2
      if first < last:
        self._mergesort( aList, first, mid )
        self._mergesort( aList, mid + 1, last )

      # merge solved pieces to get solution to original problem
      a, f, l = 0, first, mid + 1
      tmp = [None] * ( last - first + 1 )

      while f <= mid and l <= last:
        if aList[f] < aList[l] :
          tmp[a] = aList[f]
          f += 1
        else:
          tmp[a] = aList[l]
          l += 1
        a += 1

      if f <= mid :
        tmp[a:] = aList[f:mid + 1]

      if l <= last:
        tmp[a:] = aList[l:last + 1]

      a = 0
      while first <= last:
        aList[first] = tmp[a]
        first += 1
        a += 1










class Solution:

    def sortIntegers2(self, A):
        if A == None or len(0)  ==0:
            return

        # avoid created multiptle times in mergesort, create array here
        # then pass it to
        temp = []
        self.mergeSort(A, 0, len(A) - 1, temp)

    def mergeSort(self, A, start, end, temp):
        if start >= end:
            return
        # sort left half and right half
        mergeSort(A, start, (start + end) / 2, temp)
        mergeSort(A, (start + end) / 2 + 1, end, temp)
        # merge
        merge(A, start, end, temp)

    def merge(self, A, start, end, temp):
        middle = (start + end) / 2
        leftIndex = start
        rightIndex = middle + 1
        temp_index = start

        while leftIndex <= middle an rightIndex <= end:
            if A[leftIndex] < A[rightIndex]:
                A[temp_index] = A[leftIndex]
                temp_index += 1
                leftIndex += 1
            else:
                A[temp_index] = A[rightIndex]
                temp_index += 1
                rightIndex += 1

        # possible one array is not empty yet
        while leftIndex <= middle:
            temp[temp_index] = A[leftIndex]
            temp_index += 1
            leftIndex += 1
        while rightIndex <= end:
            temp[temp_index] = A[rightIndex]
            temp_index += 1
            rightIndex += 1

        for i in range(start, end):
            A[i] = temp[i]




#------------------------------------------------------------------------------

                                  """Stack Questions"""

#-----------------------------------------------------------------------------


 #	496 	Next Greater Element I 	56.9% 	Easy

class Solution(object):
    def nextGreaterElement(self, findNums, nums):

        d = {}
        stack = []
        result = []

        for x in nums:
            # if element in the stack smaller than next element,
            # pop the stack an make it as key, value is next value
            while  len(stack) and stack[-1] < x:
                d[stack.pop()] = x
            # add element in the stack
            stack.append(x)

        # go through the findNums, if key exists in the dictionary, then return
        # value, if not return the changed default value -1
        for x in findNums:
            result.append(d.get(x, -1))  # .get(KEY, DEFAULT if not found)

        return result


时间复杂度O(n * m) 其中n为nums的长度，m为findNums的长度
Python代码：
class Solution(object):
    def nextGreaterElement(self, findNums, nums):
        """
        :type findNums: List[int]
        :type nums: List[int]
        :rtype: List[int]
        """
        dmap = {v : k for k, v in enumerate(nums)}
        size = len(nums)
        ans = []
        for e, n in enumerate(findNums):
            for j in range(dmap[n] + 1, size):
                if nums[j] > n:
                    ans.append(nums[j])
                    break
            if len(ans) <= e: ans.append(-1)
        return ans


232 	Implement Queue using Stacks 	36.6% 	Easy
#225 	Implement Stack using Queues 	32.7% 	Easy
'''
push:
pop():
top():
empty()

'''


155 	Min Stack 	28.5% 	Easy
20 	Valid Parentheses 	33.3% 	Easy

class Solution:
    # @param {string} s A string
    # @return {boolean} whether the string is a valid parentheses
    def isValidParentheses(self, s):
        # Write your code here
        stack = []
        for ch in s:
            # 压栈
            if ch == '{' or ch == '[' or ch == '(':
                stack.append(ch)
            else:
                # 栈需非空
                if not stack:
                    return False
                # 判断栈顶是否匹配
                if ch == ']' and stack[-1] != '[' or \
                ch == ')' and stack[-1] != '(' or \
                ch == '}' and stack[-1] != '{':
                    return False
                # 弹栈
                stack.pop()
        return not stack


636 	Exclusive Time of Functions 	39.0% 	Medium
'''
function_id: start_or_end:temstamp
求每个函数调用的总时长
logs =
["0:start:0",
 "1:start:2",
 "1:end:5",
 "0:end:6"]

'''
class Solution(object):
    def exclusiveTime(self, n, logs):
        """
        :type n: int
        :type logs: List[str]
        :rtype: List[int]
        """
        ans = [0] * n
        stack = []
        prev_time = 0

        for log in logs:
            fn, typ, time = log.split(':')
            fn, time = int(fn), int(time)

            if typ == 'start':
                if stack:
                    ans[stack[-1]] += time - prev_time
                stack.append(fn)
                prev_time = time
            else:
                ans[stack.pop()] += time - prev_time + 1
                prev_time = time + 1

        return ans


#503 	Next Greater Element II 	47.1% 	Medium
'''
给定一个循环数组（末尾元素的下一个元素为起始元素），输出每一个元素的下一个更大的数字（
Next Greater Number）。Next Greater Number是指位于某元素右侧，大于该元素，且距离最近的元素。
如果不存在这样的元素，则输出-1。
'''
class Solution(object):
    def nextGreaterElements(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        stack = []
        size = len(nums)
        ans = [-1] * size
        for x in range(size * 2):
            i = x % size
            while stack and nums[stack[-1]] < nums[i]:
                ans[stack.pop()] = nums[i]
            stack.append(i)
        return ans






456 	132 Pattern 	28.7% 	Medium
'''

'''
439 	Ternary Expression Parser 	50.5% 	Medium
#402 	Remove K Digits 	26.2% 	Medium
#给定一个用字符串表示的非负整数num，从中移除k位数字，使得剩下的数字尽可能小。
"""
SILU: 使得栈中的数字尽可能保持递增顺序。
one can simply scan from left to right, and remove the first "peak" digit;
the peak digit is larger than its right neighbor. One can repeat this procedure k times
because it frequently remove a particular element from a string and has complexity O(k*n


One can simulate the above procedure by using a stack, and obtain a O(n) algorithm.
when the result stack (i.e. res) pop a digit, it is equivalent as remove
 that "peak" digit.
"""

 class Solution(object):
    def removeKdigits(self, num, k):
        """
        :type num: str
        :type k: int
        :rtype: str
        """
        n = len(num)
        if k == n:
            return "0"
        if k == 0:
            return num
        stack = []
        for d in num:
            while k and stack and stack[-1] > d:
                stack.pop()
                k -= 1
            stack.append(d)

        ret = stack[:-k] if k !=0 else stack
        return ''.join(ret).lstrip('0') or "0"

#394 	Decode String 	41.2% 	Medium
给定一个经过编码的字符串，返回其解码字符串。

编码规则为：k[encoded_string]，其中中括号内的encoded_string被重复k次。注意k一定是正整数。

"""
解题思路：
利用栈（Stack）数据结构。

当出现左括号时，将字符串压栈。

当出现右括号时，将字符串弹栈，并重复响应次数，累加至新的栈顶元素。
"""
class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []
        stack.append([1,""])
        num = ""
        for ch in s:
            # isdigit(): only returns if ch == '23'
            if ch.isdigit():
                num += ch
            elif ch == "[":
                stack.append([int(num), ""])
                num = ""
            elif ch == "]":
                k, st = stack.pop()
                stack[-1][1] += k * st
            else:
                stack[-1][1] += ch
        return stack[0][1]


class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []
        currNum = 0
        curr = ""

        for c in s:
            if c.isdigit():
                currNum = currNum * 10 + int(c)

            elif c == "[":
                stack.append(curr)
                stack.append(currNum)
                curr, currNum = "", 0

            # ['aaa', 2], curr = 'bc'
            # 'aaa' + bc * 2
            elif c == "]":
                num = stack.pop()
                pre = stack.pop()
                curr = pre + num * curr
            else:
                curr += c
        return curr

385 	Mini Parser 	30.4% 	Medium
#341 	Flatten Nested List Iterator 	41.3% 	Medium
'''
给定一个嵌套的整数列表，实现一个迭代器将其展开。
每一个元素或者是一个整数，或者是一个列表 -- 其元素也是一个整数或者其他列表。
'''
利用栈（Stack）数据结构对嵌套列表展开，在hasNext方法内将下一个需要访问的整数元素准备好，详见代码。


331 	Verify Preorder Serialization of a Binary Tree 	36.1% 	Medium
'''
The key here is, when you see two consecutive "#" characters on stack, pop both of them and replace the topmost element on the stack with "#". For example,

preorder = 1,2,3,#,#,#,#

Pass 1: stack = [1]

Pass 2: stack = [1,2]

Pass 3: stack = [1,2,3]

Pass 4: stack = [1,2,3,#]

Pass 5: stack = [1,2,3,#,#] -> two #s on top so pop them and replace top with #. -> stack = [1,2,#]

Pass 6: stack = [1,2,#,#] -> two #s on top so pop them and replace top with #. -> stack = [1,#]

Pass 7: stack = [1,#,#] -> two #s on top so pop them and replace top with #. -> stack = [#]

If there is only one # on stack at the end of the string then return True else return False.
'''
class Solution(object):
    def isValidSerialization(self, preorder):
        """
        :type preorder: str
        :rtype: bool
        """
        p = preorder.split(',')
        stack = []
        for s in p:
            stack.append(s)
            while len(stack) > 1 and stack[-1] == '#' and stack[-2] == '#':
                stack.pop()
                stack.pop()
                if not stack:
                    return False
                stack[-1] = '#'
        return stack == ['#']



255 	Verify Preorder Sequence in Binary Search Tree 	40.1% 	Medium
173 	Binary Search Tree Iterator 	41.1% 	Medium
'''
Next()
HasNext()
both in O(1) and O(h)
'''
class Solution:
    def __init__(self, root):
        self.stack = []
        self.pushAll(root)

    def hasNext(self):
        return self.stack

    def next(self):
        node = self.stack.pop()
        if node.right:
            self.pushAll(node.right)
        return node.val


    def pushAll(self, node):
        while node:
            self.stack.append(node)
            node = node.left



150 	Evaluate Reverse Polish Notation 	27.1% 	Medium
144 	Binary Tree Preorder Traversal 	44.9% 	Medium
'''
root, left, right

'''
class Solution:
    def preorder(self, root):

        if not root:
            return []

        preorder = []
        stack = [root]
        while stack:
            node = stack.pop()
            preorder.append(node.val)
            # add right first, so left popped first
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return preorder



103 	Binary Tree Zigzag Level Order Traversal 	34.4% 	Medium
94 	Binary Tree Inorder Traversal 	46.4% 	Medium
#71 	Simplify Path 	25.2% 	Medium
'''
Given an absolute path for a file, Simplifyit

For example,
path = "/home/", => "/home"
path = "/a/./b/../../c/", => "/c"


Speical cases:
1. ''/../''  -> just return /
2. a path may contain multiple // together, /home//foo/


'''
        # split with / then get rid of . and ''
        >>> [p for p in s.split('/') if p != '.' and p != ""]

        places = [p for p in path.split("/") if p!="." and p!=""]
        stack = []
        for p in places:
            if p == "..":
                if len(stack) > 0:
                    stack.pop()
            else:
                stack.append(p)
        return "/" + "/".join(stack)


class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        strs = path.split('/')
        strs2 = []
        for s in strs:
            if s == "." or s == "":
                continue
            if s == "..":
                if strs2:
                    strs2.pop()
            else:
                strs2.append(s)

        return '/'+'/'.join(strs2)

591 	Tag Validator 	27.2% 	Hard
316 	Remove Duplicate Letters 	29.4% 	Hard
272 	Closest Binary Search Tree Value II 	38.8% 	Hard
224 	Basic Calculator 	26.9% 	Hard
145 	Binary Tree Postorder Traversal 	40.0% 	Hard
#85 	Maximal Rectangle 	27.7% 	Hard  == 84
'''
Given binary matrix filled with 0 and 1s. FInd the largest rectangle
containing only 1's and return its area

For example, given the following matrix:

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0

Return 6.

'''

# stack
class Solution(object):
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if not matrix:
            return 0
        m, n, A = len(matrix), len(matrix[0]), 0
        height = [0 for _ in range(n)]
        # go through each row and add 1 to column if matrix[i][j] is 1 else 0
        for i in range(m):
            for j in xrange(n):

                height[j] = height[j]+1 if matrix[i][j]=="1" else 0
            A = max(A, self.largestRectangleArea(height))
        return A


    def largestRectangleArea(self, height):
        height.append(0)
        stack, A = [0], 0
        for i in range(1, len(height)):
            while stack and height[stack[-1]] > height[i]:
                h = height[stack.pop()]
                w = i if not stack else i-stack[-1]-1
                A = max(A, w*h)
            stack.append(i)
        return A



#84 	Largest Rectangle in Histogram 	26.6% 	Hard
'''
[2, 1, 5, 6, 2, 3], width for each is 1
# the largest rectangle is between 5 and 6, 5 x 2 = 10
'''
# stack in O(n), O(n)
'''
If stack is empty or value at top of stack is less than or equal to value at the current inex,
push it into stack
Otherwise keep removing values from the stack till value at top of stack is
leff than the value at current index.
while removing values from stack, calculate area.
'''
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        # add an empty bar to the end of height
        heights.append(0)
        stack = [-1]
        ans = 0
        # keep adding  indices to stack if current element is smaller than the value
        # of the stack
        for i in xrange(len(heights)):
            # last i, heights[i] is 0
            while heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i - stack[-1] - 1
                ans = max(ans, h * w)
            stack.append(i)
        return ans



42 	Trapping Rain Water 	36.6% 	Hard








#-------------------------------------------------------------------

                                    '''Hash Table Questions'''

#----------------------------------------------------------------------


205 Isomorphic Strings
'''
two strings, determine if they are isomorphic

Two strings are isomorphic if the characters in s can be replaced to get t.

Given "egg", "add", return true.

Given "foo", "bar", return false.

Given "paper", "title", return true.

You may assume both s and t have the same length.
'''

class Solution(object):
    def isIsomorphic(self, s, t):

        if len(s) != len(t):
            return False
        s2t, t2s = {}, {}

        for ss, tt in zip(s, t):
            if ss not in s2t and tt not in t2s:
                s2t[ss] = tt
                t2s[tt] = ss
            elif ss not in s2t or s2t[ss] != tt:
                return False
        return True



290	Word Pattern	33.0%	Easy ==  205 Isomorphic Strings
'''
given a patten and a string, find if str follows the same patten

pattern = "abba", str = "dog cat cat dog" should return true.
pattern = "abba", str = "dog cat cat fish" should return false.
'''
给定一个模式pattern和一个字符串str，判断str是否满足相同的pattern。


SILU: 使用字典dict分别记录pattern到word的映射以及word到pattern的映射

#str = "dog cat cat dog" should return true.
def wordPattern(self, pattern, str):
    words = str.split()
    if len(pattern) != len(words):
        return False

    patternDict, wordDict = {}, {}
    for pattern, word in zip(pattern, words):

        # pattern to word mapping
        if pattern not in patternDict:
            patternDict[pattern] = word

        #word to pattern mapping
        if word not in wordDict:
            wordDict[word] = pattern

        if wordDict[word] != pattern or patternDict[pattern] != word:
            return False
    return True


1	Two Sum	34.4%	Easy
599	Minimum Index Sum of Two Lists	46.9%	Easy
'''
two lists with strings, find the common item for both list
with LEAST LIST INDEX SUM, if there is a choice tie, output all of them
with no order requirement.
'''
class Solution:

    def findRestaurant(self, list1, list2):
        dict1 = {v : i for i, v in enumerate(list1)}
        minSum = len(list1) + len(list2)
        ans = []

        for i, v in enumerate(list2):
            if v not in dict1:
                continue
            currSum = i + dict1[v]

            if currSum < minSum:
                ans = [r]
                minSum = currSum
            elif currSum == minSum:
                ans.append(r)
        return ans


594	Longest Harmonious Subsequence	40.6%	Easy
'''
a harmonious array is an array where the difference betwen its max value and
its min value is exactly 1

给定整数数组nums，求其中最大值与最小值相差恰好为1(harmonious)的子序列的长度的最大值。

注意： 数组长度不超过20000
'''
SILU:
用字典cnt统计各数字出现的次数。

升序遍历cnt的键值对


Time complexity : O(n). One loop is required to fill mapmap and one for traversing the mapmap.

Space complexity : O(n). In worst case map size grows upto size nn.

def findLHS(self.nums):
    cnt = collections.Counter(nums)
    ans = 0
    lastKey = lastVal = None
    #> sorted(cnt.items())
    #[(1, 1), (2, 3), (3, 2), (5, 1), (7, 1)]
    for key, val in sorted(cnt.items()):
        if lastKey is not None and lastKey + 1 == key:
            ans = max(ans, val + lastVal)
        lastKey, lastVal = key, val
    return ans


575	Distribute Candies	59.5%	Easy
'''
给定一组长度为偶数的整数，其中每个数字代表一个糖果的种类标号。

将糖果均分给哥哥和妹妹，返回妹妹可以得到的最大糖果种类数。

determine the number oif uniuque elements in the first hafl of the array

the max of uniuque canies the girl can objtain is atmost
n/2.

Approaches:


Sorting

Using Set:
find the number of unique element  is to traverse
over all the elements an keepp putting elements in a set.
A set will only contain unique elements.

At the end, we count the numbers in the set,



'''

@Sorting:
O(nlogn): Sorting takes O(nlogn) time
O(1): constant space is used.

class Solution(object):
    def distributeCandies(self, candies):
        """
        :type candies: List[int]
        :rtype: int
        """
        candies.sort()
        count = 1
        idx = 1
        while idx < len(candies) and count < len(candies) / 2:
            if candies[idx] > candies[idx - 1]:
                count += 1
            idx += 1
        return count


@Set,
O(n)[array is traversed only once]
 O(n)[set will be size n in the worst case]

class Solution(object):
    def distributeCandies(self, candies):
        """
        :type candies: List[int]
        :rtype: int
        """
        s = set()
        for candy in candies:
            s.add(candy)
        # len(candies) /2, if unique elements is more than half
        # len(s), less than half
        return min(len(s), len(candies) / 2)


500	Keyboard Row	59.9%	Easy
'''
给定一组单词，返回可以用美式键盘中的某一行字母键入的所有单词。
'''

class Solution(object):
    def findWords(self, words):

        #>>> set('abcdsf')
        #{'d', 'c', 'a', 'b', 'f', 's'}
        rs = map(set, ['qwertyuiop', 'asdfghjkl', 'zxcvbnm'])
        ans = []
        for word in words:
            wset = set(word.lower())
            if any(wset <= rset for rset in rs):
                ans.append(word)
        return ans


463	Island Perimeter	57.1%	Easy
'''
给定一个二维地图，1表示陆地，0表示水域。单元格水平或者竖直相连（不含对角线）。
地图完全被水域环绕，只包含一个岛屿（也就是说，一个或者多个相连的陆地单元格）。
岛屿没有湖泊（岛屿内部环绕的水域）。单元格是边长为1的正方形。地图是矩形，长宽不超过100
。计算岛屿的周长。
'''
每一个陆地单元格的周长为4，当两单元格上下或者左右相邻时，令周长减2。
class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        result = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]:
                    result += 4
                    if i and grid[i-1][j]: # if i == if i > 0
                        result -= 2
                    if j and grid[i][j-1]:
                        result -= 2
        return result



447	Number of Boomerangs	44.9%	Easy
'''
给定平面上的n个两两不同的点，一个“回飞镖”是指一组点(i, j, k)满足i到j的距离=i到k的距离（考虑顺序）

计算回飞镖的个数。你可以假设n最多是500，并且点坐标范围在 [-10000, 10000] 之内。
'''
class Solution(object):
    def numberOfBoomerangs(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        count = 0
        for point1 in points:
            m = {}
            for point2 in points:
                dx = point1[0] - point2[0]
                dy = point1[1] - point2[1]
                d = dx*dx + dy*dy
                if d in m:
                    count += m[d]*2
                    m[d] +=1
                else:
                    m[d] = 1
        return count


438	Find All Anagrams in a String	33.7%	Easy
'''
given a string s and string p, find all the start inices of p's anagrams in s
'''
s:'cbaebabacd'   p : 'abc'

[0, 6]
0: cba is a anagram of abc
6: bac is an anagram of bac


class Solution(object):
    def findAnagrams(self, s, p):

        ls, lp = len(s), len(p)
        cp = collections.Counter(p)
        cs = collections.Counter()
        ans = []
        for i in range(ls):
            cs[s[i]] += 1
            if i >= lp:
                cs[s[i - lp]] -= 1
                if cs[s[i - lp]] == 0:
                    del cs[s[i - lp]]
            if cs == cp:
                ans.append(i - lp + 1)
        return ans




class Solution(object):
    def findAnagrams(self, s, p):

        res = []
        n, m = len(s), len(p)
        if n < m: return res
        # ord('z') - 122
        phash, shash = [0]*123, [0]*123
        for x in p:
            phash[ord(x)] += 1
        for x in s[:m-1]:
            shash[ord(x)] += 1
        for i in range(m-1, n):
            # add one each time
            shash[ord(s[i])] += 1
            if i-m >= 0:
                shash[ord(s[i-m])] -= 1
            if shash == phash:
                res.append(i - m + 1)
        return res


@ O(n)
class Solution(object):
    def findAnagrams(self, s, p):

        ls, lp = len(s), len(p)
        count = lp
        cp = collections.Counter(p)
        ans = []
        for i in range(ls):
            if cp[s[i]] >=1 :
                count -= 1
            cp[s[i]] -= 1
            if i >= lp:
                if cp[s[i - lp]] >= 0:
                    count += 1
                cp[s[i - lp]] += 1
            if count == 0:
                ans.append(i - lp + 1)
        return ans


409	Longest Palindrome	45.4%	Easy
'''
string consists of lower and uppper letters, find length of longest palindromes that can be built with those letters

case sensitive, Aa is not palindrome

给定一个只包含小写或者大写字母的字符串，寻找用这些字母可以组成的最长回文串的长度。

大小写敏感，例如"Aa"在这里不认为是一个回文。
'''
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s or len(s) == 0:
            return 0

        count = 0
        hash1 = {}
        for i in s:
            if i not in hash1:
                hash1[i] = 1
            else:
                del hash1[i]
                count += 1

        if hash1 != {}:
            return count*2 + 1
        else:
            return count*2


class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """

        d = {}
        count = 0

        for i in range(len(s)):
            if s[i] in d:
                del d[s[i]]
                count += 2
            else:
                d[s[i]] = 1

        if len(d) != 0:
            count += 1

        return count


136	Single Number	54.4%	Easy
359	Logger Rate Limiter 	59.6%	Easy
266	Palindrome Permutation 	56.8%	Easy
624	Maximum Distance in Arrays 	33.2%	Easy
242	Valid Anagram	46.3%	Easy
'''
Given two strings s and t, write a function to determine if t is an anagram of s.

For example,
s = "anagram", t = "nagaram", return true.
s = "rat", t = "car", return false.
'''

@sorting
Time complexity : O(nlogn). Assume that nn is the length of ss, sorting costs O(nlogn) and comparing two strings costs O(n).
Sorting time dominates and the overall time complexity is O(nlogn).

Space complexity : O(1). Space depends on the sorting implementation which, usually, costs O(1) auxiliary space if heapsort is used.
class Solution(object):
    def isAnagram(self, s, t):

        return sorted(s) == sorted(t)

@hashmap
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        lookup = {}
        for c in s:
            if c in lookup:
                lookup[c] += 1
            else:
                lookup[c] = 1
        for c in t:
            if c not in lookup:
                return False
            else:
                lookup[c] -= 1
        for item in lookup.values():
            if item != 0:
                return False
        return True



219	Contains Duplicate II	32.3%	Easy
217	Contains Duplicate	45.6%	Easy
205	Isomorphic Strings	33.7%	Easy
204	Count Primes	26.5%	Easy
'''
count the number of prime numbers less than a +n
'''
202	Happy Number	40.6%	Easy
645	Set Mismatch	41.1%	Easy
'''
集合S初始包含数字1到n。其中一个数字缺失，一个数字重复。

求其中重复的数字，与缺失的数字。

'''
用字典求重复的数字，用等差数列求和公式求缺失的数字

@sorting
class Solution(object):
    def findErrorNums(self, nums):

        nums.sort()
        twice = None
        for i, d in enumerate(nums):
            if twice is None and i > 0 and nums[i] == nums[i-1]:
                twice = nums[i]
        nums = set(nums)
        for i in range(len(nums) + 1):
            if i + 1 not in nums:
                return twice, i + 1


class Solution(object):
    def findErrorNums(self, nums):

        missing = set(range(1, len(nums)+1)) - set(nums)
        nums = sorted(nums)
        for i in range(0, len(nums)):
            if nums[i] == nums[i+1]:
                return [nums[i]] + list(missing)


class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        #create set, create range
        setOfNumbers, result = set(), range(2)

        #Traverse list, check if number exists in set else add
        for x in nums:
            if x in setOfNumbers:
                result[0] = x
            else:
                setOfNumbers.add(x)

        #Check which number is absent
        for x in range(len(nums)+1):
            if x not in setOfNumbers:
                result[1] = x

        #return result
        return result

170	Two Sum III - Data structure design 	24.4%	Easy
349	Intersection of Two Arrays	47.2%	Easy
350	Intersection of Two Arrays II	44.6%	Easy
246	Strobogrammatic Number 	39.7%	Easy
389	Find the Difference	50.5%	Easy
'''
给定两个字符串s和t，都只包含小写字母。

字符串t由字符串s打乱顺序并且额外在随机位置添加一个字母组成。

寻找t中新增的那个字母。
'''
分别统计s与t的字母个数，然后比对即可。若使用Python解题，可以使用collections.Counter


class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        dic = {}
        for i in t :
            if i in dic :
                dic[i] +=   1
            else :
                dic[i] = 1

        for i in s :
            if i in dic :
                dic[i] -= 1

        for key in dic :
            if dic[key] != 0 :
                return key



class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        # another solution is bit manipulation:
        res = 0

        for e in s:
            res = res ^ ord(e)

        for e in t:
            res = res ^ ord(e)

        return chr(res)



535	Encode and Decode TinyURL	73.9%	Medium
166	Fraction to Recurring Decimal	17.5%	Medium
'''
给定两个整数代表分数的分子和分母，返回字符串形式的小数

如果小数部分是循环的，用括号将循环节围起来。

例如，

给定分子 = 1, 分母 = 2, 返回 "0.5".
给定分子 = 2, 分母 = 1, 返回 "2".
给定分子 = 2, 分母 = 3, 返回 "0.(6)".
'''
347	Top K Frequent Elements	47.9%	Medium
'''

empty array of integers, return the k most frequent elements
Given [1,1,1,2,2,3] and k = 2, return [1,2].

Your algorithm's time complexity must be better than O(n log n), where n is the array's size.

给定一个非空整数数组，返回其前k个出现次数最多的元素

'''

@bucket sort . O(n)
1. 遍历数组nums，利用字典cntDict统计各元素出现次数。
2. 遍历cntDict，利用嵌套列表freqList记录出现次数为i（ i∈[1, n] ）的所有元素
3. 逆序遍历freqList，将其中的前k个元素输出。
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        n = len(nums)
        cntDict = collections.defaultdict(int)
        for i in nums:
            cntDict[i] += 1
        freqList = [[] for i in range(n + 1)]
        for p in cntDict:
            freqList[cntDict[p]] += p,
        ans = []
        for p in range(n, 0, -1):
            ans += freqList[p]
        return ans[:k]







525	Contiguous Array	38.9%	Medium
'''
find the max length of a contiguous subarray with equal number of 0 and 1
给定一个二进制数组，求其中满足0的个数与1的个数相等的最长子数组
https://leetcode.com/articles/contiguous-array/#approach-3-using-hashmap-accepted
'''
class Solution(object):
    def findMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dmap = {0 : -1}
        ans = total = 0
        for i, n in enumerate(nums):
            total += 2 * nums[i] - 1
            if total in dmap:
                ans = max(ans, i - dmap[total])
            else:
                dmap[total] = i
        return ans


508	Most Frequent Subtree Sum	52.2%	Medium
325	Maximum Size Subarray Sum Equals k 	42.3%	Medium
187	Repeated DNA Sequences	31.4%	Medium
'''
Given s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT",

Return:
["AAAAACCCCC", "CCCCCAAAAA"].

DNA由一系列简写为A,C,G,T的碱基组成，例如"ACGAATTCCG"。研究DNA时，识别DNA中的重复序列有时候会有用处。

写一个函数找出DNA分子中所有不止一次出现的10字母长度的子串序列。
'''
字典+位运算，或者进制转换。

由于直接将字符串存入字典会导致Memory Limit Exceeded，采用位操作将字符串转化为整数可以减少内存开销。



class Solution(object):
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        newSeq = set()
        repeated = set()
        for i in range(0, len(s) - 10 + 1):
            subSeq = s[i : i + 10]

            if subSeq not in newSeq:
                newSeq.add(subSeq)
            else:
                repeated.add(subSeq)

        return list(repeated)



311	Sparse Matrix Multiplication 	50.7%	Medium
299	Bulls and Cows	34.8%	Medium
'''
告诉他有多少个数字处在正确的位置上（称为"bulls" 公牛），以及有多少个数字处在错误的位置上（称为"cows" 奶牛）

神秘数字：  1807
朋友猜测：  7810
提示信息：  1公牛 3奶牛。（公牛是8， 奶牛是0, 1和7）
，使用A表示公牛，B表示母牛，在上例中，你的函数应当返回1A3B。
'''

bull = secret与guess下标与数值均相同的数字个数

cow = secret与guess中出现数字的公共部分 - bull


244	Shortest Word Distance II 	37.9%	Medium
288	Unique Word Abbreviation 	16.6%	Medium
274	H-Index	33.1%	Medium
249	Group Shifted Strings 	41.2%	Medium
314	Binary Tree Vertical Order Traversal 	36.5%	Medium
3	Longest Substring Without Repeating Characters	24.3%	Medium
49	Group Anagrams	34.5%	Medium
454	4Sum II	46.4%	Medium
451	Sort Characters By Frequency	50.9%	Medium
'''
Given a string, sort it in decreasing orer based on frequency of chars

A and a are not the same.


Input: "Aabb"

Output:
"bbAa"

Explanation:
"bbaA" is also a valid answer, but "Aabb" is incorrect.
Note that 'A' and 'a' are treated as two different characters.


给定一个字符串，将字符按照出现次数倒序排列。
'''
class Solution(object):
    def frequencySort(self, s):

        ans = ''
        c = []
        for x in set(s):
            #[[1, 'a'], [2, 'b'], [1, 'A']]
            c.append([s.count(x),x])
        # [[2, 'b'], [1, 'a'], [1, 'A']]
        c.sort(reverse=True)
        for y in c:
            ans += y[0]*y[1]
        return ans

609	Find Duplicate File in System	53.0%	Medium
18	4Sum	26.7%	Medium
94	Binary Tree Inorder Traversal	46.5%	Medium
138	Copy List with Random Pointer	26.3%	Medium
648	Replace Words	48.6%	Medium
'''
英文中，以较短的单词为前缀，可以构成较长的单词。此时前缀可以称为“词根”。

给定一组词根字典dict，一个句子sentence。将句中的单词换为字典中出现过的最短词根。
'''


@Trie
利用词根dict构建字典树trie，遍历sentence中的word，在trie中进行搜索。
class TrieNode:
    def __init__(self):
        self.children = dict()
        self.isWord = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for letter in word:
            child = node.children.get(letter)
            if child is None:
                child = TrieNode()
                node.children[letter] = child
            node = child
        node.isWord = True

    def search(self, word):
        ans = ''
        node = self.root
        for letter in word:
            node = node.children.get(letter)
            if node is None: break
            ans += letter
            if node.isWord: return ans
        return word

class Solution(object):
    def replaceWords(self, dict, sentence):
        """
        :type dict: List[str]
        :type sentence: str
        :rtype: str
        """
        trie = Trie()
        for word in dict: trie.insert(word)
        ans = []
        for word in sentence.split():
            ans.append(trie.search(word))
        return ' '.join(ans)

380	Insert Delete GetRandom O(1)	39.1%	Medium
36	Valid Sudoku	35.7%	Medium
554	Brick Wall	44.6%	Medium
356	Line Reflection 	30.1%	Medium
355	Design Twitter	25.4%	Medium
632	Smallest Range	42.7%	Hard
336	Palindrome Pairs	26.1%	Hard
159	Longest Substring with At Most Two Distinct Characters 	41.0%	Hard
358	Rearrange String k Distance Apart 	31.9%	Hard
149	Max Points on a Line	15.3%	Hard
340	Longest Substring with At Most K Distinct Characters 	38.7%	Hard
85	Maximal Rectangle	27.8%	Hard
76	Minimum Window Substring	25.2%	Hard
37	Sudoku Solver	30.1%	Hard
30	Substring with Concatenation of All Words	22.0%	Hard
381	Insert Delete GetRandom O(1) - Duplicates allowed	28.8%	Hard

#--------------------------------------------------------------------------

                            '''Breath First Search Questions '''

#------------------------------------------------------------------------------

# Number of Islands
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        # visit each num.
        # when found a 1, propagate to all its 1 neighbors and mark them with 2.
        # increment isand counter
        if len(grid) == 0 or len(grid[0]) == 0:
            return 0

        islands = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    self.propagate(grid, i, j)
                    islands += 1

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "#":
                    grid[i][j] = "1"

        return islands


    def propagate(self, grid, i, j):
        n = len(grid)
        m = len(grid[0])
        dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        stack = [[i, j]]
        while len(stack) > 0:
            i, j = stack.pop()
            grid[i][j] = "#"
            for d in dirs:
                ii = i + d[0]
                jj = j + d[1]
                if ii >= 0 and ii < n and jj >= 0 and jj < m and grid[ii][jj] == "1":
                    stack.append([ii, jj])

# Surrounded Region
class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        if not any(board):
            return

        n, m = len(board), len(board[0])
        q = [ij for k in range(max(n,m)) for ij in ((0, k), (n-1, k), (k, 0), (k, m-1))]
        while q:
            i, j = q.pop()
            if 0 <= i < n and 0 <= j < m and board[i][j] == 'O':
                board[i][j] = 'W'
                q += (i, j-1), (i, j+1), (i-1, j), (i+1, j)

        board[:] = [['XO'[c == 'W'] for c in row] for row in board]

find all regions surrrouned by X,
REverse thought: find all the 'O' taht are not surrrouned by 'X'
'O's on the left, right, up, down side MUST NOT BE  surrounded, so we could fill
water from the "O" on four sides. The remaining "O" that has not been filled with water is surrouned by 'X'.
this is because the "O" are surrouned by X, water cannot reach

change the spot where water is filled to O, change rest of spots to X()


class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        queue = collections.deque([])
        for r in xrange(len(board)):
            for c in xrange(len(board[0])):
                if (r in [0, len(board)-1] or c in [0, len(board[0])-1]) and board[r][c] == "O":
                    queue.append((r, c))

        # outer layer, filling water into "O" spots, and filling water into adjacent spots
        while queue:
            r, c = queue.popleft()
            if 0<=r<len(board) and 0<=c<len(board[0]) and board[r][c] == "O":
                board[r][c] = "D"
                queue.append((r-1, c)); queue.append((r+1, c))
                queue.append((r, c-1)); queue.append((r, c+1))


        # 'O's are the spots surrouned by the x. change them to 'X'
        # change the spots marked D, where water is filled back to O
        for r in xrange(len(board)):
            for c in xrange(len(board[0])):
                if board[r][c] == "O":
                    board[r][c] = "X"
                elif board[r][c] == "D":
                    board[r][c] = "O"


# Perfect Squares
给定一个正整数n，求相加等于n的完全平方数（例如 1, 4, 9, 16, ...）的最小个数。

例如，给定n = 12，返回3，因为12 = 4 + 4 + 4；给定n = 13，返回2，因为13 = 4 + 9。

def numSquares(self, n):
    if n < 2:
        return n
    lst = []
    i = 1
    while  i * i <= n:
        lst.append(i*i)
        i += 1
    cnt = 0
    toCheck = {n}
    while toCheck:
        cnt += 1
        temp = set()
        for x in toCheck:
            for y in lst:
                if x == y:
                    return cnt
                if x < y:
                    break
                temp.add(x - y)
        toCheck = temp
    return cnt



# clone graph
克隆一个无向图。图中的每个节点包含一个标签及其邻居的列表。
class Solution:
    # @param node, a undirected graph node
    # @return a undirected graph node
    def cloneGraph(self, node):
        if node is None: return
        d = {}
        queue = []

        queue.append(node)

        vert = UndirectedGraphNode(node.label)
        d[node] = vert

        while queue:
            current_vert = queue.pop()

            for nbr in current_vert.neighbors:
                x = d.get(nbr)

                if x:
                    d[current_vert].neighbors.append(d[nbr])
                else:
                    p = UndirectedGraphNode(nbr.label)
                    d[current_vert].neighbors.append(p)
                    d[nbr] = p
                    queue.insert(0, nbr)

        return vert


class Solution:
    # @param node, a undirected graph node
    # @return a undirected graph node
    def cloneGraph(self, node):
        if not node: return
        root = UndirectedGraphNode(node.label)
        visited = {}
        visited[node.label] = root
        queue = collections.deque()
        queue.append(node)

        while queue:
            front = queue.popleft()

            for nei in front.neighbors:
                if nei.label not in visited:
                    visited[nei.label] = UndirectedGraphNode(nei.label)
                    queue.append(nei)
                visited[front.label].neighbors.append(visited[nei.label])
        return root



# Graph Valid Tree
Given n nodes, labeled from 0 to n-1, and a list of undirected edges, write a function to check these edges make up a vali tree

class Solution:
    # @param {int} n an integer
    # @param {int[][]} edges a list of undirected edges
    # @return {boolean} true if it's a valid tree, or false
    def validTree(self, n, edges):
        # Write your code here
        if len(edges) != n - 1:
            return False

        neighbors = collections.defaultdict(list)
        for u, v in edges:
            neighbors[u].append(v)
            neighbors[v].append(u)

        visited = {}
        from Queue import Queue
        queue = Queue()

        queue.put(0)
        visited[0] = True
        while not queue.empty():
            cur = queue.get()
            visited[cur] = True
            for node in neighbors[cur]:
                if node not in visited:
                    visited[node] = True
                    queue.put(node)

        return len(visited) == n




#--------------------------------------------------------------------------

                            '''Topological Sorting Questions '''

#------------------------------------------------------------------------------


# Course Schedule




class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        graph = [[] for _ in range(numCourses)]
        indegree = [0] * numCourses
        for x, y in prerequisites:
            graph[y].append(x)
            indegree[x] += 1
        count = 0
        q = []
        for idx in range(numCourses):
            if indegree[idx] == 0:
                q.append(idx)

        while q:
            idx = q.pop()
            indegree[idx] = -1
            count += 1
            for node in graph[idx]:
                indegree[node] -=1
                if indegree[node] == 0:
                    q.append(node)
        return numCourses == count





class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        count, graph = [0]*numCourses, collections.defaultdict(list)
        for pre in prerequisites:
            graph[pre[1]].append(pre[0])
            count[pre[0]]+=1
        queue = collections.deque(c for c in range(numCourses) if count[c]==0)
        visited = len(queue)
        while queue:
            courses = graph[queue.popleft()]
            for course in courses:
                count[course]-=1
                if count[course]==0:
                    queue.append(course)
                    visited+=1
        return visited==numCourses





# Course Schedule II


class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        graph = collections.defaultdict(list)
        indegree = [0] * numCourses

        # build graph
        for pre in prerequisites:
            graph[pre[1]].append(pre[0])
            indegree[pre[0]] += 1

        queue = collections.deque(c for c in range(numCourses) if indegree[c] == 0)

        order = []    # order list to store order
        while queue:
            course = queue.popleft()   # pop the  courses in the queue, CS1, courses = graph[popleft()]
            order.append(course)       # add the course to the order
            for course in graph[course]:    # graph[course]
                indegree[course] -= 1
                if indegree[course] == 0:
                    queue.append(course)

        if len(order) == numCourses:  # only return the order if len(order) == number of courses popped from queue
            return order
        return []                     # not equal

# alien dictionary
from collections import defaultdict, deque
class Solution(object):
    def alienOrder(self, words):

        # Both first go through the word list to find letter pairs (a, b)
        #where a must come before b in the alien alphabet
        graph = defaultdict(set)
        indegrees = {}
        for w in words:
            for c in w:
                indegrees[c] = 0

        # find (a, b) pairs where a comes before b
        for i in xrange(1, len(words)):
            word1 = words[i-1]
            word2 = words[i]
            for c1, c2 in zip(word1, word2):
                if c1 == c2:
                    continue

                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    indegrees[c2] += 1
                break

       #queue = deque(k for k in indegrees if indegrees[k] == 0)
       #OR
        queue = deque()
        for char in indegrees:
            if indegrees[char] == 0:
                queue.append(char)

        ans = ''
        while queue:
            char = queue.popleft()
            ans += char
            for neighbor in graph[char]:
                indegrees[neighbor] -= 1
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)

        return ans if len(ans) == len(indegrees) else ''

# Sequence Reconstruction



#-----------------------------------------------------------------------------
            '''Basic Data Structure Implementations'''
#----------------------------------------------------------------------------


                            """Heap"""

class MinHeap():
    def __init__(self):
        # since complete, can use a list
        # first element 0, easier for integer division for letter methods
        self.heapList = [0]
        self.currentSize = 0

    # insert and bubble up the inserted element to its proper location
    def bubbleUp(self, i):
        # while it has parent
        while i // 2 > 0:
            if self.heapList[i] < self.heapList[i // 2]:
                # swap node and its parent
                tmp = self.heapList[i // 2]
                self.heapList[i // 2] = self.heapList[i]
                self.heapList[i] = tmp
            # i becomes parent
            i = i // 2

    def insert(self,k):
        self.heapList.append(k)
        self.currentSize += 1
        self.bubbleUp(self.currentSize)


    '''
    Delete done in two steps:
    1. move last item to root
    2. bubble down new root to its proper location, swap with smaller child
       continue swapping until it is at a position where it is less than both children
    '''
    def bubbleDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]:
                tmp = self.heapList[i]
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = tmp
            i = mc

    def minChild(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i*2] < self.heapList[i*2+1]:
                return i * 2:
            else:
                return i * 2 + 1


    def deleteMin(self):
        retVal = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize = self.currentSize - 1
        self.heapList.pop()
        self.bubbleDown(1)
        return retVal


    
