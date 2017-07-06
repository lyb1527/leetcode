"""
###########Linked List#########

"""
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


# fast runner and slow runner technique
def hasCycle(head):
    if head is None or head.next is None:
        return False
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        #If faster runner catches slow runner some time, it means linked list has a circle.
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
def getIntersectionNoe(headA, headB):
    nodeList = set()
    curr = headA
    while(curr != None):
        nodeList.add(curr)
        cur = cur.next

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
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        newList = []

        while head:
            newList.append(head.val)
            head = head.next

        length = len(newList) / 2

        for i in range(length):
            if newList[i] != newList[-1-i]:
                return False

        return True


def isPalindrome(self, head):
    """
    :type head: ListNode
    :rtype: bool
    """
    rev = None
    slow = fast = head
    while fast and fast.next:
        fast = fast.next.next
        rev, rev.next, slow = slow, rev, slow.next
    if fast:
        slow = slow.next
    while rev and rev.val == slow.val:
        slow = slow.next
        rev = rev.next
    return not rev

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
    if k == 0:
        return head
    if head == None:
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



# Sort List
'''
sort a LL in O(nlogn0) using constant space complexity
'''
class Solution(object):
    def merge(self, head1, head2):
        head = ListNode(0)
        current = head

        while head1 and head2:
            if head1.val <= head2.val:
                current.next = head1
                head1 = head1.next
            else:
                current.next = head2
                head2 = head2.next
            current = current.next

        if head1:
            current.next = head1
        if head2:
            current.next = head2

        return head.next

    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None or head.next is None:
            return head

        slow = fast = head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next

        head1 = head
        head2 = slow.next
        slow.next = None

        return self.merge(self.sortList(head1), self.sortList(head2))



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

class Solution:
    head = None
    # @param head, a list node
    # @return a tree node
    def sortedListToBST(self, head):
        current, length = head, 0
        while current is not None:
            current, length = current.next, length + 1
        self.head = head
        return self.sortedListToBSTRecu(0, length)

    def sortedListToBSTRecu(self, start, end):
        if start == end:
            return None
        mid = start + (end - start) / 2
        left = self.sortedListToBSTRecu(start, mid)
        current = TreeNode(self.head.val)
        current.left = left
        self.head = self.head.next
        current.right = self.sortedListToBSTRecu(mid + 1, end)
        return current



# Reverse Linked List II
'''
Reverse a linked list from position m to n. Do it in-place and in one pass
For example:
Given 1->2->3->4->5->NULL, m = 2 and n = 4,

return 1->4->3->2->5->NULL.

'''



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


class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        if not head or m == n:
            return head

        dumy = ListNode(0)
        dumy.next = head
        pre = dumy


        # find the start node to be reversed
        for _ in xrange(m-1):
            pre = pre.next


        start, tail = pre.next, pre.next

        end = None
        for _ in xrange(n-m):
            node = start
            start = start.next
            node.next = end
            end = node

        pre.next = start
        tail.next = start.next
        start.next = end

        return dumy.next




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
class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        h1 = l1 = ListNode(0)
        h2 = l2 = ListNode(0)
        while head:
            if head.val < x:
                l1.next = head
                l1 = l1.next
            else:
                l2.next = head
                l2 = l2.next
            head = head.next
        l2.next = None
        l1.next = h2.next
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
        carry = 0
        root = n = ListNode(0)
        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:
                v1 = l1.val
                l1 = l1.next
            if l2:
                v2 = l2.val
                l2 = l2.next
            carry, val = divmod(v1+v2+carry, 10)
            n.next = ListNode(val)
            n = n.next
        return root.next

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        '''
            6/29/17: Solution 1
        '''
        # Solution 1 (99.75%):
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
            even.next = odd.next.next
            odd = odd.next
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

        tail = 0
        for i in range(1, len(nums)):
            if nums[tail] != nums[i]:
                tail += 1
                nums[tail] = nums[i]

        return tail + 1
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

        length = 0
        for temp in nums:
            if temp != val:
                nums[length] = temp
                length += 1

        return length

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
# Max Consecutive Ones

# Shortest Unsorted Continuous Subarray


# Third Maximum Number

# Maximum Distance in arrays

# Shortest Word Distance

# best time to buy and sell stock Ii

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

# Pascal's Triangle Ii

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

def sortColors(nums):
    """
    :type nums: List[int]
    :rtype: void Do not return anything, modify nums in-place instead.
    """
    if not nums: return
    i, j = 0, 0
    for k in range(len(nums)):
        v = nums[k]
        nums[k] = 2
        if v < 2:
            nums[j] = 1
            j += 1
        if v == 0:
            nums[i] = 0
            i += 1

a = [1, 2, 0, 0, 1, 2, 2, 1]
sortColors(a)

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
