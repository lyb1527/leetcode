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
