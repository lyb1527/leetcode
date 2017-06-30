"""
###########Linked List#########

"""
# E: Linked List Cycle
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
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
