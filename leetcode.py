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
