# Remove duplicates from string

def removeDuplicates(str1):
    newString = str1[0]
    for char in str1[1:]:
        if char != newString[-1]:
            newString += char

    return newString

print(removeDuplicates('heelloo'))

def removeDuplicate(s):
    if (len(s)) < 2:
        return s

    result = []
    for i in s:
        if i not in result:
            result.append(i)

    return ''.join(result)


# String Homomorphism
'''
Q: chars mapping consistent


Given two strings, determine if isomorphic

Two strings are isomorphic if the chars in s can be replaced to get

For example,
Given "egg", "add", return true.

Given "foo", "bar", return false.

Given "paper", "title", return true.

'''
class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        first = {}
        second = {}
        for i in range(0,len(s)):
            if s[i] not in first:
                if t[i] not in second:
                    first[s[i]] = t[i]
                    second[t[i]] = s[i]
                else:
                    return False
            elif first[s[i]] != t[i]:
                return False
        return True

class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        mapping = {}
        used = set()
        for i in xrange(len(s)):
          if mapping.get(s[i]):
            if mapping[s[i]] != t[i]:
              return False
            continue
          if t[i] in used:
            return False
          mapping[s[i]] = t[i]
          used.add(t[i])
        return True

class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
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

class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        d1 = {}
        d2 = {}
        for i in range(len(s)):
            if s[i] in d1:
                if d1[s[i]] != t[i]:
                    return False
            else:
                d1[s[i]] = t[i]

            if t[i] in d2:
                if d2[t[i]] != s[i]:
                    return False
            else:
                d2[t[i]] = s[i]
        return True


# Check Word Abbreviation
'''
Given non-empty string and an abbreviation abbr, return whether the string matches
with the given Abbreviation


A string such as "word" contains only the following valid abbreviations:

["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1",
"w1r1", "1o2", "2r1", "3d", "w3", "4"]

Given s = "internationalization", abbr = "i12iz4n":
Return true.
Example 2:

Given s = "apple", abbr = "a2e":
Return false.

注意：

假设s只包含小写字母，abbr只包含小写字母和数字。

解题思路：
模拟题，遍历word和abbr即可

'''
class Solution:
    # @param {string} word a non-empty string
    # @param {string} abbr an abbreviation
    # @return {boolean} true if string matches with the given abbr or false
    def validWordAbbreviation(self, word, abbr):
        # Write your code here
        size = len(word)
        cnt = loc = 0
        for w in abbr:
            #checks whether the string consists of digits only.
            if w.isdigit():
                if w == '0' and cnt == 0:
                    return False
                cnt = cnt * 10 + int(w)
            else:
                loc += cnt
                cnt = 0
                if loc >= size or word[loc] != w:
                    return False
                loc += 1
        return loc + cnt == size


# Rectangle Overlap

class Solution:
    # @param {Point} l1 top-left coordinate of first rectangle
    # @param {Point} r1 bottom-right coordinate of first rectangle
    # @param {Point} l2 top-left coordinate of second rectangle
    # @param {Point} r2 bottom-right coordinate of second rectangle
    # @return {boolean} true if they are overlap or false
    def doOverlap(self, l1, r1, l2, r2):
        # Write your code here
        if l1.x > r2.x or l2.x > r1.x:
            return False

        if l1.y < r2.y or l2.y < r1.y:
            return False

        return True



# Decode ways
'''
message containing letters from A-Z is being ecoded to numbers using the following
mappping:


'A' -> 1
'B' -> 2
...
'Z' -> 26

given an encoed message containing digits, determine the total number of ways to
decode it

EX:
given message 12 it could be decoded as AB (12) or L (12)
the number of ways to decode 12 is 2
'''
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) == 0: return 0

        n = len(s)
        dp = [0 for i in range(n+1)]

        dp[0] = 1
        for i in range(1,n+1):
            if s[i-1] != "0":
                 dp[i] += dp[i-1]
            if i > 1 and s[i-2:i] < "27" and s[i-2:i] > "09":
                dp[i] += dp[i-2]

        return dp[-1]


class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return 0

        n = len(s)
        dp = [0] * (n+1)
        dp[0] = 1
        for i in range(1, n+1):
            if int(s[i-1]) != 0:
                dp[i] += dp[i-1]

            if i >= 2 and 10 <= int(s[i-2:i]) <= 26:
                dp[i] += dp[i-2]

        return dp[n]



# word Abbreviation - HARD
'''
利用字典dmap维护原始字符串word到压缩字符串abbr的映射

尝试将所有字符串从最短长度开始进行压缩

若同一个压缩字符串对应多个原串，则将这些串递归求解

否则，将该压缩串的结果加入dmap

'''


class Solution(object):

    def wordsAbbreviation(self, dict):
        """
        :type dict: List[str]
        :rtype: List[str]
        """
        self.dmap = {}
        self.solve(dict, 0)
        return map(self.dmap.get, dict)

    def abbr(self, word, size):
        if len(word) - size <= 3: return word
        return word[:size + 1] + str(len(word) - size - 2) + word[-1]

    def solve(self, dict, size):
        dlist = collections.defaultdict(list)
        for word in dict:
            dlist[self.abbr(word, size)].append(word)
        for abbr, wlist in dlist.iteritems():
            if len(wlist) == 1:
                self.dmap[wlist[0]] = abbr
            else:
                self.solve(wlist, size + 1)





# Mirror numbers
'''
Q: Look the same when rotated 180 degrees. EX: 69

The number is represented as a string

For example, the numbers "69", "88", and "818" are all mirror numbers.
Given num = "69" return true
Given num = "68" return false
'''
class Solution:
    # @param {string} num a string
    # @return {boolean} true if a number is strobogrammatic or false
    def isStrobogrammatic(self, num):
        # Write your code here
        mp = {"0":"0", "1":"1", "6":"9", "8":"8", "9":"6"}
        left, right = 0, len(num)-1
        while left <= right:
#get() is called, Python checks if the specified key exists in the dict. If it does,
 #then get() returns the value of that key. If the key does not exist, then get() returns the value specified in the second argument to get()
            if mp.get(num[left]) != num[right]:
                return False
            left += 1
            right -= 1
        return True


# Sliding Window Average From Data Stream
'''
Given a stream of integers and a window size, calculate the moving average
of all integers in the sliding window

MovingAverage m = new MovingAverage(3);
m.next(1) = 1 // return 1.00000
m.next(10) = (1 + 10) / 2 // return 5.50000
m.next(3) = (1 + 10 + 3) / 3 // return 4.66667
m.next(5) = (10 + 3 + 5) / 3 // return 6.00000

'''








class MovingAverage(object):

    def __init__(self, size):
        # Initialize your data structure here.
        from Queue import Queue
        self.queue = Queue()
        self.size = size
        self.sum = 0.0


    # @param {int} val an teger
    def next(self, val):
        # Write your code here
        self.sum += val
        if self.queue.qsize() == self.size:
            self.sum -= self.queue.get()
        self.queue.put(val)
        return self.sum * 1.0 / self.queue.qsize()



# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param = obj.next(val)



# System Longest File Path == longest absolute file path
'''
extract file system by a string in the following manner:
"dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" represents:
dir
    subdir1
    subdir2
        file.ext

"dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"
represents:

dir
    subdir1
        file1.ext
        subsubdir1
    subdir2
        subsubdir2
            file2.ext
 interested in finding the longest (number of characters) absolute path to a
  file within our file system. For example, in the second example above, the
  longest absolute path is "dir/subdir2/subsubdir2/file2.ext",

 return the length of the longest absolute path and if no file in the system, return 0


'''
class Solution(object):
    def lengthLongestPath(self, input):
        """
        :type input: str
        :rtype: int
        """
        tok = input.split('\n')
        depth = {0:0}
        maxl = 0

        for t in tok:
            l, name = self.lvl(t)
            if self.isFolder(name):
                depth[l+1] = depth[l] + 1 + len(name)
            else:
                length = depth[l] + len(name)
                maxl = max(maxl, length)

        return maxl


    def isFolder(self, s):
        t = s.split('.')
        if len(t) > 1:
            return False
        else:
            return True

    def lvl(self, s):
        name = s.lstrip('\t')
        lvl = len(s) - len(name)
        return lvl, name



class Solution(object):
    def lengthLongestPath(self, input):
        maxlen = 0
        pathlen = {0: 0}
        for line in input.splitlines():
            name = line.lstrip('\t')
            depth = len(line) - len(name)
            if '.' in name:
                maxlen = max(maxlen, pathlen[depth] + len(name))
            else:
                pathlen[depth + 1] = pathlen[depth] + len(name) + 1
        return maxlen



class Solution(object):
    def lengthLongestPath(self, input):
        """
        :type input: str
        :rtype: int
        """
        curpath = []
        out = 0
        for line in input.split('\n'):
            dep = line.count('\t')
            curpath[dep:] = [len(line) - dep]
            if '.' in line:
                out = max(out, sum(curpath) + len(curpath) - 1)
        return out


# String Serialization - Encode and decode strings
'''
Design an algorithm eo encode a list of strings to a string.
the encoded string is then sent over the network and is decoded back to the original list of strings.

Please implement encode and decode

EX:
Given strs = ["lint","code","love","you"]
string encoded_string = encode(strs)

return `["lint","code","love","you"]｀ when you call decode(encoded_string)

'''
class Solution():
    def encode(self, strs):
        return ''.join('%d:' % len(s) + s for s in strs)

    def decode(self, s):
        strs = []
        i = 0
        while i < len(s):
            j = s.find(':', i)
            i = j + 1 + int(s[i:j])
            strs.append(s[j+1:i])
        return strs


# Identify Celebrity
'''
suppose at a party with n people(labeled from 0 to n-1). There may exist one celebrity.

The definition of a celebrity is that all the other n - 1 people known him/her, but she /she does not know nay of them

Now find out who the celebrity is or verify that there is not one.
The only thing allowed to ask is 'Hi A, do you know B' get information of whehter A knows B .

Need to find out the celebrity by asking as few questions as possible

Given function, bool knows(a, b) which tells whether A knows B
Implement a function int findCelebrity(n). This func should minimize the calls to knows.


'''
def findCelebrity(self, n):
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if [knows(i,j), knows(j,i)] != [False, True]:
                break
        else:
            return i
    return -1


# O(n) calls , O(1) space
'''
Explanation

The first loop is to exclude n - 1 labels that are not possible to be a celebrity.
After the first loop, x is the only candidate.
The second and third loop is to verify x is actually a celebrity by definition.

The key part is the first loop. To understand this you can think the knows(a,b)
 as a a < b comparison, if a knows b then a < b, if a does not know b, a > b.
 Then if there is a celebrity, he/she must be the "maximum" of the n people.

However, the "maximum" may not be the celebrity in the case of no celebrity at
 all. Thus we need the second and third loop to check if x is actually celebrity
  by definition.

The total calls of knows is thus 3n at most. One small improvement is that in
 the second loop we only need to check i in the range [0, x). You can figure
  that out yourself easily
'''

# O(n)
def findCelebrity(self, n):

    # assume celebrity 0
    x = 0
    # for loop to find the only candidate, n - 1
    for i in xrange(n):
        if knows(x, i):
            x = i

    # below verify x is a celeb by definition.
    # max may not be the celebrity in case of no celebrity at all.


    # [0, x)
    if any(knows(x, i) for i in xrange(x)):
        return -1


    if any(not knows(i, x) for i in xrange(n)):
        return -1
    return x



class Solution:
    # @param {int} n a party with n people
    # @return {int} the celebrity's label or -1
    def findCelebrity(self, n):
        # Write your code here
        candidate = 0
        for i in xrange(n):
            if Celebrity.knows(candidate, i):
                candidate = i

        for i in xrange(candidate):
            if Celebrity.knows(candidate, i) or not Celebrity.knows(i, candidate):
                return -1

        for i in xrange(candidate + 1, n):
            if not Celebrity.knows(i, candidate):
                return -1

        return candidate



## Edit Distnace
'''
Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. (each operation is counted as 1 step.)

You have the following 3 operations permitted on a word:

Insert a character
Delete a character
Replace a character

'''
def one_away(s1, s2):
    if len(s1) == len(s2):
        return one_edit_replace(s1, s2)
    elif len(s1) + 1 == len(s2):
        return one_edit_insert(s1, s2)
    elif len(s1) - 1 == len(s2):
        return one_edit_insert(s2, s1)
    return False



def one_edit_replace(s1, s2):
    edited = False
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            #if edited:
            #   return False
            edited = True
    return True



def one_edit_insert(s1, s2):
    edited = False
    i, j = 0, 0

    while i < len(s1) and j < len(s2):
        if s1[i] != s2[j]:
            if edited:
                return False
            edited = True
            j += 1
        else:
            i += 1
            j += 1
    return True




class Solution(object):
    def minDistance(self, word1, word2):

        #O(m*n)
        #dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(word1[i-1]!=word2[j-1]))

        # improving space from O(m*n)to  O(n) using rolling array
        l1, l2 = len(word1)+1, len(word2)+1
        dp = range(l2)
        pre = 0
        for i in xrange(1, l1):
            pre, dp[0] = i-1, i
            for j in xrange(1, l2):
                buf = dp[j]
                dp[j] = min(dp[j]+1, dp[j-1]+1, pre+(word1[i-1]!=word2[j-1]))
                pre = buf
        return dp[-1]

# Edit Distance II - One distance apart
'''
Given two strings S and T, determine if they are both one edit Distance apart

'''
class Solution:
    # @param {string} s a string
    # @param {string} t a string
    # @return {boolean} true if they are both one edit distance apart or false
    def isOneEditDistance(self, s, t):
        # Write your code here
        m = len(s)
        n = len(t)
        if abs(m - n) > 1:
            return False

        if m > n:
            return self.isOneEditDistance(t, s)

        for i in xrange(m):
            if s[i] != t[i]:
                if m == n:
                    return s[i + 1:] == t[i + 1:]
                return s[i:] == t[i + 1:]

        return m != n

# Roman to integer
'''
Given a roman numeral, convert it to an integer.

ANswer is guarantee to be within the range from 1 to 3999

Example
IV -> 4

XII -> 12

XXI -> 21

XCIX -> 99

'''


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



# Integer to Roman
class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        n = num
        d = {1:["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII","IX"], 2:["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"], 3:["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"], 4:["","M", "MM", "MMM"]}
        res=""
        divisor = 10
        counter = 1
        while n!=0:
            #print res
            res=d[counter][n%divisor]+res
            n/=divisor
            counter+=1

        return res


class Solution:
    # @return a string
    def intToRoman(self, num):
        val = [
            1000, 900, 500, 400,
            100, 90, 50, 40,
            10, 9, 5, 4,
            1
            ]
        syb = [
            "M", "CM", "D", "CD",
            "C", "XC", "L", "XL",
            "X", "IX", "V", "IV",
            "I"
            ]
        roman = ''
        i = 0
        while  num > 0:
            for _ in range(num // val[i]):
                roman += syb[i]
                num -= val[i]
            i += 1
        return roman


class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        mapping = {1000: 'M',
                  900: 'CM',
                  500: 'D',
                  400: 'CD',
                  100: 'C',
                  90: 'XC',
                  50: 'L',
                  40: 'XL',
                  10:'X',
                  9: 'IX',
                  5: 'V',
                  4: 'IV',
                  1: 'I'}
        letter = self.floor_key(mapping, num)
        if letter == num:
            return mapping[letter]
        else:
            return mapping[letter] + self.intToRoman(num - letter)

    def floor_key(self, d, key):
        if key in d:
            return key
        return max(k for k in d if k < key)


class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        d = [
            (1000,'M'),
            (900, 'CM'),
            (500, 'D'),
            (400, 'CD'),
            (100, 'C'),
            (90, 'XC'),
            (50, 'L'),
            (40, 'XL'),
            (10, 'X'),
            (9, 'IX'),
            (5, 'V'),
            (4, 'IV'),
            (1, 'I')
        ]
        result = []
        for count, c in d:
            while num >= count:
                num -= count
                result.append(c)
            if num == 0:
                break

        return ''.join(result)



# Read N chars Given Read 4



# Read Chars From File - Multiple calls
'''

buffer uses a queue, because the order has to be the same.

If queue empty, enter the queue

if queue non empty, exit the queue and put elements in the answer



'''
class Solution:
# @param buf, Destination buffer (a list of characters)
# @param n,   Maximum number of characters to read (an integer)
# @return     The number of characters read (an integer)
def __init__(self):
    self.queue = []

def read(self, buf, n):
    idx = 0
    while True:
        buf4 = [""]*4
        l = read4(buf4)
        self.queue.extend(buf4)
        curr = min(len(self.queue), n-idx)
        for i in xrange(curr):
            buf[idx] = self.queue.pop(0)
            idx+=1
        if curr == 0:
            break
    return idx




def __init__(self):
    self.queue = [] # global "buffer"

def read(self, buf, n):
    idx = 0

    # if queue is large enough, read from queue
    while self.queue and n > 0:
        buf[idx] = self.queue.pop(0)
        idx += 1
        n -= 1

    while n > 0:
        # read file to buf4
        buf4 = [""]*4
        l = read4(buf4)

        # if no more char in file, return
        if not l:
            return idx

        # if buf can not contain buf4, save to queue
        if l > n:
            self.queue += buf4[n:l]

        # write buf4 into buf directly
        for i in range(min(l, n)):
            buf[idx] = buf4[i]
            idx += 1
            n -= 1
    return idx










#-------------------------------------------------------------------------------
'''

3

'''

# first Position unique character
'''
Given a string, find the first non-repeating char in it and return its
index. If does not exist, return 0-1

'''

class Solution:
    # @param {string} s a string
    # @return {int} it's index
    def firstUniqChar(self, s):
        # Write your code here
        alp = {}
        for c in s:
            if c not in alp:
                alp[c] = 1
            else:
                alp[c] += 1

        index = 0
        for c in s:
            if alp[c] == 1:
                return index
            index += 1
        return -1



# Substring Anagrams --  Find all anagrams in a string(leetcode)
'''
Given a string s and a non-empty string p, find all the start indices of p's
anagrams in s.
'''

'''
Hash the number of times each character appears in p. Iterate over s with a
 sliding window and maintain a similar hash. If these two hashes are ever the
 same, add that to the result.

Each of the hashes have a finite (a-z, A-Z) number of possible characters,
so the space used is O(1)

We iterate over s linearly, comparing constant length hashes at each iteration
so each iteration is also O(1), so the runtime is O(n)
'''
class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        res = []
        n, m = len(s), len(p)
        if n < m: return res
        phash, shash = [0]*123, [0]*123
        for x in p:
            phash[ord(x)] += 1
        for x in s[:m-1]:
            shash[ord(x)] += 1
        for i in range(m-1, n):
            shash[ord(s[i])] += 1
            if i-m >= 0:
                shash[ord(s[i-m])] -= 1
            if shash == phash:
                res.append(i - m + 1)
        return res







# valid Parentheses
'''
Given a string containing just the chars '(', ')', '{', '}', '[' and ']'

determine if the input is valid

The brackets must close in the correct order.
"()" and "()[]{}" are all valid but "(]" and "([)]" are not.
'''

class Solution(object):
    '''
    题意：输入一个只包含括号的字符串，判断括号是否匹配
    模拟堆栈，读到左括号压栈，读到右括号判断栈顶括号是否匹配
    '''
    def isValidParentheses(self, s):
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
                if ch == ']' and stack[-1] != '[' or ch == ')' and stack[-1] != '(' or ch == '}' and stack[-1] != '{':
                    return False
                # 弹栈
                stack.pop()
        return not stack



# Merge Intervals
'''
GIven a collection of intervals, merge all overlapping intervals

Given intervals => merged intervals:

[                     [
  [1, 3],               [1, 6],
  [2, 6],      =>       [8, 10],
  [8, 10],              [15, 18]
  [15, 18]            ]
]

'''
class Solution:
    # @param intervals, a list of Interval
    # @return a list of Interval
    def merge(self, intervals):
        intervals = sorted(intervals, key=lambda x: x.start)
        result = []
        for interval in intervals:
            if len(result) == 0 or result[-1].end < interval.start:
                result.append(interval)
            else:
                result[-1].end = max(result[-1].end, interval.end)
        return result



# Insert Inteval
'''
Given Non-overlaping interval list which is sorted by start point

Insert a new interval into it. make sure the list is still in order an
Non-overlaping (merge if necceessary)

Insert [2, 5] into [[1,2], [5,9]], we get [[1,9]].

Insert [3, 4] into [[1,2], [5,9]], we get [[1,2], [3,4], [5,9]].

'''
class Solution:
    """
    Insert a new interval into a sorted non-overlapping interval list.
    @param intevals: Sorted non-overlapping interval list
    @param newInterval: The new interval.
    @return: A new sorted non-overlapping interval list with the new interval.
    """
    def insert(self, intervals, newInterval):
        results = []
        insertPos = 0
        for interval in intervals:
            if interval.end < newInterval.start:
                results.append(interval)
                insertPos += 1
            elif interval.start > newInterval.end:
                results.append(interval)
            else:
                newInterval.start = min(interval.start, newInterval.start)
                newInterval.end = max(interval.end, newInterval.end)
        results.insert(insertPos, newInterval)
        return results








# Word Abbreviation Set
'''


'''




# Missing Interval
'''
Given a sorted integer array where the range of the elements are in the inclusive
range [lower, uppper], return its missing ranges

Given nums = [0, 1, 3, 50, 75], lower = 0 and upper = 99
return ["2", "4->49", "51->74", "76->99"].

'''

class Solution:
    # @param {int[]} nums a sorted integer array
    # @param {int} lower an integer
    # @param {int} upper an integer
    # @return {string[]} a list of its missing ranges
    def findMissingRanges(self, nums, lower, upper):
        # Write your code here
        results = []
        next = lower
        for k,v in enumerate(nums):
            if nums[k] < next:
                continue
            if nums[k] == next:
                next += 1
                continue
            results.append(self.getRange(next, nums[k] - 1))
            next = nums[k] + 1
        if next <= upper:
            results.append(self.getRange(next, upper))
        return results

    def getRange(self, l, u):
        return str(l) if l == u else str(l) + "->" + str(u)



# Load Balancer
'''
Implement a load Balancer for web servers. Provides the following funcs:

* add  a new server to the cluster, add(server_id)

* remove a bad server from the cluster, remove(server_id)

* pick a server in the cluster randomly with equal probability. pick()

EX:
At beginning, the cluster is empty => {}.

add(1)
add(2)
add(3)
pick()
>> 1         // the return value is random, it can be either 1, 2, or 3.
pick()
>> 2
pick()
>> 1
pick()
>> 3
remove(1)
pick()
>> 2
pick()
>> 3
pick()
>> 3


'''
class LoadBalancer:

    def __init__(self):
        self.server_ids = []
        self.id2index = {}

    # @param {int} server_id add a new server to the cluster
    # @return nothing
    def add(self, server_id):
        if server_id in self.id2index:
            return
        self.server_ids.append(server_id)
        self.id2index[server_id] = len(self.server_ids) - 1

    # @param {int} server_id remove a bad server from the cluster
    # @return nothing
    def remove(self, server_id):
        if server_id not in self.id2index:
            return

        # remove the server_id
        index = self.id2index[server_id]
        del self.id2index[server_id]

        # overwrite the one to be removed
        last_server_id = self.server_ids[-1]
        self.id2index[last_server_id] = index
        self.server_ids[index] = last_server_id
        self.server_ids.pop()

    # @return {int} pick a server in the cluster randomly with equal probability
    def pick(self):
        import random
        index = random.randint(0, len(self.server_ids) - 1)
        return self.server_ids[index]



# Longest consecutive Sequence
'''
Given an  unsorted array of integers, find the length of the longest consecutive elements sequence

SHOULD RUN in O(n)

Example
Given [100, 4, 200, 1, 3, 2],
The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.

'''


'''
First turn the input into a set of numbers. That takes O(n) and then we can ask
 in O(1) whether we have a certain number.

Then go through the numbers. If the number x is the start of a streak (i.e.,
 x-1 is not in the set), then test y = x+1, x+2, x+3, ... and stop at the first
  number y not in the set. The length of the streak is then simply y-x and we
  update our global best with that. Since we check each streak only once, this
   is overall O(n). This ran in 44 ms on the OJ, one of the fastest Python submissions.
'''

class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # start with the smallest
        nums = set(nums)
        maxlength = 0
        for n in nums:
            if n+1 not in nums:
                y = n-1
                while y in nums:
                    y -= 1
                maxlength = max(maxlength, n-y)
        return maxlength



class Solution:
    # @param num, a list of integer
    # @return an integer
    def longestConsecutive(self, num):
        num.sort()
        l = num[0]
        ans = 1
        tmp = 1
        for n in num:
            if(n - l == 0):
                continue;
            elif(n - l == 1):
                tmp += 1
            else:
                if tmp > ans:
                    ans = tmp
                tmp = 1
            l = n
        if tmp > ans:
            ans = tmp
        return ans




# Longest Increasing Subsequence
'''
Given a sequence of integers, find the longest increasing subsequence

Return the length of the LIS

LIS in sorted order from lowest to highest. The sequence is not necessarily
contiguous or unique

For [5, 4, 1, 2, 3], the LIS is [1, 2, 3], return 3
For [4, 2, 4, 5, 3, 7], the LIS is [2, 4, 5, 7], return 4


First do it in O(n^2)


FOLLOW-UP: improve it to O(nlogn)

'''

# DP : O(N^2)

class Solution(object):
#using dP
    def lengthOfLIS1(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        dp = [1]*len(nums)
        for i in range (1, len(nums)):
            for j in range(i):
                if nums[i] >nums[j]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)



# this is O(NlogN)
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        tails = [0] * len(nums)
        size = 0
        for x in nums:
            i, j = 0, size
            while i != j:
                m = (i + j) / 2
                if tails[m] < x:
                    i = m + 1
                else:
                    j = m
            tails[i] = x
            size = max(i + 1, size)
        return size








#############  3   ####################



# first position unqiue character



class Solution:
    # @param {string} s a string
    # @return {int} it's index
    def firstUniqChar(self, s):
        # Write your code here
        dict = {}
        for i in s:
            if  i in dict:
                dict[i] += 1
            else:
                dict[i] = 1

        for i in range(len(s)):
            if dict[s[i]] == 1:
                return i

        return -1


// version: 高频题班
public class Solution {
    /**
     * @param s a string
     * @return it's index
     */
    public int firstUniqChar(String s) {
        // Write your code here
        int[] cnt = new int[256];

        for (char c : s.toCharArray()) {
            cnt[c]++;
        }

        for (int i = 0; i < s.length(); i++) {
            if (cnt[s.charAt(i)] == 1) {
                return i;
            }
        }
        return -1;
    }
}



# Substring Anagrams
'''
Given string s and a non-empty string p, find all the start inices of p's anagrams in s

'''



'''

Hash the number of times each character appears in p. Iterate over s with a
sliding window and maintain a similar hash. If these two hashes are ever the same, add that to the result.

Each of the hashes have a finite (a-z, A-Z) number of possible characters, so the space used is O(1)

We iterate over s linearly, comparing constant length hashes at each iteration
 so each iteration is also O(1), so the runtime is O(n)
'''
class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        res = []
        n, m = len(s), len(p)
        if n < m: return res
        phash, shash = [0]*123, [0]*123
        for x in p:
            phash[ord(x)] += 1
        for x in s[:m-1]:
            shash[ord(x)] += 1
        for i in range(m-1, n):
            shash[ord(s[i])] += 1
            if i-m >= 0:
                shash[ord(s[i-m])] -= 1
            if shash == phash:
                res.append(i - m + 1)
        return res



class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        result = []
        n, m = len(s), len(p)
        if n < m:
            return []
        s_hash, p_hash = [0] * 123, [0] * 123
        # init the targeted matching windwo
        for i in p:
            p_hash[ord(i)] += 1
        # init the matched substring in s, with only the len of (n-1), will
        #checking the last element in the for loop later.
        for i in s[:m - 1]:
            s_hash[ord(i)] += 1

        # sliding window from s[m-1] until the end. If the s_hash and p_hash
        # are ever the same, log out the index. Each iter, substract the front and the next iter will add the next element.
        for i in range(m - 1, n):
            s_hash[ord(s[i])] += 1
            if i >= m:
                s_hash[ord(s[i - m])] -= 1
            if s_hash == p_hash:
                result.append(i - m + 1)
        return result


# JIUZHANG
class Solution:
    # @param {string} s a string
    # @param {string} p a non-empty string
    # @return {int[]} a list of index
    def findAnagrams(self, s, p):
        # Write your code here
        ans = []
        sum = [0 for x in range(0,30)]
        plength = len(p)
        slength = len(s)
        for i in range(plength):
            sum[ord(p[i]) - ord('a')] += 1
        start = 0
        end = 0
        matched = 0
        while end < slength:
            if sum[ord(s[end]) - ord('a')] >= 1:
                matched += 1
            sum[ord(s[end]) - ord('a')] -= 1
            end += 1
            if matched == plength:
                ans.append(start)
            if end - start == plength:
                if sum[ord(s[start]) - ord('a')] >= 0:
                    matched -= 1
                sum[ord(s[start]) - ord('a')] += 1
                start += 1
        return ans




# Valid Parentheses
class Solution(object):
    '''
    题意：输入一个只包含括号的字符串，判断括号是否匹配
    模拟堆栈，读到左括号压栈，读到右括号判断栈顶括号是否匹配
    '''
    def isValidParentheses(self, s):
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
                if ch == ']' and stack[-1] != '[' or ch == ')' and stack[-1] != '(' or ch == '}' and stack[-1] != '{':
                    return False
                # 弹栈
                stack.pop()
        return not stack

class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        brackets = {')':'(', '}':'{', ']':'['}
        for i in s:
            if i in brackets:
                if  stack == [] or stack.pop() != brackets[i]:
                    return False
            else:
                stack.append(i)
        return not stack


# Merge Intervals

'''
Sort the list first. Check if the new interval overlaps with the previous one
 in the output list. If yes, update it. Otherwise, append the new one.

'''
class Solution:
    # @param intervals, a list of Interval
    # @return a list of Interval
    def merge(self, intervals):
        intervals = sorted(intervals, key=lambda x: x.start)
        result = []
        for interval in intervals:
            if len(result) == 0 or result[-1].end < interval.start:
                result.append(interval)
            else:
                result[-1].end = max(result[-1].end, interval.end)
        return result


# Insert Intervals

class Solution:
    """
    Insert a new interval into a sorted non-overlapping interval list.
    @param intevals: Sorted non-overlapping interval list
    @param newInterval: The new interval.
    @return: A new sorted non-overlapping interval list with the new interval.
    """
    def insert(self, intervals, newInterval):
        results = []
        insertPos = 0
        for interval in intervals:
            if interval.end < newInterval.start:
                results.append(interval)
                insertPos += 1
            elif interval.start > newInterval.end:
                results.append(interval)
            else:
                newInterval.start = min(interval.start, newInterval.start)
                newInterval.end = max(interval.end, newInterval.end)
        results.insert(insertPos, newInterval)
        return results



# Word Abbreviation set
class ValidWordAbbr:

   def __init__(self, dictionary):
        # Write your code here
        self.d = collections.defaultdict(set)
        for word in dictionary:
            self.d[self.getAbbr(word)].add(word)


    # @param {string} word a string
    # @return {boolean} true if its abbreviation is unique or false
    def isUnique(self, word):
        # Write your code here
        key = self.getAbbr(word)
        if key not in self.d:
            return True
        return len(self.d[key]) == 1 and (word in self.d[key])

    def getAbbr(self, word):
        if len(word) > 2:
            return word[0] + str(len(word) - 2) + word[-1]
        return word



# Missing Interval
'''
Given nums = [0, 1, 3, 50, 75], lower = 0 and upper = 99
return ["2", "4->49", "51->74", "76->99"].
'''
class Solution:
    # @param {int[]} nums a sorted integer array
    # @param {int} lower an integer
    # @param {int} upper an integer
    # @return {string[]} a list of its missing ranges
    def findMissingRanges(self, nums, lower, upper):
        # Write your code here
        results = []
        next = lower
        for k,v in enumerate(nums):
            if nums[k] < next:
                continue
            if nums[k] == next:
                next += 1
                continue
            results.append(self.getRange(next, nums[k] - 1))
            next = nums[k] + 1
        if next <= upper:
            results.append(self.getRange(next, upper))
        return results

    def getRange(self, l, u):
        return str(l) if l == u else str(l) + "->" + str(u)


# Load Balancer



# Longest Conseutive sequence
'''
Given [100, 4, 200, 1, 3, 2],
The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.

'''
class Solution:
    """
    @param num, a list of integer
    @return an integer
    """
    def longestConsecutive(self, nums):
        # write your code here
        nums = set(nums)
        maxlength = 0
        for n in nums:
            if n+1 not in nums:
                y = n-1
                while y in nums:
                    y -= 1
                maxlength = max(maxlength, n-y)
        return maxlength





#########################   4      #################


# Guess Number Game
'''
Pick a number from 1 to n. Guess which number is picked.

will tell whether the number is hihger or lower
could call a pre-defined API guess(int num), which retusn 3 possible results(-1, 1, 0)
'''
class Solution:
    # @param {int} n an integer
    # @return {int} the number you guess
    def guessNumber(self, n):
        # Write your code here
        l, r = 1, n
        while l <= r:
            mid = l + (r - l) / 2
            res = Guess.guess(mid)
            if res == 0:
                return mid
            if res == -1:
				r = mid - 1
            else:
                l = mid + 1
        return -1



# Convert BST to Greater Tree
'''
Given a BST, convert it to a greater tree such that every key of the original
 BST is changed to the original key plus sum of all keys greater

Every key in the original key  = original key + sum(all keys greater than the orignal key )


'''


'''

Since this is a BST, we can do a reverse inorder traversal to traverse the
 nodes of the tree in descending order. In the process, we keep track of the running sum of all nodes which we have traversed thus far.
'''
class Solution(object):
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        self.sum = 0
        self.traversal(root)
        return root

    def traversal(self, node):
        if not node:
            return None
        if node.right:
            self.traversal(node.right)
        self.sum += node.val
        node.val = self.sum
        if node.left:
            self.traversal(node.left)



# Iterative
class Solution(object):
    def convertBST(self, root):

        stack = []
        n = root
        s = 0
        while n or stack:
            if n:
                stack.append(n)
                n = n.right
            else:
                n = stack.pop()
                n.val += s
                s = n.val
                n = n.left
        return root


# Binary Tree Vertical Orer traversal
'''
Given a BT, return the vertical order traversal of its nodes' values.
From top to bottom, column by column

If two nodes are in the same row and same column, hte order should be from left to right

'''

class Solution:
    # @param {TreeNode} root the root of binary tree
    # @return {int[][]} the vertical order traversal
    def verticalOrder(self, root):
        # Write your code here
        results = collections.defaultdict(list)
        import Queue
        queue = Queue.Queue()
        queue.put((root, 0))
        while not queue.empty():
            node, x = queue.get()
            if node:
                results[x].append(node.val)
                queue.put((node.left, x - 1))
                queue.put((node.right, x + 1))

        return [results[i] for i in sorted(results)]



# Binary Order leaves Order Traversal

'''

Given BT, collect a tree's nodes as if they were doing this:
collect and remove all leaves, repeat unitl tree is empty

          1
         / \
        2   3
       / \
      4   5
Returns [[4, 5, 3], [2], [1]].

'''
class Solution:
    # @param {TreeNode} root the root of binary tree
    # @return {int[][]} collect and remove all leaves
    def findLeaves(self, root):
        # Write your code here
        results = []
        self.dfs(root, results)
        return results

    def dfs(self, node, results):
        if node is None:
            return 0

        level = max(self.dfs(node.left, results), self.dfs(node.right, results)) + 1
        size = len(results)
        if level > size:
            results.append([])

        results[level - 1].append(node.val)
        return level



# Binary Tree Flip
'''
given a BT where all the right nodes are either leaf nodes
with a sibling( a left node that shares the same parent node) or empty
Flip it upside down and turn it into a tree where the original right noes turned into the left leaf nodes. ert
return new root


'''

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



 class Solution:
        # @param root, a tree node
        # @return root of the upside down tree
        def upsideDownBinaryTree(self, root):
            # take care of the empty case
            if not root:
                return root
            # take care of the root
            l = root.left
            r = root.right
            root.left = None
            root.right = None
            # update the left and the right children, form the new tree, update root
            while l:
                newL = l.left
                newR = l.right
                l.left = r
                l.right = root
                root = l
                l = newL
                r = newR
            return root



# Inorder Successor In binary Search Tree

'''
Given a BST and a node in it, find the in-order successor of that noe in the BSET.

If if the node has no in-order successor in the tree, return null

Given tree = [2,1] and node = 1:

  2
 /
1

Given tree = [2,1,3] and node = 2:

  2
 / \
1   3
return node 3.

'''
# version: 高频题班
public class Solution {
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        // write your code here
        if (root == null || p == null) {
            return null;
        }

        if (root.val <= p.val) {
            return inorderSuccessor(root.right, p);
        } else {
            TreeNode left = inorderSuccessor(root.left, p);
            return (left != null) ? left : root;
        }
    }
}



class Solution(object):
    """
    @param root <TreeNode>: The root of the BST.
    @param p <TreeNode>: You need find the successor node of p.
    @return <TreeNode>: Successor of p.
    """
    def inorderSuccessor(self, root, p):
        if root is None or p is None:
            return None

        successor = None
        while root is not None and root.val != p.val:
            if root.val > p.val:
                successor = root
                root = root.left
            else:
                root = root.right

        if root is None:
            return None

        if root.right is None:
            return successor

        root = root.right
        while root.left is not None:
            root = root.left

        return root







# Search for a range


class Solution:
    """
    @param A : a list of integers
    @param target : an integer to be searched
    @return : a list of length 2, [index1, index2]
    """
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
        if A[end] == target:
            rightBound = end
        else:
            rightBound = start
        return [leftBound, rightBound]






################################# 5 ##############
