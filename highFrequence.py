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
