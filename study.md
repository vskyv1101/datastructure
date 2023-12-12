### 9주
```python
# 코드 6.1: 단순연결노드 클래스               # LinkedStack

class Node:                                 # 노드의 생성자

    def __init__ (self, elem, link=None):   # link는 디폴트 값으로 None 사용

        self.data = elem                    # 노드의 데이터

        self.link = link                    # 다음 노드를 가리키는 링크



# 코드 6.2: 연결된 스택 클래스

class LinkedStack :                         # 용량을 지정할 필요 없음

    def __init__( self ):                   # 연결된 스택의 생성자

        self.top = None                     # 데이터 멤버는 시작 노드를 가리키는 top 뿐임



    def isEmpty( self ):                    # 공백일 떄,

        return self.top == None             # top이 None이면 공백



    def isFull( self ):                     # 포화상태일 때,

        return False                        # 포화상태는 의미 없음. 항상 False 반환



    def push( self, item ):                 # 삽입연산

        self.top = Node(item, self.top)     # 요소 item을 이용해 노드를 구성, 링크를 top으로 연결하며느 이 노드가 top이 됨

        """

        n = Node(item)                      # 입력데이터 e를 이용해 새로운 노드 n을 생성

        n.link = self.top                   # n의 링크가 시작 노드를 가리키도록 함

        self.top = n                        # top이 n을 가리키도록 함

        """



    def peek( self ):                       # 참조연산

        if not self.isEmpty():              # 공백이 아니라면,

            return self.top.data            # top의 데이터 반환



    def pop( self ):                        # 삭제연산

        if not self.isEmpty():              # 공백검사 (삭제연산은 공백검사가 먼저 필요)

            data = self.top.data            # n이 현재 상단을 가리키게 하고

            self.top = self.top.link        # 이제 상단 다음 노드가 top이 되고

            return data                     # data 반환



    # 코드 6.3: 연결된 스택의 전체 요소의 수 계산

    def size( self ):

        node = self.top

        count = 0

        while not node == None :            # None이 아니면

            node = node.link                # 다음 노드로 이동

            count += 1                      # count + 1



    # 코드 6.4: 문자열 변환을 위한 str 연산자 중복

    def __str__(self):

        arr = []                            # 요소들을 저장할 공백 리스트 준비

        node = self.top

        while not node == None :

            arr.append(node.data)           # 각 노드의 데이터를 리스트에 추가

            node = node.link

        return str(arr)                     # 리스트를 문자열로 변환해 반환
```
```python
# Linked List.

class Node:

    def __init__ (self, elem, next=None):   # 노드의 생성자, next는 디폴트 값으로 None 사용

        self.data = elem                    # 노드의 데이터

        self.link = next                    # 다음 노드를 가리키는 링크



# 코드 6.5: 연결리스트 클래스

class LinkedList:                           

    # 리스트의 데이터: 생성자에서 정의 및 초기화

    def __init__( self ):                   # 연결리스트 생성자, 용량을 지정할 필요 없음

        self.head = None                    # 데이터 멤버로 시작 노드를 가리키는 head를 가짐



    # 리스트의 연산: 클래스의 메소드

    def isEmpty( self ): return self.head == None   # head가 None이면 공백상태

    def isFull( self ): return False                # 포화상태가 될 수는 없음



    def getNode(self, pos) :

        if pos < 0 : return None            

        node = self.head;                   # 시작 노드에서부터 

        while pos > 0 and node != None :    # pos번 링크를 따라 움직이면,

            node = node.link                

            pos -= 1

        return node                         # pos 위치의 노드에 도착



    def getEntry(self, pos) :               

        node = self.getNode(pos)            # pos 위치의 노드를 먼저 구한 후,

        if node == None : return None       

        else : return node.data             # 노드의 데이터 필드를 반환함



    def insert(self, pos, elem) :

        before = self.getNode(pos-1)        # pos-1위치의 노드 before를 먼저 구함

        if before == None :                 # 시작 위치에 삽입하느 상황

            self.head = Node(elem, self.head)

        else :

            node = Node(elem, before.link)  # 노드 생성,

            before.link = node              # 이전 노드가 새로운 노드를 가리키게 설정



    def delete(self, pos) :

        before = self.getNode(pos-1)        # pos-1 위치의 노드 before를 먼저 구함

        if before == None :         # 맨 앞 노드를 삭제

            if self.head is not None :

                self.head = self.head.link

        elif before.link != None :

            before.link = before.link.link  # before의 link가 삭제할 노드의 다음 노드를 가리키도록 함



    # 추가 연산들

    def size( self ) :

        node = self.head;

        count = 0;

        while node is not None :

            node = node.link

            count += 1

        return count



    def __str__( self ) :

        arr = []

        node = self.head

        while node is not None :

            arr.append(node.data)

            node = node.link

        return str(arr)

    



    def replace(self, pos, elem) :

        node = self.getNode(pos)

        if node != None : node.data = elem



    def find(self, val) :

        node = self.head;

        while node is not None:

            if node.data == val : return node

            node = node.link

        return node
```
```python
# LinkedQueue.py

class Node:

    def __init__ (self, elem, next=None):

        self.data = elem 

        self.link = next





# 코드 6.6: 원형으로 연결된 큐 클래스

class LinkedQueue:

    def __init__( self ):                           # 연결된 큐의 생성자, 용량지정 없음

        self.tail = None                            # 데이터 후단 노드를 가리키는 tail을 가짐



    def isEmpty( self ): return self.tail == None   # tail이 None이면 공백상태

    def isFull( self ): return False                # 연결된 구조에서 포화상태는 의미 없음



    def enqueue( self, item ):

        node = Node(item, None)                     # 삽입할 노드

        if self.isEmpty() :                         # 공백상태일때, 삽입하는 경우

           self.tail = node

           node.link = node

        else :                                      # 공백상태가 아닐 떄, 삽입하는 경우

            node.link = self.tail.link

            self.tail.link = node

            self.tail = node



    def dequeue( self ):

        if not self.isEmpty():

            data = self.tail.link.data                  # 반환할 데이터를 저장해 둠

            if self.tail.link == self.tail : 

               self.tail = None                         # case1 : 큐의 요소가 하나인 경우의 삭제

            else:

                self.tail.link = self.tail.link.link    # case2 : 요소가 여러 개인 경우의 삭제

            return data



    def peek( self ):

        if not self.isEmpty():

            return self.tail.link.data







    # 코드 6.7: 원형으로 연결된 큐의 요소의 수 계산

    def size( self ):

        if self.isEmpty() : return 0

        else :

            count = 1

            node = self.tail.link           # node는 front노드

            while not node == self.tail :   # 반복문 종료 조건

                node = node.link            # 다음 노드로 진행

                count += 1

            return count



    # 코드 6.8: 문자열 변환을 위한 str 연산자 중복

    def __str__( self ):

        arr = []

        if not self.isEmpty() :

            node = self.tail.link           # node는 font노드

            while not node == self.tail :   # 반복문 종료 조건

                arr.append(node.data)       # 노드의 데이터를 리스트에 추가

                node = node.link            # 다음 노드로 진행

            arr.append(node.data)           # 마지막으로 rear의 데이터 추가

        return str(arr)                     # 리스트를 문자열로 변환해 반환
```
### 10주차
```python
# 코드 6.9: 이중연결구조의 노드 클래스        # DoublyLinkedDeque 
class DNode:
    def __init__ (self, elem, prev, next):
        self.data = elem 
        self.prev = prev
        self.next = next



# 코드 6.10: 이중연결구조로 구현한 덱
class DoublyLinkedDeque:
    def __init__( self ):
        self.front = None
        self.rear = None

    def isEmpty( self ): return self.front == None
    def isFull( self ): return False

    def addFront( self, item ):
        node = DNode(item, None, self.front)
        if( self.isEmpty()):
            self.front = self.rear = node
        else :
            self.front.prev = node
            self.front = node

    def addRear( self, item ):
        node = DNode(item, self.rear, None)
        if( self.isEmpty()):
            self.front = self.rear = node
        else :
            self.rear.next = node
            self.rear = node

    def deleteFront( self ):
        if not self.isEmpty():
            data = self.front.data
            self.front = self.front.next
            if self.front==None :
                self.rear = None
            else:
                self.front.prev = None
            return data

    def deleteRear( self ):
        if not self.isEmpty():
            data = self.rear.data
            self.rear = self.rear.prev
            if self.rear==None :
                self.front = None
            else:
                self.rear.next = None
            return data

    def __str__( self ) :
        arr = []
        node = self.front
        while not node == None :
            arr.append(node.data)
            node = node.next
        return str(arr)

#======================================================================
if __name__ == "__main__":
    dq = DoublyLinkedDeque()

    for i in range(9):
        if i%2==0 : dq.addRear(i)
        else : dq.addFront(i)
    print("홀수->전단, 짝수->후단:", dq)

    for i in range(2): dq.deleteFront()
    for i in range(3): dq.deleteRear()
    print(" 전단삭제x2 후단삭제x3:", dq)

    for i in range(9,14): dq.addFront(i)
    print("   전단삽입 9,10,...13:", dq)

```
```python
def printStep(arr, val) :
    print("  Step %2d = " % val, end='')
    print(arr)


# 코드 7.1: 선택 정렬 알고리즘        참고 파일: ch07/basic_sort.py
def selection_sort(A) :
    n = len(A)
    for i in range(n-1) :
        least = i;
        for j in range(i+1, n) :
            if (A[j]<A[least]) :
                least = j
        A[i], A[least] = A[least], A[i]	    # 배열 항목 교환 
        printStep(A, i + 1);	            # 중간 과정 출력용 문장 

# 코드 7.2: 삽입 정렬 알고리즘        참고 파일: ch07/basic_sort.py
def insertion_sort(A) :
    n = len(A)
    for i in range(1, n) :
        key = A[i]
        j = i-1
        while j>=0 and A[j] > key :
            A[j + 1] = A[j]
            j -= 1
        A[j + 1] = key
        printStep(A, i)

# 코드 7.3: 버블 정렬 알고리즘        참고 파일: ch07/basic_sort.py
def bubble_sort(A) :
    n = len(A)
    for i in range(n-1, 0, -1) :
        bChanged = False
        for j in range (i) :
            if (A[j]>A[j+1]) :
                A[j], A[j+1] = A[j+1], A[j] 
                bChanged = True

        if not bChanged: break;			# 교환이 없으면 종료
        printStep(A, n - i);			# 중간 과정 출력용 문장


if __name__ == "__main__":
    org = [ 5, 3, 8, 4, 9, 1, 6, 2, 7 ]

    data = list(org)
    print("Original  :", org)
    selection_sort(data)
    print("Selection :", data)

    data = list(org)
    print("Original  :", org)
    insertion_sort(data)
    print("Insertion :", data)

    data = list(org)
    print("Original  :", org)
    bubble_sort(data)
    print("Bubble    :", data)

```
### 12주차
```python
# 코드 8.1: 이진트리를 위한 노드 클래스           # Binary tree
class TNode:                                    # 이진트리를 위한 노드 클래스
    def __init__ (self, elem, left, right):     # 생성자
        self.data = elem                        # 노드의 데이터
        self.left = left                        # 왼쪽 자식을 위한 링크
        self.right = right                      # 오른쪽 자식을 위한 링크

    def isLeaf(self):
        return self.left is None and self.right is None

# 코드 8.2: 이진트리의 전위순회
def preorder(n) :                               # 전위순회 함수
    if n is not None :                          
        print(n.data, end=' ')                  # 먼저 루트노드 처리(화면 출력)
        preorder(n.left)                        # 왼쪽 서브트리 처리
        preorder(n.right)                       # 오른쪽 서브트리 처리

# 코드 8.3: 이진트리의 중위순회
def inorder(n) :                                # 중위순회 함수
    if n is not None :          
        inorder(n.left)                         # 왼쪽 서브트리 처리
        print(n.data, end=' ')                  # 루트노드 처리(화면 출력)
        inorder(n.right)                        # 오른쪽 서브트리 처리

# 코드 8.4: 이진트리의 후위순회
def postorder(n) :                              # 후위순회 함수
    if n is not None :
        postorder(n.left)                       # 왼쪽 서브트리 처리
        postorder(n.right)                      # 오른쪽 서브트리 처리
        print(n.data, end=' ')                  # 루트노드 처리(화면출력)

# 코드 8.5: 이진트리의 레벨순회
from CircularQueue import CircularQueue

def levelorder(root) :
    queue = CircularQueue()                     # 큐 객체 초기화
    queue.enqueue(root)                         # 최초에 큐에는 루트 노드만 들어있음
    while not queue.isEmpty() :                 # 큐가 공백상태가 아닌 동안,
        n = queue.dequeue()                     # 큐에서 맨 앞의 노드 N을 꺼냄
        if n is not None :
            print(n.data, end=' ')              # 먼저 노드의 정보를 출력
            queue.enqueue(n.left)               # N의 왼쪽 자식 노드를 큐에 삽입
            queue.enqueue(n.right)              # N의 오른쪽 자식 노드를 큐에 삽입

# 코드 8.6: 이진트리의 노드 수 계산
def count_node(n) :                             # 순환을 이용해 트리의 노드 수를 계산하는 함수
    if n is None :                              # N이 none면 공백트리 >> 0을 반환
        return 0
    else :                                      # 좌우 서브트리의 노드 수의 합 +1을 반환 (순환이용)
        return 1 + count_node(n.left) + count_node(n.right)

# 코드 8.7: 이진트리의 단말노드 수 계산      
def count_leaf(n) :
    if n is None : return 0                     # 공백트리 >> 0을 반환
    elif n.isLeaf() : return 1                  # 단말노드 >> 1을 반환
    else : return count_leaf(n.left) + count_leaf(n.right)  # 비난말노드 >> 좌우 서브트리의 결과 합을 반환


# 코드 8.8: 이진트리의 트리의 높이 계산
def calc_height(n) :
    if n is None : return 0                     # 공백트리 >> 0을 반환
    hLeft = calc_height(n.left)
    hRight = calc_height(n.right)
    if (hLeft > hRight) : return hLeft + 1
    else: return hRight + 1


# 코드 8.9: 이진 트리 연산 테스트 프로그램
if __name__ == "__main__":
    print("\n======= 이진트리 테스트 ===================================") 
    d = TNode('D', None, None)
    e = TNode('E', None, None)
    b = TNode('B', d, e)
    f = TNode('F', None, None)
    c = TNode('C', f, None)
    root = TNode('A', b, c)

    print('\n   In-Order : ', end='')
    inorder(root)
    print('\n  Pre-Order : ', end='')
    preorder(root)
    print('\n Post-Order : ', end='')
    postorder(root)
    print('\nLevel-Order : ', end='')
    levelorder(root)
    print()

    print(" 노드의 개수 = %d개" % count_node(root))
    print(" 단말의 개수 = %d개" % count_leaf(root))
    print(" 트리의 높이 = %d" % calc_height(root))
```
```python

# 코드 8.14: 최대힙의 삽입 알고리즘         참고 코드: ch08/MaxHeap.py
def heappush(heap, n) :
    heap.append(n)		    # 맨 마지막 노드로 일단 삽입
    i = len(heap)-1			# 노드 n의 위치
    while i != 1 :          # n이 루트가 아니면 up-heap 진행
        pi = i//2           # 부모 노드의 위치
        if n <= heap[pi]:   # 부모보다 작으면 up-heap 종료
            break
        heap[i] = heap[pi]	# 부모를 끌어내림
        i = pi			    # i가 부모의 인덱스가 됨
    heap[i] = n			    # 마지막 위치에 n 삽입


# 코드 8.15: 최대힙의 삭제 알고리즘         참고 코드: ch08/MaxHeap.py
def heappop(heap) :
    size = len(heap) - 1    # 노드의 개수
    if size == 0 :          # 공백상태
       return None

    root = heap[1]		    # 삭제할 루트 노드(사장)
    last = heap[size]	    # 마지막 노드(말단사원)
    pi = 1                  # 부모 노드의 인덱스
    i = 2                   # 자식 노드의 인덱스

    while (i <= size):	    # 마지막 노드 이전까지
        if i<size and heap[i] < heap[i+1]:  # right가 더 크면 i를 1 증가 (기본은 왼쪽 노드)
            i += 1          # 비교할 자식은 오른쪽 자식
        if last >= heap[i]: # 자식이 더 작으면 down-heap 종료
            break
        heap[pi] = heap[i]  # 아니면 down-heap 계속
        pi = i              
        i *= 2

    heap[pi] = last	        # 맨 마지막 노드를 parent위치에 복사
    heap.pop()		        # 맨 마지막 노드 삭제
    return root			    # 저장해두었던 루트를 반환


if __name__ == "__main__":
    # 코드 8.16: 최대힙 테스트 프로그램
    data = [2, 5, 4, 8, 9, 3, 7, 3]		# 힙에 삽입할 데이터
    heap = [0]
    print("입력: ", data)
    for e in data :			    # 모든 데이터를 힙에 삽입
        heappush(heap, e)
        print("heap: ", heap[1:])

    print("삭제: ", heappop(heap))
    print("heap: ", heap[1:])
    print("삭제: ", heappop(heap))
    print("heap: ", heap[1:])
```
```python
# 너비 우선 탐색 ( 인접 행렬 방식)
def DFS(vtx, adj, s, visited) :

    print(vtx[s], end=' ')          # 현재 정점 s를 출력함

    visited[s] = True               # 현재 정점 s를 visited에 추가함

    for v in range(len(vtx)) :      # 인접행렬

        if adj[s][v] != 0 :         # 모든 간선 (s,v)에 대해

            if visited[v]==False:   # v를 아직 방문하지 않았으면 

                DFS(vtx, adj, v, visited)
vtx =  ['A', 'B','C','D','E','F','G','H']

edge = [ [  0,  1,  1,  0,  0,  0,  0,  0],

         [  1,  0,  0,  1,  0,  0,  0,  0],

         [  1,  0,  0,  1,  1,  0,  0,  0],

         [  0,  1,  1,  0,  0,  1,  0,  0],

         [  0,  0,  1,  0,  0,  0,  1,  1],

         [  0,  0,  0,  1,  0,  0,  0,  0],

         [  0,  0,  0,  0,  1,  0,  0,  1],

         [  0,  0,  0,  0,  1,  0,  1,  0] ]



print('DFS(출발:A) : ', end="")

DFS(vtx, edge, 0, [False]*len(vtx))

print()
```
```python
# 너비 우선 탐색 ( 인접 리스트 방식)
from queue import Queue                 # queue 모듈의 Queue 사용

def BFS_AL(vtx, aList, s):

    n = len(vtx)                        # 그래프의 정점 수

    visited = [False]*n                 # 방문 확인을 위한 리스트

    Q = Queue()                         # 공백상태의 큐 생성

    Q.put(s)                            # 맨 처음에는 시작 정점만 있음

    visited[s] = True                   # s는 "방문"했다고 표시

    while not Q.empty() :

        s = Q.get()                     # 큐에서 정점을 꺼냄

        print(vtx[s], end=' ')          # 정점을 출력(처리)함

        for v in aList[s] :               # s의 모든 이웃 v에 대해

            if visited[v]==False :      # 방문하지 않은 이웃 정점이면

                Q.put(v)                # 큐에 삽입

                visited[v] = True       # "방문"했다고 표시
vtx = [ 'A','B','C','D','E','F','G','H']

aList = [[ 1, 2 ],      # 'A'의 인접정점 인덱스

         [ 0, 3 ],      # 'B'의 인접정점 인덱스

         [ 0, 3, 4 ],   # 'C'

         [ 1, 2, 5 ],   # 'D'

         [ 2, 6, 7 ],   # 'E'

         [ 3 ],         # 'F'

         [ 4, 7 ],      # 'G'

         [ 4, 6 ] ]     # 'H'



print('BFS_AL(출발:A): ', end="")

BFS_AL(vtx, aList, 0)

print()
```
```python
# 딕셔너리와 집합으로 표현된 그래프의 깊이우선탐색
def DFS2(graph, v, visited):

    if v not in visited :           # v가 방문되지 않았으면

        visited.add(v)              # v를 방문했다고 표시

        print(v, end=' ')           # v를 출력

        nbr = graph[v] - visited    # v의 인접 정점 리스트

        for u in nbr:               # v의 모든 인접 정점에 대해 

            DFS2(graph, u, visited)  # 순환 호출
mygraph = { "A" : {"B","C"},

            "B" : {"A", "D"},

            "C" : {"A", "D", "E"},

            "D" : {"B", "C", "F"},

            "E" : {"C", "G", "H"},

            "F" : {"D"},

            "G" : {"E", "H"},

            "H" : {"E", "G"}

          }



print('DFS2(출발:A) : ', end="")

DFS2(mygraph, "A", set())

print()
```
```python
# 연결성분검사 주 함수
def find_connected_component(vtx, adj) :

    n = len(vtx)

    visited = [False]*n

    groups = []# 부분 그래프별 정점 리스트



    for v in range(n) :

        if visited[v] == False :

            color = bfs_cc(vtx, adj, v, visited)

            groups.append( color )



    return groups
```
```python
# 너비우선탐색을 이용한 연결성분 검사
from queue import Queue

def bfs_cc(vtx, adj, s, visited):

    group = [s]    # 새로운 연결된 그룹 생성

    Q = Queue()

    Q.put(s)

    visited[s] = True

    while not Q.empty() :

        s = Q.get()

        for v in range(len(vtx)) :

            if visited[v]==False and adj[s][v] != 0 :

                Q.put(v)

                visited[v] = True

                group.append(v)

    return group
```
```python
# 연결성분검사 테스트 프로그램
vertex =    ['A', 'B','C','D','E']

adjMat =  [ [  0,  1,  1,  0,  0 ],

            [  1,  0,  0,  0,  0 ],

            [  1,  0,  0,  0,  0 ],

            [  0,  0,  0,  0,  1 ],

            [  0,  0,  0,  1,  0 ] ]



colorGroup = find_connected_component(vertex, adjMat)

print("연결성분 개수 = %d " % len(colorGroup))

print(colorGroup)# 정점 리스트들을 출력
```
```python
# 깊이우선탐색을 이용한 신장트리
def ST_DFS(vtx, adj, s, visited) :

    visited[s] = True               # 현재 정점 s를 visited에 추가함

    for v in range(len(vtx)) :      # 인접행렬

        if adj[s][v] != 0 :         # 모든 간선 (s,v)에 대해

            if visited[v]==False:   # v를 아직 방문하지 않았으면 

                print("(", vtx[s], vtx[v], ")", end=' ')  # 간선 출력

                ST_DFS(vtx, adj, v, visited)





# 테스트 프로그램

vtx =  ['A', 'B','C','D','E','F','G','H']

edge = [ [  0,  1,  1,  0,  0,  0,  0,  0],

         [  1,  0,  0,  1,  0,  0,  0,  0],

         [  1,  0,  0,  1,  1,  0,  0,  0],

         [  0,  1,  1,  0,  0,  1,  0,  0],

         [  0,  0,  1,  0,  0,  0,  1,  1],

         [  0,  0,  0,  1,  0,  0,  0,  0],

         [  0,  0,  0,  0,  1,  0,  0,  1],

         [  0,  0,  0,  0,  1,  0,  1,  0] ]



print('신장트리(DFS): ', end="")

ST_DFS(vtx, edge, 0, [False]*len(vtx))

print()
```
```python
# 위상정렬
def topological_sort_AM(vertex, edge) :

    # 정점의 진입차수 리스트 inDeg 생성 및 초기화

    n = len(vertex)             # 정점의 개수

    inDeg = [0] * n             # inDeg: 진입차수 저장 리스트

    for i in range(n) :

        for j in range(n) :

            if edge[i][j]>0 :  # 모든 간선 <i,j>에 대해

                inDeg[j] += 1   # j의 진입차수를 1 증가



    # 진입차수가 0인 정점 리스트 생성 및 초기화

    vlist = []                  

    for i in range(n) :

        if inDeg[i]==0 : 

            vlist.append(i)



    # 진입차수가 0인 정점이 더 이상 없을 때 까지 위상 정렬

    while len(vlist) > 0 :

        v = vlist.pop()                 # 진입차수가 0인 정점을 꺼냄

        print(vertex[v], end=' ')       # 화면 출력(방문, 또는 수강)



        for u in range(n) :

            if v!=u and edge[v][u]>0:  # 간선 <v,u>가 있으면

                inDeg[u] -= 1           # u의 진입차수 감소

                if inDeg[u] == 0 :      # u의 진입차수가 0이면

                    vlist.append(u)     # u를 vlist에 추가







vertex = ['A', 'B', 'C', 'D', 'E', 'F' ]

adj =  [ [ 0,   0,   1,   1,   0,   0 ],

         [ 0,   0,   0,   1,   1,   0 ],

         [ 0,   0,   0,   1,   0,   1 ],

         [ 0,   0,   0,   0,   0,   1 ],

         [ 0,   0,   0,   0,   0,   1 ],

         [ 0,   0,   0,   0,   0,   0 ] ]

print('topological_sort: ')

topological_sort_AM(vertex, adj)

print()
```
