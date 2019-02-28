from coin_flip import flip
from stack_end import Stack


class Map:
    def __init__(self):
        self.collection = SkipList()

    def put(self, key, value):
        self.collection.insert(key, value)

    def get(self, key):
        return self.collection.search(key)

        
class SkipList:
    def __init__(self):
        self.head = None


class HeaderNode:
    def __init__(self):
        self.next = None
        self.down = None

    def getNext(self):
        return self.next

    def getDown(self):
        return self.down

    def setNext(self, newNext):
        self.next = newNext

    def setDown(self, newDown):
        self.down = newDown


class DataNode:
    def __init__(self, key, value):
        self.key = key
        self.data = value
        self.next = None
        self.down = None

    def getKey(self):
        return self.key

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def getDown(self):
        return self.down

    def setData(self, newData):
        self.data = newData

    def setNext(self, newNext):
        self.next = newNext

    def setDown(self, newDown):
        self.down = newDown


def search(self, key):
    current = self.head
    found = False
    stop = False
    while not found and not stop:
        if current == None:
            stop = True
        else:
            if current.getNext() == None:
                current = current.getDown()
            else:
                if current.getNext().getKey() == key:
                    found = True
                else:
                    if key < current.getNext().getKey():
                        current = current.getDown()
                    else:
                        current = current.getNext()
    if found:
        return current.getNext().getData()
    else:
        return None

def insert(self, key, data):
    if self.head == None:
        self.head = HeaderNode()
        temp = DataNode(key, data)
        self.head.setNext(temp)
        top = temp
        while flip() == 1:
            newHead = HeaderNode()
            temp = DataNode(key, data)
            temp.setDown(top)
            newHead.setNext(temp)
            newHead.setDown(self.head)
            self.head = newHead
            top = temp
    else:
        towerStack = Stack()
        current = self.head
        stop = False
        while not stop:
            if current == None:
                stop = True
            else:
                if current.getNext() == None:
                    towerStack.push(current)
                    current = current.getDown()
                else:
                    if current.getNext().getKey() > key:
                        towerStack.push(current)
                        current = current.getDown()
                    else:
                        current = current.getNext()
        lowestLevel = towerStack.pop()
        temp = DataNode(key, data)
        temp.setNext(lowestLevel.getNext())
        lowestLevel.setNext(temp)
        top = temp
        while flip() == 1:
            if towerStack.isEmpty():
                newHead = HeaderNode()
                temp = DataNode(key, data)
                temp.setDown(top)
                newHead.setNext(temp)
                newHead.setDown(self.head)
                self.head = newHead
                top = temp
            else:
                nextLevel = towerStack.pop()
                temp = DataNode(key, data)
                temp.setDown(top)
                temp.setNext(nextLevel.getNext())
                nextLevel.setNext(temp)
                top = temp
