# Stack implementation where the end of the list holds
# the top elements in the stack
class Stack:
    def __init__(self):
        self.items = []
        
    def isEmpty(self):
        return self.items == []
    
    def push(self, item):
        self.items.append(item)
        
    def pop(self):
        return self.items.pop()
    
    def peek(self):
        return self.items[len(self.items)-1]
    
    def size(self):
        return len(self.items)
    
    
# Stack example
s = Stack()
s.isEmpty() # True
s.push(4)
s.push('dog')
s.peek() # dog
s.push(True)
s.size() # 3
s.isEmpty() # False
s.push(8.4)
s.pop() # 8.4
s.pop() # True
s.size() # 2


# Alternative stack implmentation where the beginning
# of the list holds the top elements of the stack
class Stack:
    def __init__(self):
        self.items = []
        
    def isEmpty(self):
        return self.items == []
    
    def push(self, item):
        self.items.insert(0, item)
        
    def pop(self):
        return self.items.pop(0)
    
    def peek(self):
        return self.items[0]
    
    def size(self):
        return len(self.items)
