import sys


class OctTree:
    def __init__(self):
        self.root = None
        self.maxLevel = 5
        self.numLeaves = 0
        self.leafList = []

    def insert(self, r, g, b):
        if not self.root:
            self.root = self.otNode(outer = self)
        self.root.insert(r, g, b, 0, self)

    def find(self, r, g, b):
        if self.root:
            return self.root.find(r, g, b, 0)

    def reduce(self, maxCubes):
        while len(self.leafList) > maxCubes:
            smallest = self.findMinCube()
            smallest.parent.merge()
            self.leafList.append(smallest.parent)
            self.numLeaves = self.numLeaves + 1

    def findMinCube(self):
        minCount = sys.maxsize
        maxLev = 0
        minCube = None
        for i in self.leafList:
            if i.count <= minCount and i.level >= maxLev:
                minCube = i
                minCount = i.count
                maxLev = i.level
        return minCube


    class otNode:
        def __init__(self, parent = None, level = 0, outer = None):
            self.red = 0
            self.green = 0
            self.blue = 0
            self.count = 0
            self.parent = parent
            self.level = level
            self.oTree = outer
            self.children = [None] * 8

        def insert(self, r, g, b, level, outer):
            if level < self.oTree.maxLevel:
                idx = self.computeIndex(r, g, b, level)
                if self.children[idx] == None:
                    self.children[idx] = outer.otNode(parent = self,
                                                      level = level + 1,
                                                      outer = outer)
                self.children[idx].insert(r, g, b, level + 1, outer)
            else:
                if self.count == 0:
                    self.oTree.numLeaves = self.oTree.numLeaves + 1
                    self.oTree.leafList.append(self)
                self.red += r
                self.green += g
                self.blue += b
                self.count = self.count + 1

        def computeIndex(self, r, g, b, level):
            shift = 8 - level
            rc = r >> shift - 2 & 0x4
            gc = g >> shift - 1 & 0x2
            bc = b >> shift & 0x1
            return (rc | gc | bc)

        def find(self, r, g, b, level):
            if level < self.oTree.maxLevel:
                idx = self.computeIndex(r, g, b, level)
                if self.children[idx]:
                    return self.children[idx].find(r, g, b, level + 1)
                elif self.count > 0:
                    return ((self.red // self.count,
                             self.green // self.count,
                             self.blue // self.count))
                else:
                    print('Error: No leaf node for this color')
            else:
                return ((self.red // self.count,
                         self.green // self.count,
                         self.blue // self.count))

        def merge(self):
            for i in self.children:
                if i:
                    if i.count > 0:
                        self.oTree.leafList.remove(i)
                        self.oTree.numLeaves -= 1
                    else:
                        print('Recursively merging non-leaf...')
                        i.merge()
                    self.count += i.count
                    self.red += i.red
                    self.green += i.green
                    self.blue += i.blue
            for i in range(8):
                self.children[i] = None
