import sys
import os
import unittest

class Vertex:
    def __init__(self, num):
        self.id = num
        self.connectedTo = {}
        self.color = 'white'
        self.dist = sys.maxsize
        self.pred = None
        self.disc = 0
        self.fin = 0

    def addNeighbor(self, nbr, weight = 0):
        self.connectedTo[nbr] = weight
        
    def setColor(self, color):
        self.color = color
        
    def setDistance(self, d):
        self.dist = d
        
    def setPred(self, p):
        self.pred = p
        
    def setDiscovery(self, dtime):
        self.disc = dtime
        
    def setFinish(self, ftime):
        self.fin = ftime
        
    def getFinish(self):
        return self.fin
    
    def getDiscovery(self):
        return self.disc
    
    def getPred(self):
        return self.pred
    
    def getDistance(self):
        return self.dist
    
    def getColor(self):
        return self.color

    def __str__(self):
        return str(self.id) + ':color ' + self.color + ':disc ' \
            + str(self.disc)

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, nbr):
        return self.connectedTo[nbr]


class Graph:
    def __init__(self):
        self.vertices = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertices[key] = newVertex
        return newVertex

    def getVertex(self, n):
        if n in self.vertices:
            return self.vertices[n]
        else:
            return None

    def __contains__(self, n):
        return n in self.vertices

    def addEdge(self, f, t, cost = 0):
        if f not in self.vertices:
            nv = self.addVertex(f)
        if t not in self.vertices:
            nv = self.addVertex(t)
        self.vertices[f].addNeighbor(self.vertices[t], cost)

    def getVertices(self):
        return list(self.vertices.keys())

    def __iter__(self):
        return iter(self.vertices.values())
