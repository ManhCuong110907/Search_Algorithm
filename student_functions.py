import numpy as np
from collections import deque

def findPath(visited, end):
    """
    Find the path from visited nodes
    Parameters:
    ---------------------------
    visited: dictionary
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    end: integer
        ending node
    
    Returns
    ---------------------
    path: list
        Founded path
    """   
    path = []
    if end in visited:
        step = end
        while step is not None:
            path.insert(0, step)
            step = visited[step]
    return path
def BFS(matrix, start, end):
    """"
    BFS algorithm:
    Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO: 
   
    path = []
    visited = {}
    expand = {}
    frontier = deque([start])
    visited[start] = None
    expand[start] = None

    while frontier:
        current = frontier.popleft()
        visited[current] = None #Add current node to visited nodes

        for neighbor, connected in enumerate(matrix[current]):
            if connected != 0 and neighbor not in visited and neighbor not in frontier:
                if neighbor == end:
                    visited[neighbor] = current #Add goal node to visited nodes
                    expand[neighbor] = current
                    for node in visited:
                        visited[node] = expand[node] #Update the adjacent node visited before it
                    path = findPath(visited, end)
                    return visited, path
                
                frontier.append(neighbor)  
                expand[neighbor] = current
                
    return visited, path

def DFS(matrix, start, end):
    """
    DFS algorithm
     Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited 
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """

    # TODO: 
    visited = {}
    expand = {}
    frontier = [start]
    visited[start] = None
    expand[start] = None

    while frontier:
        current = frontier.pop(0)
        visited[current] = None

        if current == end:
            for node in visited:
                visited[node] = expand[node]
            path = findPath(visited, end)
            return visited, path

        for neighbor, connected in enumerate(matrix[current]):
            if connected != 0 and neighbor not in visited:
                frontier.append(neighbor)
                expand[neighbor] = current
                break

    return visited, []


def UCS(matrix, start, end):
    """
    Uniform Cost Search algorithm
     Parameters:visited
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO:  
    path=[]
    visited={}
    return visited, path

def GBFS(matrix, start, end):
    """
    Greedy Best First Search algorithm 
    heuristic : edge weights
     Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
   
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO: 
    path=[]
    visited={}
    return visited, path

def Astar(matrix, start, end, pos):
    """
    A* Search algorithm
    heuristic: eclid distance based positions parameter
     Parameters:
    ---------------------------
    matrix: np array UCS
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    pos: dictionary. keys are nodes, values are positions
        positions of graph nodes
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO: 

    path=[]
    visited={}
    return visited, path

