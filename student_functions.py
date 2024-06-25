import numpy as np
from collections import deque
import heapq

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
    frontier = [(start, None)] # Stack with tuples of (node, parent)
    visited[start] = None

    while frontier:
        current, parent = frontier.pop()
        visited[current] = parent

        if current == end:
            path = findPath(visited, end)
            return visited, path

        for neighbor, connected in reversed(list(enumerate(matrix[current]))):
            if connected != 0 and neighbor not in visited:
                frontier.append((neighbor, current))
                

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
    numNodes = len(matrix)
    visited = {}
    expand = {}
    visited[start] = None  # Starting node has no predecessor
    expand[start] = None

    pq = [(0, start)]  # Priority queue with tuples of (cost, node)
    heapq.heapify(pq)  # Heapify the priority queue

    cost = {node: float('inf') for node in range(numNodes)}
    cost[start] = 0

    while pq:
        currCost, currNode = heapq.heappop(pq)
        visited[currNode] = None

        if currNode == end:
            for node in visited:
                visited[node] = expand[node]
            path = findPath(visited, end)
            return visited, path

        for neighbor, weight in enumerate(matrix[currNode]):
            if weight != 0:
                new_cost = currCost + weight
                if new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor))
                    expand[neighbor] = currNode
    
    return visited, []
def heuristic(matrix, node, end):
    """
    Calculate the heuristic value of a node
    Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    node: integer 
        current node
    end: integer
        ending node
    
    Returns
    ---------------------
    heuristic value
    """
    if matrix[node][end] == 0 and node != end:
        return float('inf')
    elif node == end:
        return 0
    return matrix[node][end]
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
    
    path = []
    visited = {}
    expand = {}
    visited[start] = None  # Starting node has no predecessor
    expand[start] = None

    pq = [(heuristic(matrix, start, end), start)]  # Priority queue with tuples (heuristic_value, node)
    heapq.heapify(pq)  # Heapify the priority queue



    while pq:
        _, current = heapq.heappop(pq)  # Dequeue node with smallest heuristic value
        visited[current] = None  # Add current node to visited nodes

        if current == end:
            for node in visited:
                visited[node] = expand[node]
            path = findPath(visited, end)
            return visited, path
        
        for neighbor, weight in enumerate(matrix[current]):
            if weight != 0 and neighbor not in visited:
                heapq.heappush(pq, (heuristic(matrix, neighbor, end), neighbor))  # Push neighbor with its heuristic value
                expand[neighbor] = current  # Record current node as predecessor

    return visited, path
def euclidean_distance(pos1, pos2):
    """
    Calculate the Euclidean distance between two positions.
    
    Parameters:
    ---------------------------
    pos1: tuple
        Position of the current node (x, y)
    pos2: tuple
        Position of the goal node (x, y)
    
    Returns:
    ---------------------
    float:
        Euclidean distance between pos1 and pos2
    """
    return np.linalg.norm(np.array(pos1) - np.array(pos2))
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
    path = []
    visited = {}
    expand = {}
    pq = [(euclidean_distance(start, end), start)]  # Priority queue with tuples (f_score, node)
    heapq.heapify(pq)  # Heapify the priority queue
    g_score = {start: 0}  # Cost from start to node
    visited[start] = None  # Starting node has no predecessor
    expand[start] = None

    while pq:
        current_f, current = heapq.heappop(pq)  # Dequeue node with smallest f_score
        visited[current] = None  # Add current node to visited nodes

        if current == end:
            for node in visited:
                visited[node] = expand[node]
            path = findPath(visited, end)
            return visited, path
        
        for neighbor, weight in enumerate(matrix[current]):
            if weight != 0:  # Check for edge existence
                temp_g_score = g_score[current] + weight
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    g_score[neighbor] = temp_g_score
                    h_score = euclidean_distance(pos[neighbor], pos[end])
                    f_score = temp_g_score + h_score
                    heapq.heappush(pq, (f_score, neighbor))
                    expand[neighbor] = current

    return visited, path
def DLS(matrix, start, end, depth_limit):
    """
    Depth-Limited Search function for IDDFS
    
    Parameters:
    ---------------------------
    matrix: np.array 
        The graph's adjacency matrix
    start: int
        Starting node
    end: int
        Target node to reach
    depth_limit: int
        Maximum depth limit for the search
    
    Returns:
    ---------------------
    visited: dictionary
        Contains visited nodes: each key is a visited node,
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    visited = {}  
    path = [] 
    frontier = [(start, None, 0)] # Stack with tuples of (node, parent, depth)  
    visited[start] = None

    
    while frontier:
        current, parent, cur_depth = frontier.pop()
        visited[current] = parent

        if current == end:
            path = findPath(visited, end)
            return visited, path

        if cur_depth < depth_limit:
            for neighbor, connected in reversed(list(enumerate(matrix[current]))):
                if connected != 0 and neighbor not in visited:
                    frontier.append((neighbor, current, cur_depth + 1))

    print("No path found within depth limit")
    return visited, []  
def IDS(matrix, start, end):
    """
    Iterative Deepening Search algorithm
    
    Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns:
    ---------------------
    visited: dictionary
        Contains visited nodes: each key is a visited node,
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    depth_limit = 0
    while True:
        visited, path = DLS(matrix, start, end, depth_limit)
        if path is not None:
            return visited, path
        depth_limit += 1