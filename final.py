import heapq
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import math



"""

This Python code implements a GPS navigation system using graph-based algorithms. It includes functionalities for administrators to manage the graph by adding or deleting places, 
updating traffic information, and adding amenities. Users can interactively find routes between places, visualize maps with or without amenities, and locate amenities along the path. 
The core functionality relies on Dijkstra's algorithm for calculating the shortest path, with a time complexity of O((V+E)logV), 
where V is the number of vertices and E is the number of edges. Overall, the system provides efficient navigation capabilities by leveraging graph data structures and algorithms."""
class Graph:
    def _init_(self):
        self.adjacency_matrix = []
        self.traffic_info = {}
        self.amenities = {}
        self.vertex_names = {}
        self.vertex_indices = {}

    def add_vertex(self, name):
        index = len(self.vertex_names)
        self.vertex_names[index] = name
        self.vertex_indices[name] = index

        # Increase the size of the adjacency matrix
        for row in self.adjacency_matrix:
            row.append(float('inf'))
        self.adjacency_matrix.append([float('inf')] * (len(self.vertex_names)))

    def add_edge(self, from_vertex, to_vertex, weight):
        from_index = self.vertex_indices[from_vertex]
        to_index = self.vertex_indices[to_vertex]
        self.adjacency_matrix[from_index][to_index] = weight
        self.adjacency_matrix[to_index][from_index] = weight

    def add_edge_with_traffic(self, from_vertex, to_vertex, weight,tr):
        traffic = tr
        total_weight = weight + traffic
        self.add_edge(from_vertex, to_vertex, total_weight)
        self.traffic_info[(from_vertex, to_vertex)] = traffic
        self.traffic_info[(to_vertex, from_vertex)] = traffic

    def delete_vertex(self, name):
        if name not in self.vertex_indices:
            print(f"Vertex {name} does not exist.")
            return
        
        index_to_remove = self.vertex_indices[name]

        # Remove the vertex from the adjacency matrix
        self.adjacency_matrix.pop(index_to_remove)
        for row in self.adjacency_matrix:
            row.pop(index_to_remove)

        # Remove the vertex from vertex_names and vertex_indices
        del self.vertex_names[index_to_remove]
        del self.vertex_indices[name]

        # Update the indices in vertex_names and vertex_indices
        self.vertex_names = {i: v for i, (k, v) in enumerate(self.vertex_names.items())}
        self.vertex_indices = {v: i for i, v in self.vertex_names.items()}

        # Remove associated traffic info and amenities
        self.traffic_info = {k: v for k, v in self.traffic_info.items() if k[0] != name and k[1] != name}
        self.amenities = {k: v for k, v in self.amenities.items() if k[0] != name and k[1] != name}

    def add_amenities(self, from_vertex, to_vertex, amenities):
        if (from_vertex, to_vertex) in self.amenities:
            self.amenities[(from_vertex, to_vertex)].extend(amenities)
        else:
            self.amenities[(from_vertex, to_vertex)] = amenities

    """
    The time complexity of dijkstra_shortest_path is O((V+E)logV) 
    where V is the number of vertices and E is the number of edges in the Graph.
    """


    """
    Since the visualize_shortest_path is mainly using dijkstra_shortest_path.So the
    overall time complexity is O((V+E)logV.
    """
    def dijkstra_shortest_path(self, start, target):
            start_index = self.vertex_indices[start]
            target_index = self.vertex_indices[target]

            distances = {vertex: float('inf') for vertex in range(len(self.vertex_names))}
            distances[start_index] = 0
            priority_queue = [(0, start_index)]
            previous_nodes = {vertex: None for vertex in range(len(self.vertex_names))}

            while priority_queue:
                current_distance, current_vertex = heapq.heappop(priority_queue)

                if current_vertex == target_index:
                    path = []
                    while current_vertex is not None:
                        path.append(self.vertex_names[current_vertex])
                        current_vertex = previous_nodes[current_vertex]
                    return path[::-1]

                if current_distance > distances[current_vertex]:
                    continue

                for neighbor in range(len(self.vertex_names)):
                    weight = self.adjacency_matrix[current_vertex][neighbor]
                    if weight != float('inf'):
                        distance = current_distance + weight
                        if distance < distances[neighbor]:
                            distances[neighbor] = distance
                            previous_nodes[neighbor] = current_vertex
                            heapq.heappush(priority_queue, (distance, neighbor))

            return []

    def visualize_shortest_path(self, start, target, stops=[]):
        if start not in self.vertex_indices or target not in self.vertex_indices:
            print("Start or target node not found in the graph.")
            return

        G = nx.Graph()
        for i in range(len(self.vertex_names)):
            for j in range(i + 1, len(self.vertex_names)):
                if self.adjacency_matrix[i][j] != float('inf'):
                    G.add_edge(self.vertex_names[i], self.vertex_names[j], weight=self.adjacency_matrix[i][j], length=60)

        shortest_path = self.dijkstra_shortest_path(start, target)

        if not shortest_path:
            print(f"No path found between {start} and {target}.")
            return

        # Insert stops into the shortest path
        for stop in stops:
            if stop in shortest_path:
                print(f"Stop '{stop}' is already in the shortest path.")
            else:
                for i in range(len(shortest_path) - 1):
                    if self.adjacency_matrix[self.vertex_indices[shortest_path[i]]][self.vertex_indices[stop]] != float('inf') \
                            and self.adjacency_matrix[self.vertex_indices[stop]][self.vertex_indices[shortest_path[i + 1]]] != float('inf'):
                        shortest_path.insert(i + 1, stop)
                        print(f"Stop '{stop}' added to the shortest path.")
                        break

        pos = nx.spring_layout(G, k=0.5, iterations=50)

        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', linewidths=1,
                font_size=10, alpha=0.8)

        edge_labels = {}
        for u, v, d in G.edges(data=True):
            original_weight = d['weight'] - self.traffic_info.get((u, v), 0)
            traffic = self.traffic_info.get((u, v), 0)
            edge_labels[(u, v)] = f"Dist: {original_weight} Trf: {traffic}"

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black',
                                     font_size=7, verticalalignment="baseline", label_pos=0.5)

        nx.draw_networkx_nodes(G, pos, nodelist=shortest_path, node_color='r', node_size=1500)

        nx.draw_networkx_edges(G, pos, edgelist=[(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)],
                               edge_color='r', width=3)

        plt.title(f"Shortest Path from {start} to {target}")

        # Calculate the time traveled
        total_weight = 0
        for i in range(len(shortest_path) - 1):
            from_vertex = shortest_path[i]
            to_vertex = shortest_path[i + 1]
            total_weight += self.adjacency_matrix[self.vertex_indices[from_vertex]][self.vertex_indices[to_vertex]]

        # Calculate time traveled
        distance_km = total_weight  # Assuming weight represents distance in this context
        average_speed_kph = 30  # Average speed in km/hr
        time_hours = distance_km / average_speed_kph
        time_minutes = math.ceil(time_hours * 60)

        print(f"\nTotal distance: {distance_km} km")
        print(f"Estimated travel time: {time_minutes} minutes\n")
        plt.text(0.5, 0.95, f"Total Distance: {distance_km} kms", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.text(0.5, 0.93, f"Estimated travel time: {time_minutes:.2f} minutes", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

        # Find amenities along the path
        amenities_along_path = []
        for i in range(len(shortest_path) - 1):
            from_vertex = shortest_path[i]
            to_vertex = shortest_path[i + 1]
            if (from_vertex, to_vertex) in self.amenities:
                amenities_along_path.extend(self.amenities[(from_vertex, to_vertex)])
            if (to_vertex, from_vertex) in self.amenities:
                amenities_along_path.extend(self.amenities[(to_vertex, from_vertex)])

        plt.show()




    """
    For the show_vertices_and_edges  the time complexity is O(V+E). 
    Where V is the number of vertices and E is the number of edges in the graph.
    """
    def show_vertices_and_edges(self,ch):
        G = nx.Graph()
        for i in range(len(self.vertex_names)):
            for j in range(i + 1, len(self.vertex_names)):
                if self.adjacency_matrix[i][j] != float('inf'):
                    G.add_edge(self.vertex_names[i], self.vertex_names[j], weight=self.adjacency_matrix[i][j], length=60)
        pos = nx.spring_layout(G, k=7, iterations=50)
        plt.figure(figsize=(70, 8))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', linewidths=1,
                font_size=10, alpha=1)
        if(ch==1):
            edge_labels = {}
            for u, v, d in G.edges(data=True):
                original_weight = d['weight'] - self.traffic_info.get((u, v), 0)
                traffic = self.traffic_info.get((u, v), 0)
                edge_labels[(u, v)] = f"Dist: {original_weight} Trf: {traffic}"
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=7, verticalalignment="baseline", label_pos=0.5)
            
        if(ch==2):
            edge_labels = {}
            for u, v in G.edges():
                if (u, v) in self.amenities or (v, u) in self.amenities:
                    amenities_text = "\n".join(self.amenities.get((u, v), []))

                    edge_labels[(u, v)] = amenities_text

            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=7, verticalalignment="top", label_pos=0.5, clip_on=False)
        plt.show()
   

    """
    For find_amenities_along_path the overall time ccomplexity is
    O(k+m+n) where k represent the number of vertices in the shortest path
    between the source and destination.  m represents the number of edges in the 
    graph that have associated amenities. n represents the number of amenities found
    along the path
    """
    def find_amenities_along_path(self, path, amenity_type):
        amenities_along_path = []
        for i in range(len(path) - 1):
            from_vertex = path[i]
            to_vertex = path[i + 1]
            if (from_vertex, to_vertex) in self.amenities:
                for amenity in self.amenities[(from_vertex, to_vertex)]:
                    if amenity_type in amenity.lower():
                        amenities_along_path.append((from_vertex, to_vertex, amenity))
            if (to_vertex, from_vertex) in self.amenities:
                for amenity in self.amenities[(to_vertex, from_vertex)]:
                    if amenity_type in amenity.lower():
                        amenities_along_path.append((to_vertex, from_vertex, amenity))
        return amenities_along_path

    def save_graph(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_graph(cls, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            return cls()
        
    """
    Here also the dominating time complexity is the Dijkstra's algorithm
    so the overall time complexity will be O((V+E)logV.
    """
    
    def shortest_path_with_stops(self, start, end, stops=[]):
        if start not in self.vertex_indices or end not in self.vertex_indices:
            print("\nStart or end node not found in the graph.\n")
            return None

        # Create a new graph with only the vertices and edges along the path
        G = nx.Graph()
        for i in range(len(self.vertex_names)):
            for j in range(i + 1, len(self.vertex_names)):
                if self.adjacency_matrix[i][j] != float('inf'):
                    G.add_edge(self.vertex_names[i], self.vertex_names[j], weight=self.adjacency_matrix[i][j])

        # Add stops as temporary vertices to the graph
        for stop in stops:
            G.add_node(stop)

        # Find shortest path using Dijkstra's algorithm
        path = nx.dijkstra_path(G, start, end, weight='weight')

        # Insert stops into the path
        for stop in stops:
            if stop in path:
                print(f"\nStop '{stop}' is already in the shortest path.\n")
            else:
                for i in range(len(path) - 1):
                    if self.adjacency_matrix[self.vertex_indices[path[i]]][self.vertex_indices[stop]] != float('inf') \
                        and self.adjacency_matrix[self.vertex_indices[stop]][self.vertex_indices[path[i+1]]] != float('inf'):
                        path.insert(i+1, stop)
                        break
        return path
    """
    Here the time complexity of the change_traffic is O(1) because it performs 
    a constant number of dictionary lookups and updates
    """
    def change_traffic(self, from_vertex, to_vertex, new_traffic):
        if (from_vertex, to_vertex) in self.traffic_info:
            # Update the traffic information only, not the edge weight
            self.traffic_info[(from_vertex, to_vertex)] = new_traffic
        else:
            print(f"No traffic information found for edge ({from_vertex}, {to_vertex}).")
            
        if (to_vertex, from_vertex) in self.traffic_info:
            # Update the traffic information only, not the edge weight
            self.traffic_info[(to_vertex, from_vertex)] = new_traffic
        else:
            print(f"No traffic information found for edge ({to_vertex}, {from_vertex}).")

        g.save_graph('graph_state.pkl')

"""
Here the time complexity for admin side is O(V+E). Where V is the number of vertices
and E is the number of edges in the graph.
"""
def admin_side():
    print("\nOptions:")
    print("1. Show map\n2. Insert a new place\n3. Delete a place\n4. Insert a new hotel/petrol pump\n5. Update traffic info\n6. Log out")
    ch = 0
    while ch != 6:
        try:
            ch = int(input("Enter your choice: "))
        except ValueError:
            print("Invalid input! Please enter a number.")
            continue

        if ch == 1:
            g.show_vertices_and_edges(ch)
        elif ch == 2:
            place_1 = input("Enter new place name: ")
            place_2 = input("Enter connection place: ")
            try:
                distance = int(input("Enter distance between them: "))
                traffic = int(input("Enter traffic: "))
            except ValueError:
                print("Invalid input! Distance and traffic should be integers.")
                continue
            g.add_vertex(place_1)
            g.add_edge_with_traffic(place_1, place_2, distance, traffic)
            sub_ch = input("Any other connections to be added? (Y/N): ")
            while sub_ch.lower() != 'n':
                place_2 = input("Enter connection place: ")
                try:
                    distance = int(input("Enter distance between them: "))
                    traffic = int(input("Enter traffic: "))
                except ValueError:
                    print("Invalid input! Distance and traffic should be integers.")
                    continue
                g.add_edge_with_traffic(place_1, place_2, distance, traffic)
                sub_ch = input("Any other connections to be added? (Y/N): ")
            g.save_graph('graph_state.pkl')
            print(" ")
            
        elif ch == 3:
            place = input("Enter the name of the place to be deleted: ")
            if place in g.vertex_names.values():
                g.delete_vertex(place)
                print("\nPlace deleted..\n")
                g.save_graph('graph_state.pkl')
            else:
                print(f"Place '{place}' does not exist.")
        elif ch == 4:
            place_1 = input("Enter location 1: ")
            place_2 = input("Enter location 2: ")
            try:
                amenities_cnt = int(input("Enter number of amenities to be added: "))
            except ValueError:
                print("Invalid input! Number of amenities should be an integer.")
                continue
            amenities_list = []
            for i in range(amenities_cnt):
                amenities = input("Enter the amenities available: ")
                amenities_list.append(amenities)
            g.add_amenities(place_1, place_2, amenities_list)
            g.save_graph('graph_state.pkl')

        elif ch == 5:
            place_1 = input("Enter location 1: ")
            place_2 = input("Enter location 2: ")
            if place_1 not in g.vertex_names.values() or place_2 not in g.vertex_names.values():
                print("Invalid input! One or both of the entered locations do not exist.")
                continue
            try:
                new_traffic = int(input("Enter new traffic: "))
            except ValueError:
                print("Invalid input! Traffic must be an integer.")
                continue
            g.change_traffic(place_1, place_2, new_traffic)
            g.save_graph('graph_state.pkl')  # Save the updated graph state
            print("New traffic info is updated..\n")
        
        elif ch == 6:
            print("\nLogging out...\n")
            break
        else:
            print("\nInvalid choice! Enter again..")
    main_screen()

"""
Here the time complexity of user_side is O(V+E) or O(P+A). Where is E is number of edges,
V is the number of vertices, P is the length of the path, and A is the number of amenities 
along the path.
"""

def user_side():
    print("\nOptions:")
    print("1. Show map without amenities\n2. Show map with amenities\n3. Find route\n4. Logout")
    ch = 0
    while ch != 4:
        try:
            ch = int(input("Enter your choice: "))
        except ValueError:
            print("Invalid input! Please enter a number.")
            continue
            
        if ch == 1 or ch == 2:
            g.show_vertices_and_edges(ch)
        elif ch == 3:
            # Check if the entered source and destination are valid places
            valid_places = list(g.vertex_names.values())
            start = input("Enter source place: ")
            if start not in valid_places:
                print("Error: Invalid source place.")
                continue
            end = input("Enter destination place: ")
            if end not in valid_places:
                print("Error: Invalid destination place.")
                continue

            # Ask whether to add a stop
            add_stop = input("Do you want to add a stop between source and destination? (Y/N): ").strip().lower()
            stops = []
            while add_stop == 'y':
                stop = input("Enter the name of the stop: ")
                if stop not in valid_places:
                    print("Error: Invalid stop place.")
                    continue
                stops.append(stop)
                add_stop = input("Do you want to add another stop? (Y/N): ").strip().lower()
            edge_list = []
            for i in range(len(g.vertex_names)):
                for j in range(i + 1, len(g.vertex_names)):
                    if g.adjacency_matrix[i][j] != float('inf'):
                        edge_list.append((g.vertex_names[i], g.vertex_names[j], g.adjacency_matrix[i][j]))
            G = nx.Graph()
            G.add_weighted_edges_from(edge_list)

            if len(stops) > 0:
                path = g.shortest_path_with_stops(start, end, stops)
            else:
                path = nx.shortest_path(G, start, end, weight='weight')

            g.visualize_shortest_path(start, end, stops)
            
            sub_ch = input("Do you want to find nearest hotels/petrol pumps/restaurants? (Y/N): ")

            if sub_ch.lower() == 'y':
                print("Find")
                print("1. Hotels\n2. Restaurants\n3. Petrol pumps\n4. Go back to menu")
                sub_ch = 0
                while sub_ch != 4:
                    try:
                        sub_ch = int(input("Enter choice: "))
                    except ValueError:
                        print("Invalid input! Please enter a number.")
                        continue
                    if sub_ch in [1, 2, 3]:
                        amenity_types = {1: 'hotel', 2: 'restaurant', 3: 'petrol pump'}
                        amenity_type = amenity_types[sub_ch]
                        amenities_along_path = g.find_amenities_along_path(path, amenity_type)
                        if amenities_along_path:
                            print(f"Nearest {amenity_type}s along the path from {start} to {end}:")
                            for from_vertex, to_vertex, amenity in amenities_along_path:
                                print(f"- {amenity} between {from_vertex} and {to_vertex}")
                        else:
                            print(f"No {amenity_type}s found along the path from {start} to {end}.")
                    elif sub_ch == 4:
                        print("\nGoing back to menu.\n")
                        break
                    else:
                        print("\nInvalid choice! Enter again..")
        elif ch == 4:
            print("\nLogging out...\n")
            break
        else:
            print("\nInvalid choice! Enter again..")
    main_screen()



"""
For the main_screen also the time complexity is O(V+E).
"""
def main_screen():
    global g
    g = Graph.load_graph('graph_state.pkl')

    print("Main Menu:")
    print("1. Admin Login\n2. User Login\n3. Exit")
    ch = 0
    while ch != 3:
        ch = int(input("Enter your choice: "))
        if ch == 1:
            passwd = input("Enter password: ")
            errcnt = 3
            if passwd == "admin":
                print("\nLogin success!\n")
                admin_side()
            else:
                errcnt -= 1
                print("\nInvalid password! Tries left", errcnt)
                
                while errcnt >= 1:
                    passwd = input("Enter password again: ")
                    if passwd == "admin":
                        break
                    else:
                        errcnt -= 1
                        print("\nInvalid password! Tries left", errcnt)

                if errcnt <= 0:
                    print("\nUnauthorized user. System locked\n")
                else:
                    print("\nLogin success!\n")
                    admin_side()
        elif ch == 2:
            user_side()
           
        elif ch == 3:
            print("\nExiting the system...\n")
            g.save_graph('graph_state.pkl')
            break
        else:
            print("\nInvalid choice! Enter again..")


# Example usage:
g = Graph.load_graph('graph_state.pkl')
if not g.vertex_names:
    g.add_vertex('Coimbatore')
    g.add_vertex('Tiruppur')
    g.add_vertex('Erode')
    g.add_vertex('Salem')
    g.add_vertex('Palakkad')
    g.add_vertex('Ooty')
    g.add_vertex('Mettupalayam')

    g.add_edge_with_traffic('Coimbatore', 'Tiruppur', 20,3)
    g.add_edge_with_traffic('Coimbatore', 'Erode', 40,9)
    g.add_edge_with_traffic('Coimbatore', 'Salem', 60,0)
    g.add_edge_with_traffic('Coimbatore', 'Palakkad', 70,4)
    g.add_edge_with_traffic('Coimbatore', 'Ooty', 100,8)
    g.add_edge_with_traffic('Coimbatore', 'Mettupalayam', 30,8)
    g.add_edge_with_traffic('Tiruppur', 'Erode', 50,3)
    g.add_edge_with_traffic('Erode', 'Salem', 40,6)
    g.add_edge_with_traffic('Salem', 'Palakkad', 80,4)
    g.add_edge_with_traffic('Palakkad', 'Ooty', 120,6)
    g.add_edge_with_traffic('Palakkad', 'Mettupalayam', 90,4)
    g.add_edge_with_traffic('Ooty', 'Mettupalayam', 50,9)

    g.add_amenities('Coimbatore', 'Tiruppur', ['Restaurant A', 'Hotel B'])
    g.add_amenities('Coimbatore', 'Erode', ['Petrol Pump C', 'Hotel D'])
    g.add_amenities('Coimbatore', 'Salem', ['Restaurant E', 'Shop F'])
    g.add_amenities('Coimbatore', 'Palakkad', ['Hotel G', 'Petrol Pump H'])
    g.add_amenities('Coimbatore', 'Ooty', ['Restaurant I', 'Hotel J'])
    g.add_amenities('Coimbatore', 'Mettupalayam', ['Shop K', 'Restaurant L'])
    g.add_amenities('Tiruppur', 'Erode', ['Hotel M', 'Petrol Pump N'])
    g.add_amenities('Erode', 'Salem', ['Restaurant O', 'Hotel P'])
    g.add_amenities('Salem', 'Palakkad', ['Shop Q', 'Restaurant R'])
    g.add_amenities('Palakkad', 'Ooty', ['Restaurant S', 'Hotel T'])
    g.add_amenities('Palakkad', 'Mettupalayam', ['Shop U', 'Restaurant V'])
    g.add_amenities('Ooty', 'Mettupalayam', ['Hotel W', 'Petrol Pump X'])
    
g.save_graph('graph_state.pkl')
print("\nWelcome to GPS Navigation System\n")
main_screen()