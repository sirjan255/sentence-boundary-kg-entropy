import pickle
import matplotlib.pyplot as plt
import networkx as nx

# Load the pickled graph
with open('outputs/kg.pkl', 'rb') as f:
    G = pickle.load(f)

plt.figure(figsize=(20, 15))  # Large figure for the whole graph

# Layout
pos = nx.spring_layout(G, seed=42, k=0.5)  # k controls node spacing

# Draw nodes and edges
nx.draw(
    G, pos, with_labels=True, node_color='skyblue', 
    edge_color='gray', node_size=2000, font_size=9, alpha=0.85
)

# Draw edge labels if your edges have a 'verb' or 'label' attribute
edge_attr = 'verb' if all('verb' in d for _,_,d in G.edges(data=True)) else (
    'label' if all('label' in d for _,_,d in G.edges(data=True)) else None
)
if edge_attr:
    edge_labels = nx.get_edge_attributes(G, edge_attr)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

plt.title("Full Knowledge Graph")
plt.tight_layout()
plt.show()