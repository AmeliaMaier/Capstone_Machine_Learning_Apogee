from graph_tool.all import *
import pandas as pd
import matplotlib.pyplot as plt

d=8
b=3544
sp=1952
root = 'subgraphs/'
depth = f'depth_{d}/'
bredth = f'bredth_{b}/'
file = f'starting_point_{sp}.csv'
path = root+depth+bredth+file
edges_df = pd.read_csv(path)

g = graph_tool.Graph()


#add the property to vertex object
vprop = g.new_vertex_property("string") 

print('Create set of vertexs')
#Create set of vertexs
from_ = edges_df.from_url_id
to_ = edges_df.to_url_id
v_ = set(list(from_)+list(to_))

print('adding vertexs')
for v in v_:
	vertex_x = g.add_vertex()
	vprop[vertex_x] = str(v)
	from_.replace(v, vertex_x)
	to_.replace(v, vertex_x)


#assign properties as a dic value
g.vertex_properties["name"]=vprop

#add edges
print('adding edge')
e = g.add_edge_list(list(zip(from_,to_)))


print('betweeness')
bv, be = betweenness(g)
be.a /= be.a.max() / 5

print('drawing')
graph_draw(g,  vertex_text=g.vertex_properties["name"], vertex_fill_color=bv, edge_pen_width=be,
           output="filtered-bt.svg")
