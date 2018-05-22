from graph_tool.all import *
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle


client = boto3.client('s3')


g = Graph()


#add the property to vertex object
vprop = g.new_vertex_property("string") 

from_ = pd.Series([1,1,1,2,5,4])
to_ = pd.Series([2,3,4,3,4,1])
v_ = set(list(from_)+list(to_))
print(v_)

for v in v_:
	vertex_x = g.add_vertex()
	vprop[vertex_x] = str(v)
	from_.replace(v, vertex_x)
	to_.replace(v, vertex_x)


#assign properties as a dic value
g.vertex_properties["name"]=vprop 

print('adding edge')
e = g.add_edge_list(list(zip(from_,to_)))

print('betweeness')

bv, be = betweenness(g)

be.a /= (be.a.max() / 5)

print('drawing')
graph_draw(g,  vertex_text=g.vertex_properties["name"], vertex_fill_color=bv, edge_pen_width=be,
           output="filtered-bt.svg")


