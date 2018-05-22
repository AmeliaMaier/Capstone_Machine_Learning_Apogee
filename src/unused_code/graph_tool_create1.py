from graph_tool.all import *
import pandas as pd
import boto3
import pickle
import os
from io import StringIO
import numpy as np

client = boto3.client('s3')

#d=8
#b=3544
#sp=1952
#root = 'subgraphs/'
#depth = f'depth_{d}/'
#bredth = f'bredth_{b}/'
#file = f'starting_point_{sp}.csv'
paths = ['subgraphs/depth_0/bredth_25/starting_point_43211.csv', 'subgraphs/depth_0/bredth_5/starting_point_8320.csv','subgraphs/depth_0/bredth_25/starting_point_7602.csv', 'subgraphs/depth_0/bredth_25/starting_point_5029.csv', 'subgraphs/depth_1/bredth_50/starting_point_677194.csv']
#paths = ['subgraphs/depth_4/bredth_142/starting_point_3747.csv']
starting_points = [43211, 8320, 7602, 5029, 677194]
#starting_points = [3747]

for path, sp in zip(paths, starting_points):

    print('reading file')
    obj = client.get_object(Bucket='websitelinksigraph', Key=path)
    body = obj['Body']
    csv_string = body.read().decode('utf-8')
    edges_df = pd.read_csv(StringIO(csv_string))

    print('create graph object')
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

    print('degree')
    deg = g.degree_property_map("in")
    deg.a = 4 * (np.sqrt(deg.a) * 0.5 + 0.4)

    print('betweeness')
    bv, be = betweenness(g)
    be.a /= be.a.max() / 10.
    eorder = be.copy()
    eorder.a *= -1

    print('layout')
    pos = sfdp_layout(g)

    print('edge_control')
    control = g.new_edge_property("vector<double>")
    for e in g.edges():
        d = np.sqrt(sum((pos[e.source()].a - pos[e.target()].a) ** 2)) / 5
        control[e] = [0.3, d, 0.7, d]

    print('drawing')
    graph_draw(g,  vertex_text=g.vertex_properties["name"],vertex_size=deg,
        vertex_fill_color=deg, vorder=deg, edge_color=ebet, eorder=eorder,
        edge_pen_width=ebet, edge_control_points=control, # some curvy edges
           output=f"graph-draw-{sp}-ran.pdf")
    print('drawing')
    graph_draw(g, pos=pos, vertex_text=g.vertex_properties["name"],vertex_size=deg,
        vertex_fill_color=deg, vorder=deg, edge_color=ebet, eorder=eorder,
        edge_pen_width=ebet, edge_control_points=control, # some curvy edges
           output=f"graph-draw-{sp}-sfdp.pdf")

    print('creating pickles')
    pickle.dump(g, open(f'g{sp}.pkl', 'wb'))
    pickle.dump(bv, open(f'bv{sp}.pkl', 'wb'))
    pickle.dump(be, open(f'be{sp}.pkl', 'wb'))


#print('saving image to bucket')
#my_bucket.upload_file(f'filtered-bt-{sp}.svg', Key=f'filtered-bt-{sp}.svg')

#print('saving and moving bv')
#my_bucket.upload_file('bv.pkl',Key='bv-{sp}.pkl')

#print('saving and moving be')
#my_bucket.upload_file('be-{sp}.pkl', 'wb')
#print('done')
