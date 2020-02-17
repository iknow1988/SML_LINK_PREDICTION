from  utility import *
import numpy as np
import pandas as pd
import csv
import networkx as nx
import random
random.seed(9001)
import matplotlib.pyplot as plt

# ****************** Data Structure for Node and Edges *****************
#represents an edge
class Edge:
  def __init__(self, source, sink):
    self.source = source
    self.sink = sink

#represents node
class Node:
    def __init__(self, id):
        self.id = id
        self.inNodes = set()
        self.outNodes = set()

    def addOutNode(self, node):
        self.outNodes.add(node)

    def addInNode(self, node):
        self.inNodes.add(node)

    def getOutDegree(self):
        return len(self.outNodes)

    def getInDegree(self):
        return len(self.inNodes)

    def isFollower(self, node):
        if node in self.inNodes:
            return True
        else:
            return False

    def isFollowing(self, node):
        if node in self.outNodes:
            return True
        else:
            return False

    def getCommonFollowersCount(self, node):
        return len(self.inNodes.intersection(node.inNodes))

    def getCommonFolloweesCount(self, node):
        return len(self.outNodes.intersection(node.outNodes))

    def getFollowBack(self, node):
        if self.id in node.outNodes:
            return 1
        else:
            return 0

# ****************** Data Structure for Node and Edges ENDS*****************


# A node mapper to optimize the space
def node_mapper(fileName):
    nodeMapper = {}
    antiNodeMapper = set()
    test_nodes = load_obj('test_nodes')
    with open(fileName) as f:
        content = f.readlines()
        print("Size of text file:", len(content))
        for line in content:
            # print(line)
            line = line.strip()
            nodes = line.split("\t")
            source = nodes[0]
            if source in nodeMapper:
                source = nodeMapper[source]
            else:
                temp = len(nodeMapper)
                nodeMapper[source] = temp
                source = temp
                antiNodeMapper.add(source)
            for sink in nodes[1:]:
                if sink in nodeMapper:
                    sink = nodeMapper[sink]
                else:
                    temp = len(nodeMapper)
                    nodeMapper[sink] = temp
                    sink = temp
                    antiNodeMapper.add(sink)
    save_obj(nodeMapper,'nodeMapper')
    save_obj(antiNodeMapper,'nodes')


# Find nodes connected to test node pairs
def related_nodes(fileName):
    all_nodes = set()
    test_nodes = load_obj('test_nodes_sources')
    nodeMapper = load_obj('nodeMapper')
    with open(fileName) as f:
        content = f.readlines()
        print("Size of text file:", len(content))
        for index, line in enumerate(content):
            # print(line)
            line = line.strip()
            nodes = line.split("\t")
            source = nodeMapper[nodes[0]]
            print(index)
            if source in test_nodes:
                all_nodes.add(source)
                for sink in nodes[1:]:
                    sink = nodeMapper[sink]
                    all_nodes.add(sink)
    print(len(all_nodes))
    save_obj(all_nodes,'nodes')

# Creates a list of all edges in the graph with optimized node mapping
def read_file(fileName):
    edges=list()
    nodeMapper = load_obj('nodeMapper')
    testNodes = load_obj('nodes_test')
    with open(fileName) as f:
        content = f.readlines()
        print("Size of text file:", len(content))
        for index, line in enumerate(content):
            line = line.strip()
            nodes = line.split("\t")
            source = nodeMapper[nodes[0]]
            print(index)
            for sink in nodes[1:]:
                sink = nodeMapper[sink]
                edge = Edge(source,sink)
                edges.append(edge)
    print("len of edges:",len(edges))
    save_obj(edges, 'edges')


# creates a dictionary of directed edges of full training graph nodes with encoding
def create_adjacency_list_with_mapping():
    nodes = {}
    edges = load_obj("edges")
    print(len(edges))
    for edge in edges:
        source = edge.source
        sink = edge.sink
        if source not in nodes:
            nodes[source] = list()
        nodes[source].append(sink)

    print(len(nodes))
    save_obj(nodes,"adjacency_list")


# Creates a dictionary of object of 'Node' class for every nodes of graph
def create_dictionary_of_nodes_with_node_properties():
    nodes = {}
    edges = load_obj("edges")
    testNodes = load_obj("nodes_test")
    print(len(edges))
    count = 1
    for edge in edges:
        sourceId = edge.source
        sinkId = edge.sink
        if sourceId not in nodes:
            nodes[sourceId] = Node(sourceId)
        if sinkId not in nodes:
            nodes[sinkId] = Node(sinkId)
        nodes[sourceId].addOutNode(sinkId)
        nodes[sinkId].addInNode(sourceId)
        count = count + 1
    print(len(nodes))
    save_obj(nodes, 'node_properties')


# Creates a list of all edges in the whole training graph
def create_list_all_edges_training_graph_with_features(fileName):
    nodeMapper = load_obj('nodeMapper')
    df = list()
    nodes = load_obj('node_properties')
    adjlist = load_obj("adjacency_list")
    print(len(nodes))
    pr = load_obj('pr')
    testNodes = load_obj('nodes_test')
    count = 0
    for sourceId, sinks in adjlist.items():
        source = nodes[sourceId]
        for sinkId in sinks:
            print(count)
            sink = nodes[sinkId]
            followees = source.outNodes.difference({sinkId})
            already = 0
            for followeeId in followees:
                followee = nodes[followeeId]
                if sinkId in followee.outNodes:
                    already = already + 1
            df.append([sourceId, sinkId, source.getInDegree(), source.getOutDegree(), sink.getInDegree(),
                             sink.getOutDegree(), source.getCommonFollowersCount(sink),
                             source.getCommonFolloweesCount(sink), source.getFollowBack(sink), already, pr[sourceId], pr[sinkId]])
            count = count + 1
    print(count)
    save_obj(df, 'train_orig_short')


# Create dictionary of test nodes
def create_test_nodes(fileName):
    nodeMapper = load_obj('nodeMapper')
    nodes = set()
    sources = set()
    sinks = set()
    i = 0
    df = list()
    with open(fileName) as f:
        for line in f:
            print(i)
            if (i != 0):
                line = line.strip()
                ids = line.split("\t")
                sourceId = nodeMapper[ids[1]]
                sinkId = nodeMapper[ids[2]]
                nodes.add(sourceId)
                nodes.add(sinkId)
                sources.add(sourceId)
                sinks.add(sinkId)
            i = i + 1
    print(len(nodes), len(sources), len(sinks))
    save_obj(nodes,'test_nodes')
    save_obj(sources, 'test_nodes_sources')
    save_obj(sinks, 'test_nodes_sinks')


# Take nodes only present in test dataset
def create_dictionary_of_nodes_in_test_dataset():
    refined = {}
    nodes_prop = load_obj('node_properties')
    test_nodes = load_obj('test_nodes')
    for node in test_nodes:
        if node not in refined:
            refined[node] = nodes_prop[node]
    save_obj(refined, 'node_properties_refined')


# Creates list of edges in test dataset with features
def create_test_dataset_list(fileName):
    nodeMapper = load_obj('nodeMapper')
    nodes = load_obj('node_properties')
    pr = load_obj('pr_whole')
    i = 0
    df = list()
    with open(fileName) as f:
        for line in f:
            print(i)
            if (i != 0):
                line = line.strip()
                ids = line.split("\t")
                sourceId = nodeMapper[ids[1]]
                sinkId = nodeMapper[ids[2]]
                if sourceId in nodes:
                    source = nodes[sourceId]
                else:
                    source = Node(sourceId)
                if sinkId in nodes:
                    sink = nodes[sinkId]
                else:
                    sink = Node(sinkId)
                followees = source.outNodes.difference({sinkId})
                already = 0
                for followeeId in followees:
                    followee = nodes[followeeId]
                    if sinkId in followee.outNodes:
                        already = already + 1

                source_pr = 0
                sink_pr = 0
                if(sourceId in pr):
                    source_pr = pr[sourceId]
                if (sinkId in pr):
                    sink_pr = pr[sinkId]

                commons = source.inNodes.intersection(sink.inNodes)
                aa = 0
                if len(commons) > 0:
                    for common in commons:
                        cm = nodes[common]
                        ind = cm.getInDegree() + cm.getOutDegree()
                        if ind > 0:
                            aa = aa + 1.0 / np.log(ind)
                else:
                    aa = -123

                df.append([sourceId, sinkId, source.getInDegree(), source.getOutDegree(), sink.getInDegree(),
                           sink.getOutDegree(), source.getCommonFollowersCount(sink),
                           source.getCommonFolloweesCount(sink), source.getFollowBack(sink), already, source_pr,
                           sink_pr, aa])
            i = i + 1
    save_obj(df, 'dataset_test')


# Creates dataframe of test dataset from list
def create_test_dataset_dataframe():
    df = pd.DataFrame(columns=['index', 'source', 'sink',
                               'source_inDegree', 'source_outDegree',
                               'sink_inDegree', 'sink_outDegree',
                               'common_followers', 'common_followees'])
    data = load_obj('dataset_test')
    print(len(data))
    index = 0
    for value in data:
        print(value)
        df.loc[index] = value
        index = index + 1
    save_obj(df, 'df_test')

# Creates fake edges with features
def create_dataframe_fake_edges_in_graph_with_features():
    all_nodes = load_obj("nodes_test")
    sources = load_obj("nodes_test_sources")
    sinks = load_obj("nodes_test_sinks")
    nodes_props = load_obj('node_properties')
    df_noedge = pd.DataFrame(columns=['source', 'sink',
                                      'source_inDegree', 'source_outDegree',
                                      'sink_inDegree', 'sink_outDegree',
                                      'common_followers', 'common_followees'])
    index = 0
    count = 0
    for sourceId in sources:
        source = nodes_props[sourceId]
        possible_nodes = sinks.difference(source.outNodes)
        match = len(possible_nodes)
        if match > 50:
            possible_nodes = random.sample(possible_nodes, 50)

        for sinkId in possible_nodes:
            print(count)
            sink = nodes_props[sinkId]
            value = [sourceId, sinkId, source.getInDegree(), source.getOutDegree(), sink.getInDegree(),
                     sink.getOutDegree(), source.getCommonFollowersCount(sink),
                     source.getCommonFolloweesCount(sink)]
            df_noedge.loc[index] = value
            index = index + 1
            count = count + 1
    save_obj(df_noedge, 'df_train_fake')


# Fake dataset creation (first try)
def create_dataframe_fake_edges_in_graph_with_features_new():
    nodes = load_obj("nodes_test")
    sources = load_obj("nodes_test_sources")
    sinks = load_obj("nodes_test_sinks")
    nodes_props = load_obj('node_properties')
    nodes_test_dict = load_obj('nodes_test_dict')
    pr = load_obj('pr')
    df = list()
    count = 0
    for sinkId in sinks:
        if sinkId in nodes_props:
            sink = nodes_props[sinkId]
        else:
            sink = Node(sinkId)
        possible_nodes = sources.difference(sink.inNodes).difference(set(nodes_test_dict[sinkId]))
        print(len(possible_nodes))
        possible_nodes = random.sample(possible_nodes, 150)
        print(sinkId, len(possible_nodes))
        for sourceId in possible_nodes:
            if sourceId in nodes_props:
                source = nodes_props[sourceId]
            else:
                source = Node(sourceId)
            followees = source.outNodes.difference({sinkId})
            already = 0
            for followeeId in followees:
                if followeeId in nodes_props:
                    followee = nodes_props[followeeId]
                else:
                    followee = Node(followeeId)
                if sinkId in followee.outNodes:
                    already = already + 1
            if sourceId in pr:
                source_pr = pr[sourceId]
            if sinkId in pr:
                sink_pr = pr[sinkId]
            value = [sourceId, sinkId, source.getInDegree(), source.getOutDegree(), sink.getInDegree(),
                     sink.getOutDegree(), source.getCommonFollowersCount(sink),
                     source.getCommonFolloweesCount(sink), source.getFollowBack(sink), already, source_pr, sink_pr]
            df.append(value)
            count = count + 1
            # print(count)
    print(count)
    save_obj(df, 'train_fake')


# ************************* SCC feature ***********************************************


def scc_feature(df_training, name):
    scc = load_obj('scc')
    # merge negative and positive samples
    df = list()
    for index, row in df_training.iterrows():
        sourceId = row['source']
        sinkId = row['sink']
        if sourceId in scc and sinkId in scc:
            df.append(1)
        elif sourceId in scc and sinkId not in scc:
            df.append(2)
        elif sourceId not in scc and sinkId in scc:
            df.append(3)
        else:
            df.append(4)
    df_training['scc'] = df
    save_obj(df_training, name)


def scc_test():
    scc = load_obj('scc')
    df_test = load_obj('df_test')
    df = list()
    for index, row in df_test.iterrows():
        sourceId = row['source']
        sinkId = row['sink']
        if sourceId in scc and sinkId in scc:
            df.append(1)
        elif sourceId in scc and sinkId not in scc:
            df.append(2)
        elif sourceId not in scc and sinkId in scc:
            df.append(3)
        else:
            df.append(4)
    df_test['scc'] = df
    save_obj(df_test,'df_test')


def add_scc():
    scc_feature(load_obj('df_train'),'df_train')
    scc_feature(load_obj('df_train_fake'), 'df_train_fake')
    scc_test()


# ************************* PR feature ***********************************************


def pr_feature(df_training, name):
    pr = load_obj('pr')
    # merge negative and positive samples
    df = list()
    for index, row in df_training.iterrows():
        sourceId = row['source']
        sinkId = row['sink']
        df_training.at[index, 'source_pr'] = pr[sourceId]
        df_training.at[index, 'sink_pr'] = pr[sinkId]
    save_obj(df_training, name)


def pr_test():
    pr = load_obj('pr_second_2')
    df_test = load_obj('df_test')
    df = list()
    for index, row in df_test.iterrows():
        sourceId = row['source']
        sinkId = row['sink']
        df_test.at[index, 'source_pr'] = pr[sourceId]
        df_test.at[index, 'sink_pr'] = pr[sinkId]
    save_obj(df_test,'df_test')


def add_pr():
    pr_feature(load_obj('df_train'),'df_train')
    pr_feature(load_obj('df_train_fake'), 'df_train_fake')
    pr_test()

# ************************* NETWORKX ***********************************************

# Create graph on networkx
def create_networkx_graph():
    mapper = set()
    nodes = load_obj('nodes_test')
    adjList = load_obj('adjacency_list')
    nodes_props = load_obj('node_properties')
    DG = nx.DiGraph()
    count = 0
    for sourceId, edges in adjList.items():
        print(count)
        count = count + 1
        source = nodes_props[sourceId]
        if sourceId not in mapper:
            DG.add_node(sourceId)
            mapper.add(sourceId)
        possible_nodes = source.outNodes
        for sinkId in possible_nodes:
            if sinkId not in mapper:
                DG.add_node(sinkId)
                mapper.add(sinkId)
            DG.add_edge(sourceId, sinkId)
    print(len(mapper))
    save_obj(DG, 'DG')

	
# Add new nodes to graph
def add_new_nodes():
    DG = load_obj('DG_train_second')
    mapper = set()
    sources = load_obj('nodes_test_sources')
    all_nodes = load_obj('nodes_test')
    nodes_props1 = load_obj('node_properties_refined')
    nodes_props = load_obj('node_properties_refined_second_2')
    count = 0
    for sourceId in sources:
        if sourceId not in nodes_props:
            source = nodes_props1[sourceId]
        else:
            source = nodes_props[sourceId]
        print(count, len(source.outNodes))
        for sinkId in source.outNodes:
            if sinkId in all_nodes:
                continue
            else:
                if sinkId not in mapper:
                    mapper.add(sinkId)
                    DG.add_node(sinkId)
            DG.add_edge(sourceId, sinkId)
        count = count + 1
    save_obj(DG, 'DG_train_second_2')


# creates a dataframe given a list
def create_dataframe_from_list():
    df_train = pd.DataFrame(columns=['source_inDegree', 'source_outDegree',
                                      'sink_inDegree', 'sink_outDegree',
                                      'common_followers', 'common_followees', 'followback'])
    df_list = load_obj('new_dataset_list_new')
    count = 0
    for index, row in enumerate(df_list):
        print(count)
        df_train.loc[index] = row
        count = count + 1
    save_obj(df_train,'df_train_new_without_id')


def create_test_dict(fileName):
    nodeMapper = load_obj('nodeMapper')
    nodes = set()
    sources = set()
    sinks = set()
    i = 0
    df = list()
    dd = {}
    with open(fileName) as f:
        for line in f:
            # print(i)
            if (i != 0):
                line = line.strip()
                ids = line.split("\t")
                sourceId = nodeMapper[ids[1]]
                sinkId = nodeMapper[ids[2]]
                if sinkId not in dd:
                    print (sinkId, sourceId)
                    dd[sinkId] = list()
                    dd[sinkId].append(sourceId)
                else:
                    dd[sinkId].append(sourceId)
                nodes.add(sourceId)
                nodes.add(sinkId)
                sources.add(sourceId)
                sinks.add(sinkId)
            i = i + 1
    print(len(nodes), len(sources), len(sinks))
    save_obj(dd,'nodes_test_dict')

def createDegreeDictionary():
    inDegreeDict = {}
    outDegreeDict = {}
    refined = {}
    nodes_prop = load_obj('node_properties')
    test_nodes = load_obj('nodes')
    for key, node in nodes_prop.items():
        inDegree = node.getInDegree()
        outDegree = node.getOutDegree()
        if inDegree not in inDegreeDict:
            inDegreeDict[inDegree] = list()
        inDegreeDict[inDegree].append(key)

        if outDegree not in outDegreeDict:
            outDegreeDict[outDegree] = list()
        outDegreeDict[outDegree].append(key)
        if key in test_nodes:
            refined[key] = node
    save_obj(refined, 'node_properties_refined_2')
    save_obj(inDegreeDict, 'inDegreeDict')
    save_obj(outDegreeDict, 'outDegreeDict')

def main():
    # related_nodes("data/train.txt")
    # node_mapper("data/train.txt")
    # read_file("data/train.txt")
    # create_adjacency_list_with_mapping()
    # create_dictionary_of_nodes_with_node_properties()
    # create_list_all_edges_training_graph_with_features("data/train.txt")
    # create_test_nodes("data/test-public.txt")
    # create_dictionary_of_nodes_in_test_dataset()
    # create_test_dataset_list("data/test-public.txt")
    # create_test_dataset_dataframe()
    # create_dataframe_valid_edges_in_graph_with_features()

    # add_follow_back_feature()
    # add_triadic_closure_feature()
    # add_triadic_closure_feature_2()
    # add_scc()
    # add_pr()

    # possible_nodes()
    # create_networkx_graph()
    # possible_nodes_1(4)
    # possible_nodes_2()

    # valid_edges_in_graph_with_features()
    # node_properties_second_refined()
    # valid_edges_in_graph_with_features_new_dataset()
    # add_new_nodes()
    # create_dataframe_from_list()
    # df = load_obj('new_dataset_list')
    # print(len(df))
    # create_dataframe_fake_edges_in_graph_with_features_new()
    # valid_edges_in_graph_with_features_new_dataset
    # create_test_dict("data/test-public.txt")
    # pr_test()

    # valid_edges_in_graph_with_features_new_dataset_2()
    # node_properties_second_refined()
    # test = load_obj('node_properties_refined_second_2')
    # print(len(test))
    # valid_edges_in_graph_with_features_new_dataset_2()

    # print(len(load_obj('node_properties')))
    # print(len(load_obj('nodes_test')))
    # print("Done")
    # adj = load_obj("adjacency_list")
    # sources = load_obj('nodes_test_sources')
    # count = 0
    # for source,value in adj.items():
    #     if(len(value)<5):
    #         print(len(value))
    #         count  = count + 1
    # print(count)
    # create_new_logic_train("data/test-public.txt")
    # create_new_logic_train_sink("data/test-public.txt")
    # create_new_logic_all("data/test-public.txt")
    # create_new_logic_all_fake("data/test-public.txt")
    # print(len(load_obj('not_found')))
    createDegreeDictionary()
    # print(len(load_obj('outDegreeDict')))
if __name__ == '__main__':
    main()