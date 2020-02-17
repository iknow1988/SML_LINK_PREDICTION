from  utility import *
import numpy as np
import pandas as pd
import csv
import networkx as nx
import random
random.seed(9001)
rng1 = np.random.RandomState(100)
import matplotlib.pyplot as plt

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

			
# Creates Fake edge list
def create_new_logic_all_fake(fileName):
    edges_all = {}
    nodeMapper = load_obj('nodeMapper')
    nodes = load_obj('node_properties_refined')
    pr = load_obj('pr_whole')
    sources = load_obj('test_nodes_sources')
    sinks = load_obj('test_nodes_sinks')
    inDegreeDict = load_obj('inDegreeDict')
    outDegreeDict = load_obj('outDegreeDict')
    round_number = 2
    all_nodes = load_obj('nodes')
    i = 0
    cnt= 0
    with open(fileName) as f:
        for line in f:
            if (i != 0):
                sample_number = 50
                edges = {}
                line = line.strip()
                ids = line.split("\t")
                sourceId = nodeMapper[ids[1]]
                sinkId = nodeMapper[ids[2]]
                source = nodes[sourceId]
                sink = nodes[sinkId]
                source_pr = round(np.log(pr[sourceId]),round_number)
                sink_pr = round(np.log(pr[sinkId]), round_number)
                source_inDegree = source.getInDegree()
                source_outDegree = source.getOutDegree()
                sink_inDegree = sink.getInDegree()
                sink_outDegree = sink.getOutDegree()
                followers = set(outDegreeDict[source_outDegree])
                followees = set(inDegreeDict[sink_inDegree])
                similar_nodes_sources = list()
                count_sources = 0
                for followerId in followers.difference(sink.inNodes).difference({sourceId}):
                    follower_pr = round(np.log(pr[followerId]),round_number)
                    if source_pr == follower_pr:
                        similar_nodes_sources.append(followerId)
                        count_sources = count_sources + 1
                sample_sources = list()
                need = sample_number
                if sink.getInDegree() < need:
                    need = sink.getInDegree()
                if count_sources == 0:
                    if(len(followers)>need):
                        sample_sources = random.sample(followers.difference(sink.inNodes).difference({sourceId}), need)
                    else:
                        sample_sources = random.sample(all_nodes.difference(sink.inNodes).difference({sourceId}), need)
                elif count_sources >= need:
                    sample_sources = random.sample(similar_nodes_sources, need)
                else:
                    need = need - len(similar_nodes_sources)
                    if need < len(followers.difference(sink.inNodes).difference({sourceId}).difference(similar_nodes_sources)):
                        possible = followers.difference(sink.inNodes).difference({sourceId}).difference(similar_nodes_sources)
                    else:
                        possible = all_nodes.difference(sink.inNodes).difference({sourceId}).difference(
                            similar_nodes_sources)
                    possible = random.sample(possible,need)
                    sample_sources = similar_nodes_sources + possible
                if(len(sample_sources) != sample_number):
                    print(i, "SOURCE:", sink.getInDegree(), len(sample_sources))
                for src in sample_sources:
                    if src not in edges:
                        edges[src] = set()
                    edges[src].add(sinkId)
                count_sinks = 0
                similar_nodes_sinks = list()
                for followeeId in followees.difference(source.outNodes).difference({sinkId}):
                    followee_pr = round(np.log(pr[followeeId]), round_number)
                    if sink_pr == followee_pr:
                        count_sinks = count_sinks + 1
                        similar_nodes_sinks.append(followeeId)
                sample_sinks = list()
                need = sample_number
                if source.getOutDegree() < need:
                    need = source.getOutDegree()
                if count_sinks == 0:
                    if (len(followees) > need):
                        sample_sinks = random.sample(followees.difference(source.outNodes).difference({sinkId}), need)
                    else:
                        sample_sinks = random.sample(all_nodes.difference(source.outNodes).difference({sinkId}), need)
                elif count_sinks >= need:
                    sample_sinks = random.sample(similar_nodes_sinks, need)
                else:
                    need = need - len(similar_nodes_sinks)
                    if (len(followees.difference(source.outNodes).difference({sinkId}).difference(similar_nodes_sinks)) > need):
                        possible = followees.difference(source.outNodes).difference({sinkId}).difference(similar_nodes_sinks)
                    else:
                        possible = all_nodes.difference(source.outNodes).difference({sinkId}).difference(
                            similar_nodes_sinks)
                    possible = random.sample(possible, need)
                    sample_sinks = similar_nodes_sinks + possible
                if (len(sample_sinks) != sample_number):
                    print(i, "SINK:", source.getOutDegree(), len(sample_sinks))
                if sourceId not in edges:
                    edges[sourceId] = set()
                edges[sourceId].update(sample_sinks)
                edges_all[i] = edges
                # print(i, count_sinks,  count_sources)
                if(count_sinks + count_sources == 0):
                    cnt = cnt + 1
                    # print(i, source.getOutDegree(), sink.getInDegree())
            i = i + 1
    print(cnt)
    save_obj(edges_all,'edges_fake_new_3')


# Creates Originial edge list
def create_new_logic_orig(fileName):
    edges_all = {}
    nodeMapper = load_obj('nodeMapper')
    nodes = load_obj('node_properties_refined')
    pr = load_obj('pr_whole')
    testNodes = load_obj('test_nodes')
    inDegreeDict = load_obj('inDegreeDict')
    outDegreeDict = load_obj('outDegreeDict')
    round_number = 1
    sample_number = 50
    i = 0
    cnt= 0
    with open(fileName) as f:
        for line in f:
            if (i != 0):
                edges = {}
                line = line.strip()
                ids = line.split("\t")
                sourceId = nodeMapper[ids[1]]
                sinkId = nodeMapper[ids[2]]
                source = nodes[sourceId]
                sink = nodes[sinkId]
                source_pr = round(np.log(pr[sourceId]),round_number)
                sink_pr = round(np.log(pr[sinkId]), round_number)
                source_inDegree = source.getInDegree()
                source_outDegree = source.getOutDegree()
                sink_inDegree = sink.getInDegree()
                sink_outDegree = sink.getOutDegree()
                similar_nodes_sources_degree = set()
                similar_nodes_sources_pr1 = set()
                similar_nodes_sources_pr2 = set()
                count_sources = 0
                for followerId in sink.inNodes.difference({sourceId}):
                    followers = outDegreeDict[source_outDegree]
                    follower_pr1 = round(np.log(pr[followerId]), round_number)
                    follower_pr2 = int(follower_pr1)
                    if followerId in followers:
                        similar_nodes_sources_degree.add(followerId)
                        count_sources = count_sources + 1
                    if source_pr == follower_pr1:
                        similar_nodes_sources_pr1.add(followerId)
                    if int(source_pr) == follower_pr2:
                        similar_nodes_sources_pr2.add(followerId)

                if count_sources == 0:
                    print(i, ": Source: No degree found ", source_outDegree)
                    count_sources = len(similar_nodes_sources_pr1)
                    similar_nodes_sources_degree = similar_nodes_sources_pr1
                if count_sources == 0:
                    print(i, ": Source: No PR1 found ")
                    count_sources = len(similar_nodes_sources_pr2)
                    similar_nodes_sources_degree = similar_nodes_sources_pr2

                sample_sources = set()
                if count_sources == 0:
                    print(i, ": Source: No PR2 found. In degree is:", sink.getInDegree())
                    if sink.getInDegree() >= sample_number:
                        sample_sources = random.sample(sink.inNodes, sample_number)
                    else:
                        sample_sources = sink.inNodes
                elif count_sources >= sample_number:
                    sample_sources = random.sample(similar_nodes_sources_degree, sample_number)
                else:
                    need = sample_number - len(similar_nodes_sources_degree)
                    possible = sink.inNodes.difference(similar_nodes_sources_degree)
                    print(i, ": Source: Extra needed:", need, " and POSSIBLE:", len(possible))
                    if possible and (len(possible) >= need):
                        possible = random.sample(possible, need)
                    similar_nodes_sources_degree.update(possible)
                    sample_sources = similar_nodes_sources_degree
                for src in sample_sources:
                    if src not in edges:
                        edges[src] = set()
                    edges[src].add(sinkId)

                #*********************************************
                count_sinks = 0
                similar_nodes_sinks_degree = set()
                similar_nodes_sinks_pr1 = set()
                similar_nodes_sinks_pr2 = set()
                for followeeId in source.outNodes.difference({sinkId}):
                    followees = inDegreeDict[sink_inDegree]
                    followee_pr1 = round(np.log(pr[followeeId]), round_number)
                    followee_pr2 = int(followee_pr1)
                    if followeeId in followees:
                        similar_nodes_sinks_degree.add(followeeId)
                        count_sinks = count_sinks + 1
                    if sink_pr == followee_pr1:
                        similar_nodes_sinks_pr1.add(followeeId)
                    if int(sink_pr) == followee_pr2:
                        similar_nodes_sinks_pr2.add(followeeId)

                if count_sinks == 0:
                    print(i, ": SINK: No degree found ", sink_inDegree)
                    count_sinks = len(similar_nodes_sinks_pr1)
                    similar_nodes_sinks_degree = similar_nodes_sinks_pr1
                if count_sinks == 0:
                    print(i, ": SINK: No PR1 found ")
                    count_sinks = len(similar_nodes_sinks_pr2)
                    similar_nodes_sinks_degree = similar_nodes_sinks_pr2

                sample_sinks = set()
                if count_sinks == 0:
                    print(i, ": SINK: No PR2 found ")
                    if source.getOutDegree() >= sample_number:
                        sample_sinks = random.sample(source.outNodes, sample_number)
                    else:
                        sample_sinks = source.outNodes
                elif count_sinks >= sample_number:
                    sample_sinks = random.sample(similar_nodes_sinks_degree, sample_number)
                else:
                    need = sample_number - len(similar_nodes_sinks_degree)
                    possible = source.outNodes.difference(similar_nodes_sinks_degree)
                    print(i, ": SINK: Extra needed:", need, " and POSSIBLE:", len(possible))
                    if possible and (len(possible) >= need):
                        possible = random.sample(possible, need)
                    similar_nodes_sinks_degree.update(possible)
                    sample_sinks = similar_nodes_sinks_degree
                if sourceId not in edges:
                    edges[sourceId] = set()
                edges[sourceId].update(sample_sinks)
                edges_all[i] = edges
            i = i + 1
    print(cnt)
    save_obj(edges_all,'edges_orig_big_3')


# Create Fake training dataset with features
def create_list_all_edges_training_graph_with_features_fake():
    df = list()
    nodes = load_obj('node_properties')
    edges_all = load_obj('edges_fake_new_3')
    pr = load_obj('pr_whole')
    count = 0
    for key, edges in edges_all.items():
        print(key)
        for sourceId, sinks in edges.items():
            source = nodes[sourceId]
            for sinkId in sinks:
                sink = nodes[sinkId]
                followees = source.outNodes.difference({sinkId})
                already = 0
                for followeeId in followees:
                    followee = nodes[followeeId]
                    if sinkId in followee.outNodes:
                        already = already + 1
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
                df.append([key, sourceId, sinkId, source.getInDegree(), source.getOutDegree(), sink.getInDegree(),
                                 sink.getOutDegree(), source.getCommonFollowersCount(sink),
                                 source.getCommonFolloweesCount(sink), source.getFollowBack(sink), already, pr[sourceId], pr[sinkId], aa])
                count = count + 1
    print(count)
    save_obj(df, 'train_fake_big_new_3')


# Create Original rainining dataset with features
def create_list_all_edges_training_graph_with_features_orig():
    df = list()
    nodes = load_obj('node_properties')
    edges_all = load_obj('edges_orig_big_3')
    pr = load_obj('pr_whole')
    count = 0
    for key, edges in edges_all.items():
        print(key)
        for sourceId, sinks in edges.items():
            source = nodes[sourceId]
            for sinkId in sinks:
                sink = nodes[sinkId]
                followees = source.outNodes.difference({sinkId})
                already = 0
                for followeeId in followees:
                    followee = nodes[followeeId]
                    if sinkId in followee.outNodes:
                        already = already + 1
                commons = source.inNodes.intersection(sink.inNodes)
                aa = 0
                if len(commons) > 0:
                    for common in commons:
                        cm = nodes[common]
                        ind = cm.getInDegree() + cm.getOutDegree()
                        if ind > 0:
                            aa = aa + 1.0/np.log(ind)
                else:
                    aa = -123
                df.append([key, sourceId, sinkId, source.getInDegree(), source.getOutDegree(), sink.getInDegree(),
                                 sink.getOutDegree(), source.getCommonFollowersCount(sink),
                                 source.getCommonFolloweesCount(sink), source.getFollowBack(sink), already, pr[sourceId], pr[sinkId], aa])
                count = count + 1
    print(count)
    save_obj(df, 'train_big_3')


def main():
    # create_new_logic_all_fake("data/test-public.txt")
    # create_new_logic_orig("data/test-public.txt")
    # count  = 0
    # edges = load_obj('edges_orig')
    # nodes = load_obj('save_nodes')
    # nodes2 = load_obj('nodes_test')
    # not_found = load_obj('not_found')
    # save_nodes = load_obj('save_nodes')
    # for key, values in edges.items():
    #     save_nodes.add(key)
    #     for value in values:
    #         save_nodes.add(value)
    # save_obj(save_nodes,'save_nodes')
    # print(len(load_obj('save_nodes')))
    # create_list_all_edges_training_graph_with_features()
    # print(count)
    count = 0
    edges = load_obj('edges_orig_big_3')
    for key, values in edges.items():
        for k,v in values.items():
            count = count + len(v)
    print(len(edges), count)
    count = 0
    edges = load_obj('edges_orig_big_3')
    for key, values in edges.items():
        for k, v in values.items():
            count = count + len(v)
    print(len(edges),count)
    # kk = load_obj('node_properties_refined_new')
    # save_obj(set(kk.keys()).difference(load_obj('nodes_test')),'fake_nodes_set')
    create_list_all_edges_training_graph_with_features_fake()
    create_list_all_edges_training_graph_with_features_orig()
    # # kk = load_obj('node_properties_refined_new')
    # fake_nodes_set = load_obj('fake_nodes_set')
    # print(len(fake_nodes_set))
    # fakes_sources = set()
    # fakes_sinks1 = set()
    # fakes_sinks2 = set()
    # for fake in fake_nodes_set:
    #     prop = kk[fake]
    #     if(prop.getOutDegree()>0 and prop.getInDegree()>0):
    #         fakes_sources.add(fake)
    #     if (prop.getOutDegree() == 0 and prop.getInDegree() > 0):
    #         fakes_sinks1.add(fake)
    #     if (prop.getOutDegree() >= 1 and prop.getInDegree() >= 50):
    #         fakes_sinks2.add(fake)
    # save_obj(fakes,'fake_nodes_set')
    # save_obj(fakes_sources, 'fakes_sources')
    # save_obj(fakes_sinks1, 'fakes_sinks1')
    # save_obj(fakes_sinks2, 'fakes_sinks2')
    # print(len(fakes_sources), len(fakes_sinks1), len(fakes_sinks2))

if __name__ == '__main__':
    main()
