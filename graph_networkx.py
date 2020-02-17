import networkx as nx
from utility import *
import operator

def experiment():
    DG = load_obj('DG')
    print(DG.number_of_nodes(), DG.number_of_edges())

def node_edges(fileName):
    nodeMapper = {}
    edges = 0
    count = 0
    with open(fileName) as f:
        for line in f:
            line = line.strip()
            nodes_1 = line.split("\t")
            sourceId = nodes_1[0]
            print(count, len(nodes_1))
            if sourceId in nodeMapper:
                sourceId = nodeMapper[sourceId]
            else:
                temp = len(nodeMapper)
                nodeMapper[sourceId] = temp
                sourceId = temp
            for sinkId in nodes_1[1:]:
                if sinkId in nodeMapper:
                    sinkId = nodeMapper[sinkId]
                else:
                    temp = len(nodeMapper)
                    nodeMapper[sinkId] = temp
                    sinkId = temp
                edges = edges + 1
            count = count + 1
    print(len(nodeMapper), edges)

def create_graph(fileName):
    nodeMapper = load_obj('nodeMapper')
    mapper = set()
    DG = nx.DiGraph()
    count = 0
    with open(fileName) as f:
        for line in f:
            line = line.strip()
            nodes_1 = line.split("\t")
            sourceId = nodes_1[0]
            print(count, len(nodes_1))
            sourceId = nodeMapper[sourceId]
            if sourceId not in mapper:
                DG.add_node(sourceId)
                mapper.add(sourceId)
            for sinkId in nodes_1[1:]:
                sinkId = nodeMapper[sinkId]
                if sinkId not in mapper:
                    DG.add_node(sinkId)
                    mapper.add(sinkId)
                DG.add_edge(sourceId, sinkId)
            count = count + 1
    #save_obj(DG, 'DG_whole')
    nx.write_gpickle(DG, "test.gpickle")

def create_networkx_graph():
    mapper = set()
    nodes = load_obj('nodes_test_sources')
    nodes_props = load_obj('node_properties')
    DG = nx.DiGraph()
    count = 0
    for sourceId in nodes:
        source = nodes_props[sourceId]
        if sourceId not in mapper:
            DG.add_node(sourceId)
            mapper.add(sourceId)
        possible_nodes = source.outNodes
        for sinkId in possible_nodes:
            sink = nodes_props[sinkId]
            if sinkId not in mapper:
                DG.add_node(sinkId)
                mapper.add(sinkId)
            DG.add_edge(sourceId, sinkId)
            count = count + 1
    print(count, len(mapper))
    save_obj(DG, 'DG')

def graph_exploration():
    DG = load_obj('DG_train')
    print(DG.number_of_nodes(), DG.number_of_edges(), nx.number_connected_components(DG.to_undirected()))
    # graphs = list(nx.connected_component_subgraphs(DG.to_undirected()))
    # Gc = max(nx.connected_component_subgraphs(DG.to_undirected()), key=len)
    # save_obj(Gc, 'Gc')
    # nx.draw(DG)
    # plt.draw()
    # plt.show()
    # plt.savefig("graph_DG.PDF")
    # pr = nx.pagerank(DG, alpha=0.9)
    # print(len(pr),pr)
    # save_obj(pr, 'nodes_pr')

def drawing():
    Gc = load_obj('Gc')
    print(Gc.number_of_nodes(), Gc.number_of_edges())
    nx.draw(Gc)
    plt.draw()
    plt.savefig("graph_GC.pdf")
    # plt.show()

def CC():
    DG = load_obj('DG')
    # DG_undirected = DG.to_undirected()
    print(DG.number_of_nodes(), DG.number_of_edges())
    # print(DG_undirected.number_of_nodes(), DG_undirected.number_of_edges(), nx.number_connected_components(DG_undirected))
    # graphs = list(nx.connected_component_subgraphs(DG.to_undirected()))
    # Gc = max(nx.connected_component_subgraphs(DG.to_undirected()), key=len)
    # print(Gc.number_of_nodes(), Gc.number_of_edges())
    # ccs = sorted(nx.connected_component_subgraphs(DG.to_undirected()), key=len, reverse=True)
    sccs = sorted(nx.strongly_connected_components(DG), key=len, reverse=True)
    for scc in sccs:
        print(len(scc))
    save_obj(sccs[0],'scc')
    tot_nodes = 0
    # tot_edges = 0
    # for cc in sccs:
    #     tot_nodes = tot_nodes + cc.number_of_nodes()
    #     tot_edges = tot_edges + cc.number_of_edges()
    # # print(cc.number_of_nodes(), cc.number_of_edges())

    # for cc in sccs:
    #     tot_nodes = tot_nodes + len(cc)
    #     print(len(cc))
    # print(cc.number_of_nodes(), cc.number_of_edges())

    # test = load_obj('df_test')
    # cnt = 0
    # cnt_x = 0
    # cnt_y = 0
    # cnt_x1 = 0
    # cnt_y1 = 0
    # print(test.shape)
    # for index, row in test.iterrows():
    #     sourceId = row['source']
    #     sinkId = row['sink']
    #     if sourceId in sccs[0]:
    #         cnt_x = cnt_x + 1
    #     if sinkId in sccs[0]:
    #         cnt_y = cnt_y + 1
    #     if sourceId in sccs[0] and sinkId in sccs[0]:
    #         cnt = cnt + 1
    #     else:
    #         if sourceId not in sccs[0]:
    #             cnt_x1 = cnt_x1 + 1
    #         if sinkId not in sccs[0]:
    #             cnt_y1 = cnt_y1 + 1
    # print(cnt, cnt_x, cnt_y, cnt_x1, cnt_y1)

def pageRank():
   # DG = nx.read_gpickle("test.gpickle")
   DG = load_obj('DG')
   print("Graph Loaded")
   print(DG.number_of_nodes(), DG.number_of_edges())
   pr = nx.pagerank(DG)
   print("PR calculation done")
   save_obj(pr,'pr')
   # sorted_d = sorted(pr.items(), key=operator.itemgetter(1), reverse=True)
   # print(pr[882079])

def main():
    # create_graph("data/train.txt")
    # experiment()
    # node_edges("data/train.txt")
    # CC()
    # create_networkx_graph()
    # graph_exploration()
    # create_networkx_graph()
    pageRank()
    print("done")

if __name__ == '__main__':
    main()