from collections import defaultdict
from tqdm import tqdm
import json
import numpy as np
import scipy.sparse
from random import random, shuffle
import random
import pdb

class InteractionGraph:
    def __init__(self, user_data, item_data, interactions, ratings_by_edges) -> None:
        self.user_data = user_data
        self.item_data = item_data
        self.interactions = interactions
        self.ratings_by_edges = ratings_by_edges
        self.train_edges, self.validation_edges, self.test_edges = [], [], []
        self.adj_matrix: scipy.sparse.dok_matrix = None

    def split_statistics(self):
        training_items = set(self.train_edges[:, 1])
        validation_items = set(self.validation_edges[:, 1])
        test_items = set(self.test_edges[:, 1])

        print("Total number of items = {}".format(len(self.item_data)))
        print("Total number of users = {}".format(len(self.user_data)))
        print("Number of items present across training edges = {}".format(len(training_items)))
        print("Number of items present across val edges = {}".format(len(validation_items)))
        print("Number of items present across test edges = {}".format(len(test_items)))
        print("Average item degree = {}".format(np.mean(self.item_degrees)))
        print("Average user degree = {}".format(np.mean(self.user_degrees)))

        train_val_common_items = training_items.intersection(validation_items)
        train_test_common_items = training_items.intersection(test_items)

        print('Number of items common between train and validation edges = {}'.format(len(train_val_common_items)))
        print('Number of items common between train and test edges = {}'.format(len(train_test_common_items)))

        validation_items = np.array(list(validation_items))
        test_items = np.array(list(test_items))

        num_cold_items_in_val = np.sum(self.is_cold[validation_items])
        num_cold_items_in_test = np.sum(self.is_cold[test_items])

        print('Number of cold items in validation set = {}'.format(num_cold_items_in_val))
        print('Number of cold items in test set = {}'.format(num_cold_items_in_test))


    def create_bipartite_graph(self):
        num_nodes = len(self.user_data) + len(self.item_data) # Num users + num items 
        self.adj_matrix = scipy.sparse.dok_matrix((num_nodes, num_nodes), dtype=bool)  # TODO: Maybe we can optimize with lower precision data types
        
        for edge in self.train_edges:
            self.adj_matrix[edge[0], edge[1]] = 1
            self.adj_matrix[edge[1], edge[0]] = 1

        self.adj_matrix = self.adj_matrix.tocsr()
    
    def compute_tail_distribution(self, warm_threshold):
        self.is_cold = np.zeros((self.adj_matrix.shape[0]), dtype=bool)
        self.start_item_id = len(self.user_data)

        self.user_degrees = np.array(self.adj_matrix[:self.start_item_id].sum(axis=1)).flatten()
        self.item_degrees = np.array(self.adj_matrix[self.start_item_id:].sum(axis=1)).flatten()

        cold_items = np.argsort(self.item_degrees)[:int((1 - warm_threshold) * len(self.item_degrees))] + self.start_item_id
        self.is_cold[cold_items] = True

    def create_data_split(self):
        raise NotImplementedError()
    
class BARDInteractionGraph(InteractionGraph):
    def __init__(self, user_data, item_data, interactions, ratings_by_edges, warm_threshold=0.2) -> None:
        super().__init__(user_data, item_data, interactions, ratings_by_edges)
        self.create_data_split()
        self.create_bipartite_graph()
        assert (warm_threshold < 1.0 and warm_threshold > 0.0)
        self.warm_threshold = warm_threshold
        self.compute_tail_distribution()
        self.create_train_edges_per_user()
        self.create_user_MultiADI_splits()
    
    def create_data_split(self):
        # Leave one out validation - for each user the latest interaction is a test item and the second latest item is the validation item
        print('Creating data split')
        self.all_edges = set()
        self.interaction_time_stamps = {}
        self.positive_edges = 0
        for user_id in tqdm(self.interactions):
            shuffle(self.interactions[user_id])
            self.test_edges.append([user_id, self.interactions[user_id][-1][0]])
            for interaction in self.interactions[user_id][:-1]:
                self.train_edges.append([user_id, interaction[0]])
        
        self.train_edges = np.array(self.train_edges)
        self.validation_edges = np.array(self.test_edges)
        self.test_edges = np.array(self.test_edges)
    
    def create_train_edges_per_user(self):
        self.edges_per_user = {}
        self.candidates_and_weights_per_user = {}
        for k, v in self.train_edges:
            dialect_info = self.ratings_by_edges[(k,v)]
            self.edges_per_user.setdefault(k, []).append((k,v, dialect_info))

    def create_user_MultiADI_splits(self):

        self.range_user_ids = defaultdict(list)

        for user_id, attributes in self.user_data.items():
            # Get the value for the specified second key
            key_value = attributes['most_common_dilects'][0]
            # Group the user by this key value
            self.range_user_ids[key_value].append(user_id)

        for group, users in self.range_user_ids.items():
            print(f"Group {group}: {len(users)} users")
                
    def compute_tail_distribution(self):
        return super().compute_tail_distribution(self.warm_threshold)

    def __getitem__(self, user_id):
        assert user_id < len(self.user_data), "User ID out of bounds"
        assert isinstance(self.adj_matrix, scipy.sparse.csr_matrix), "Bipartite graph not created: must call create_bipartite_graph first"
        return np.array(self.adj_matrix[user_id, self.start_item_id:].todense()).flatten().nonzero()[0] + self.start_item_id