import numpy as np
import networkx as nx

import scipy as sp
import scipy.sparse, scipy.io, scipy.optimize, scipy.sparse.csgraph
from scipy.sparse.csgraph import connected_components
from scipy.special import expit, logit
from scipy.io import mmread

import os


class LoadGraph:
	def __init__( self ):
		self.adj = None
		self.G = None
		self.network_name = None

	def loadNetwork( self, network_name ):
		self.network_name = network_name
		path_folder = 'datasets/'
		if network_name[:3] == 'com':
			ext = '.txt'
		elif network_name == "web-edu" or network_name == "road-minnesota":
			ext = '.mtx'
		elif network_name == 'CORA-ML' or network_name == 'CORA-ML_train':
			ext = '.npz'
		else:
			ext = '.mat'
		
		self._loadNetwork( path_folder + network_name + ext )
		self._standardize()
		self._setNetworkX()
		self.expected_overlap = 1.0
		return
		
	def _loadNetwork( self, network_filename ):
		ext = os.path.splitext(network_filename)[-1].lower()
		if ext == ".mat":
			try:
				self.adj = scipy.io.loadmat( network_filename )['network']
			except:
				try:
					self.adj = scipy.io.loadmat( network_filename )['A']
				except:
					self.adj = scipy.io.loadmat( network_filename )['Problem']['A'].item()
		elif ext == ".txt":
			G = nx.read_edgelist(network_filename)
			n = G.number_of_nodes()
			mapping = dict(zip(G, range(0, n)))
			G = nx.relabel_nodes(G, mapping)
			self.adj = nx.to_scipy_sparse_matrix(G)
		elif ext == ".npz":
			cora_ml = np.load( network_filename )
			self.adj = sp.sparse.csr_matrix((cora_ml['data'], cora_ml['indices'], cora_ml['indptr']), shape=cora_ml['shape'])
		elif ext == ".mtx":
			self._readMTX( network_filename )
		return

	def _readMTX( self, network_filename ):
		r_index = []
		c_index = []
		data = []
		node_dict = dict()
		k = 0
		with open(network_filename,'r') as f:
			header = False
			for line in f:
				if line[0] == "%":
					continue
				if header == False:
					rows, columns, edges = line.split()
					header = True
				else:
					if len(line.split()) == 2:
						row, col = line.split()
						data.append( 1. )
					else:
						row, col, w = line.split()
						data.append( int(w) )
					row = int(row)
					col = int(col)
					r_index.append( int(row) - 1 )
					c_index.append( int(col) - 1 )
		self.adj = scipy.sparse.csc_matrix((data, (r_index, c_index)), shape=(int(rows), int(columns)))
		return

	def _standardize( self ):
		"""
		Make the graph undirected and select only the nodes
		belonging to the largest connected component.

		:param adj_matrix: sp.spmatrix
			Sparse adjacency matrix
		:param labels: array-like, shape [n]
		
		:return:
			standardized_adj_matrix: sp.spmatrix
			Standardized sparse adjacency matrix.
			standardized_labels: array-like, shape [?]
			Labels for the selected nodes.
		"""
		# copy the input
		standardized_adj_matrix = self.adj.copy()

		# make the graph unweighted
		standardized_adj_matrix[standardized_adj_matrix != 0] = 1

		# make the graph undirected
		standardized_adj_matrix = standardized_adj_matrix.maximum(standardized_adj_matrix.T)

		# select the largest connected component
		_, components = connected_components(standardized_adj_matrix)
		c_ids, c_counts = np.unique(components, return_counts=True)
		id_max_component = c_ids[c_counts.argmax()]
		select = components == id_max_component
		standardized_adj_matrix = standardized_adj_matrix[select][:, select]

		# remove self-loops
		standardized_adj_matrix = standardized_adj_matrix.tolil()
		standardized_adj_matrix.setdiag(0)
		standardized_adj_matrix = standardized_adj_matrix.tocsr()
		standardized_adj_matrix.eliminate_zeros()

		self.adj =  standardized_adj_matrix
		return

	def _setNetworkX( self ):
		self.G = nx.from_scipy_sparse_matrix( self.adj )
		self.G.remove_edges_from( nx.selfloop_edges(self.G) )
		return