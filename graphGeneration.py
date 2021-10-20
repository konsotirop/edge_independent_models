import numpy as np
import scipy as sp
import scipy.sparse, scipy.io, scipy.optimize, scipy.sparse.csgraph
from scipy.sparse.csgraph import connected_components
from scipy.special import expit, logit
from scipy.io import mmread
import torch
import networkx as nx
import math
from random import randint
from scipy import stats
import random
import subprocess
import os
import io
import gzip
import math
import argparse
import sys
import contextlib
import sys
import pickle as pk
from collections import defaultdict
from dsd import *
from graph_tool.all import *
from cell.cell import Cell, EdgeOverlapCriterion
from scipy.special import expit
import powerlaw

rel_frob_error = lambda observed, actual : np.linalg.norm(observed - actual) / np.linalg.norm(actual)
edge_overlap = lambda observed, actual : (observed.multiply(actual)).sum() / actual.sum()


@contextlib.contextmanager
def nostdout():
	save_stdout = sys.stdout
	sys.stdout = io.BytesIO()
	yield
	sys.stdout = save_stdout

class Graph:
	def __init__( self ):
		self.adj = None
		self.G = None
		self.network_name = None
		self.expected_overlap = None # Not always defined
		self.triangles = None
		self.adj_copy = None # Only for generated networks

	def importNetwork( self, adj, exp_ovrlp, network_name ):
		self.adj = adj
		#self._standardize()
		self._setNetworkX()
		self.expected_overlap = exp_ovrlp
		self.network_name = network_name
		return

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

	def printNodesEdges( self ):
		print("Nodes:{}, Edges:{}".format(self.adj.shape[0],self.G.number_of_edges()))
		return

	def _maximumDegree( self ):
		self.max_degree =  max(self.adj.sum(axis=1)).item()
	
	def _assortativity( self, w=None ):
		self.assortativity = nx.degree_assortativity_coefficient( self.G, weight=w )
	
	def _latapyInput( self ):
		outF = open( 'latapy_temp_' + self.network_name + '.txt', 'w')
		# Write number of nodes
		n = self.G.number_of_nodes()
		outF.write(str(n)+'\n')
		# Write nodes and degrees
		for node,deg in self.G.degree:
			outF.write(str(node)+' '+str(deg)+'\n')
		# Write edges
		for u,v in self.G.edges():
			outF.write(str(u)+' '+str(v)+'\n')
		outF.close()
		return
		
	def _latapyTriangles( self ):
		self._latapyInput()
		inputFile = 'latapy_temp_' + self.network_name + '.txt'
		outputFile = 'output_' + self.network_name + '.txt'
		cmd = "./tr -c -cc -p -cf < " + inputFile +' > ' + outputFile
		p = subprocess.Popen(cmd, shell=True)
		p.wait()
		n_nodes = self.G.number_of_nodes()
		self.triangle_sequence = [0 for i in range(n_nodes)]
		with open(outputFile,'r') as f:
			for i in range(n_nodes):
				node_id, node_tr = f.readline().split()
				node_id, node_tr = int(node_id), int(node_tr)
				self.triangle_sequence[node_id] = node_tr
			f.readline()
			self.triangles = int( f.readline().split()[1] )
			self.avg_cc = float( f.readline().split()[2] )
			self.global_cc = float( f.readline().split()[2] )
		return

	def _characteristicPathLength( self ):
		pairs_count = 1000
		number_of_nodes = self.G.number_of_nodes()
		pairs = self._gencoordinates(0, number_of_nodes-1, pairs_count )
		self.cpl = self._average_paths(pairs, None )
		return
	
	def _gencoordinates(self, m, n, total_pairs):
		seen = set()
		pairs = []
		cnt = 0
		x, y = randint(m, n), randint(m, n)
		while cnt < total_pairs:
			seen.add((x, y))
			pairs.append( (x, y) )
			cnt += 1
			x, y = randint(m, n), randint(m, n)
			while (x, y) in seen or x == y:
				x, y = randint(m, n), randint(m, n)
		return pairs
		
	def _average_paths( self, pairs, weight=None ):
		length = 0
		found = 0
		for p in pairs:
			u,v = p
			try:
				l = nx.shortest_path_length(self.G, source=u, target=v, weight=weight)
				found += 1
			except:
				print("No path found!!" )
				l = 0
			length += l
		return length / found

	def _motifs4( self ):
		"""
		Get motifs of size 4 (using graph-tool libraries)	
		"""
		nx.write_graphml(self.G, self.network_name + 'temp.graphml')
		G_gt = load_graph( self._network_name + 'temp.graphml' )
		self.motifs4 = graph_tool.clustering.motifs(G_gt,4)[1]
	
	def _networkStats( self ):
		self._maximumDegree()
		self._assortativity()
		if not self.triangles:
			self._latapyTriangles()
		#self._motifs4()
		self._standardize() # For generated networks
		self._setNetworkX() # For generated networks
		self._characteristicPathLength()
		self._power_law_alpha()
		return
	
	def _power_law_alpha( self ):
		degrees = np.array(self.adj.sum(axis=-1)).flatten()
		self.pla = powerlaw.Fit(
			degrees, xmin=max(np.min(degrees), 1), verbose=False
			).power_law.alpha
		return

	def getAdjTriangles( self ):
		if not self.triangles:
			self._latapyTriangles()
		return self.adj, self.triangle_sequence
	
	def getAdjG( self ):
		return self.adj, self.G

	def getStatistics( self, adj, triangle_seq ):
		# Pre-calculation
		targets = np.array(adj.sum(axis=1)).flatten()
		vol = targets.sum()
		appx = np.array(self.adj.sum(axis=1)).flatten()
		actual_overlap = (self.adj.multiply(adj)).sum()/vol
		self._networkStats()
		# Result dictionary to be saved (& returned)
		self.results = {
		'max_degree': [self.max_degree],
		'assortativity': [round(self.assortativity,3)],
		'triangles': [self.triangles],
		'triangle_pearson': [round(stats.pearsonr(triangle_seq, self.triangle_sequence)[0],3)],
		'average_cc': [round(self.avg_cc,3)],
		'global_cc': [round(self.global_cc,3)],
		'cpl': [round(self.cpl,3)],
		'lcc': [self.G.number_of_nodes()],
		'degree_pearson': [round(stats.pearsonr(targets, appx)[0],3)],
		'overlap': [round(actual_overlap,3)],
		#'motifs4': self.motifs4,
		'expected_overlap' : [self.expected_overlap],
		'volume': [self.adj.sum()],
		'powerlaw_exponent': [round(self.pla,3)]
		}
		return self.results

class NetworkGenerator:
	def __init__( self, adj, G, network_name ):
		self.input_adj = adj
		self.input_G = G
		self.targets = None
		self.adj_deg = None
		self.network_name = network_name
		return

	def initDSPeeling( self ):
		self.peeled_G = self.input_G.copy()
		self.peeled_iter = 0
		self.peeled_nodes = []
		self.peeled_edges = []
		self.densities = []
		return  
		

	def initGeneration( self, option, parameter=1, method='deterministic'):
		"""
		Option:
			-1: Simple "match degree sequence" method
			0: Fix adjacency vector of maximum degree nodes
			1: Fix densest subgraphs
			2: Fix triangle-densest subgraph
			3: Fix 4-clique densest subgraph
		Parameter:
			(Only applies to 0,1,2,3)
			Fraction of adjacency vectors to fix (OR)
			How many densest subgraphs to fix
		Method:
			deterministic:
				Keep densest subgraph(s)
			randomized:
				Replace densest subgraph(s) with an E-R graph
				of same density
		"""
		self.targets = np.array( self.input_adj.sum(axis=1) ).flatten()
		
		if option <= 0:
			self.adj_deg = self._fix_edges( self.input_adj.toarray(), option, parameter)
		
		self._expectedOverlap()

	def _writeMace( self ):
		n = self.peeled_G.number_of_nodes()
		f2 = open( self.network_name + 'mace_format.txt', 'w' )
		for i in range(n):
			i_neighbors = [str(v) for v in self.peeled_G.neighbors(i) if v > i]
			i_neighbors = ' '.join(i_neighbors)
			f2.write(i_neighbors+'\n')
		f2.close()
		return
		
	def _executeMace( self, k ):
		cmd = "mace22/mace C -l "+str(k)+" -u "+str(k)+ " " + self.network_name + 'mace_format.txt'+" "+ self.network_name + "mace.cliques"
		p = subprocess.Popen(cmd, shell=True)
		p.wait()

		return

	def _readMaceCliques( self ):
		cliques = []
		with open(self.network_name + 'mace.cliques') as f:
			for clique in f:
				nodes = clique.split()
				current_clique = list(map(int,nodes))
				cliques.append( current_clique )
		return cliques


	def _calc_adj_match_deg_fixed(self, targets, nodesInDS, thresh=1e-12):
		n = targets.size
		logit_vec = np.zeros(n)
		adj_deg = expit(logit_vec[:,np.newaxis]+logit_vec[np.newaxis,:])
		np.fill_diagonal(adj_deg, 0.)
		adj_deg[np.ix_(nodesInDS, nodesInDS)] = 0.
		deg_recon = adj_deg.sum(axis=1)
		error = rel_frob_error(deg_recon, targets)
		#print("Iter 0, Error %s" % np.format_float_scientific(error, precision=3))
		i = 0
		while error > thresh:
			adj_deg_jacob = adj_deg * (1.-adj_deg)
			adj_deg_jacob[np.diag_indices_from(adj_deg_jacob)] += (adj_deg * (1.-adj_deg)).sum(axis=1)
			logit_vec_increment = np.linalg.solve(adj_deg_jacob, adj_deg.sum(axis=1) - targets)
			logit_vec = logit_vec - logit_vec_increment
        
			adj_deg = expit(logit_vec[:,np.newaxis]+logit_vec[np.newaxis,:])
			np.fill_diagonal(adj_deg, 0.)
			adj_deg[np.ix_(nodesInDS, nodesInDS)] = 0.
			deg_recon = adj_deg.sum(axis=1)
			error = rel_frob_error(deg_recon, targets)
			i += 1
			#print("Iter %i, Error %s" % (i, np.format_float_scientific(error, precision=3)))

		return adj_deg

	def _expectedOverlap( self ):
		vol = self.adj_deg.sum()
		try:
			self.expected_overlap = round(self.adj_deg.power(2).sum() / vol,3)
		except:
			self.expected_overlap = round(np.power(self.adj_deg, 2).sum() / vol,3)
		print("Expected Overlap: {}".format(round(self.expected_overlap,3)))
		return
		
	def _fix_edges( self, adj, option=-1, fraction=0.1 ):
		n = adj.shape[0]
		sqrt_n = int( math.sqrt(n) )
		deg = adj.sum(axis=1)
		degree_order = np.argsort(-deg)
		if option == -1: # No fixed edges
			fixed_idcs = []
		elif option == 0: # Large degree
			n_fixed = int(np.round(n*fraction))
			fixed_idcs = degree_order[:int(n_fixed)]
		free_idcs = np.setdiff1d(np.arange(n), fixed_idcs)
		free_deg = adj[np.ix_(free_idcs, free_idcs)].sum(axis=1).flatten()
		free_vol = free_deg.sum()
		joint_adj = 1. * np.copy(self.input_adj)
		joint_adj[np.ix_(free_idcs, free_idcs)] = self._calc_adj_match_deg(free_deg)
		
		return joint_adj

	def _calc_adj_match_deg(self, targets, thresh=1e-12, allow_loops=False, return_vec=False, init=None, print_iters=False):
		n = targets.size
		if init is None:
			logit_vec = np.zeros(n)
		else:
			logit_vec = init
    
		adj_deg = expit(logit_vec[:,np.newaxis]+logit_vec[np.newaxis,:])
		if not allow_loops:
			np.fill_diagonal(adj_deg, 0.)
		deg_recon = adj_deg.sum(axis=1)
		error = rel_frob_error(deg_recon, targets)
		if print_iters:
			print("Iter 0, Error %s" % np.format_float_scientific(error, precision=3))
		i = 0
		while error > thresh:
			adj_deg_jacob = adj_deg * (1.-adj_deg)
			adj_deg_jacob[np.diag_indices_from(adj_deg_jacob)] += (adj_deg * (1.-adj_deg)).sum(axis=1)
			logit_vec_increment = np.linalg.solve(adj_deg_jacob, adj_deg.sum(axis=1) - targets)
			logit_vec = logit_vec - logit_vec_increment
			adj_deg = expit(logit_vec[:,np.newaxis]+logit_vec[np.newaxis,:])
			if not allow_loops:
				np.fill_diagonal(adj_deg, 0.)
			deg_recon = adj_deg.sum(axis=1)
			error = rel_frob_error(deg_recon, targets)
			i += 1
			if print_iters:
				print("Iter %i, Error %s" % (i, np.format_float_scientific(error, precision=3)))
		if return_vec:
			return adj_deg, logit_vec
		else:
			return adj_deg
	
	def _probabilisticConfiguration_fixed( self, targets, nodesInDS ):
		nodes = targets.shape[0]
		vol = targets.sum()
		P = targets @ targets.T / vol
		# Excess prob. mass
		P[np.ix_(nodesInDS, nodesInDS)] = 0.
		e_i,e_j = np.where(np.triu(P) > 1)
		excessEdges = len(e_i)
		print("Number of edges having excess mass: {}".format( excessEdges ))
		for t in range(excessEdges):
			if t % 100 == 0:
				print("Processed {} out of {}".format(t,excessEdges))
			i,j = e_i[t], e_j[t]
			while P[i,j] > 1:
				e_ij = P[i,j] - 1
				k, l = np.random.randint(nodes, size=2)
				if i == k or i == l or j == k or j == l or k == l:
					continue
				if random.random() <= 0.5: # 1st Configuration Change
					excessToRemove = max(0,min(e_ij, 1-P[i,k],1-P[j,l],P[k,l]))
					if i == j:
						sameEdgeRemoval = excessToRemove
						excessToRemove /= 2
					else:
						sameEdgeRemoval = excessToRemove
					P[i,j] = P[i,j] - sameEdgeRemoval
					P[j,i] = P[i,j]
					P[i,k] += excessToRemove
					P[k,i] = P[i,k]
					P[j,l] += excessToRemove
					P[l,j] = P[j,l]
					P[k,l] = P[k,l] - excessToRemove
					P[l,k] = P[k,l]
				else:
					# 2nd Configuration Change
					excessToRemove = max(0,min(e_ij, 1-P[i,l],1-P[j,k],P[k,l]))
					if i == j:
						sameEdgeRemoval = excessToRemove
						excessToRemove /= 2
					else:
						sameEdgeRemoval = excessToRemove
					P[i,j] = P[i,j] - sameEdgeRemoval
					P[j,i] = P[i,j]
					P[i,l] += excessToRemove
					P[l,i] = P[i,l]
					P[j,k] += excessToRemove
		# (Try to) remove self-loops
		max_efforts = 500
		for i in range(nodes):
			efforts = 0
			while ( P[i,i] > 0 ) and (max_efforts > efforts ):
				e_ii = P[i,i]
				k, l = np.random.randint(nodes, size=2)
				if i == k or i == l or k == l:
					continue
				excessToRemove = max(0,min(e_ii, 2.0 * min(P[k,l], 1-P[i,k], 1-P[i,l]) ) )
				P[i,i] -= excessToRemove
				P[i,k] += excessToRemove / 2.0
				P[k,i] = P[i,k]
				P[i,l] += excessToRemove / 2.0
				P[l,i] = P[i,l]
				P[k,l] -= excessToRemove / 2.0
				P[l,k] = P[k,l]
				efforts += 1
			P[i,i] = 0 
		return P
	
	def sample_from_expected( self ):
		n,_ = self.adj_deg.shape
		adj_bin = np.zeros((n,n))
		adj_bin[np.triu_indices(n,1)] = (1. * (self.adj_deg >= np.random.uniform(size=(n,n))))[np.triu_indices(n,1)]
		adj_bin = adj_bin + adj_bin.T
		self.sampled = sp.sparse.csr_matrix(adj_bin)

	def getOutput( self ):
		return self.sampled, self.expected_overlap

class OddsProductBaseline( NetworkGenerator ):
	def __init__( self, adj, G, network_name ):
		super().__init__(adj, G, network_name)
	
	def weightedSum( self, org_weight=0.5 ):
		assert(0 <= org_weight and org_weight <= 1)
		targets = np.array( self.input_adj.sum(axis=1) ).flatten()
		adj_odds_product = self._calc_adj_match_deg( targets )
		self.adj_deg = (1.-org_weight)*adj_odds_product + org_weight * self.input_adj
		self._expectedOverlap()
	
class CELLHandler:
	def __init__( self, adj ):
		self.adj = adj
		self.model = None
		self.transition_matrix = None
		return

	def train( self, overlap_limit=0.5, rank=9 ):
		invoke_step = max(1,int(overlap_limit*8))
		self.model = Cell(A=self.adj,
			H=rank,
			callbacks=[EdgeOverlapCriterion(invoke_every=invoke_step, edge_overlap_limit=overlap_limit)])
		self.model.train(steps=200,
			optimizer_fn=torch.optim.Adam,
			optimizer_args={'lr': 0.1,
			'weight_decay': 1e-7})
		return
	
	def sampleScore( self ):
		self._empiricalOverlap()
		self.sampled = self.model.sample_graph()

	def sample( self ):
		return self.model.sample_graph()

	def sampleTransition( self ):
		self._getTransition()
		n,_ = self.transition_matrix.shape
		adj_bin = np.zeros((n,n))
		adj_bin[np.triu_indices(n,1)] = (1. * (self.transition_matrix >= np.random.uniform(size=(n,n))))[np.triu_indices(n,1)]
		adj_bin = adj_bin + adj_bin.T
		self.sampled = sp.sparse.csr_matrix(adj_bin)

	def getOutput( self ):
		return self.sampled, self.expected_overlap

	def _empiricalOverlap( self, it=10 ):
		ovrlp = []
		for i in range(it):
			adj1 = self.sample()
			adj2 = self.sample()
			ovrlp.append( edge_overlap(adj1,adj2) )
		print("Mean Empirical overlap: {}, Variance:{}".format(np.mean(ovrlp),np.std(ovrlp)))
		self.expected_overlap = round(np.mean(ovrlp),3)

	def _getTransition( self ):
		self.transition_matrix = self.model.update_scores_matrix( True )
		vol = self.transition_matrix.sum()
		try:
			self.expected_overlap = round(self.transition_matrix.power(2).sum() / vol,3)
		except:
			self.expected_overlap = round(np.power(self.transition_matrix, 2).sum() / vol,3)
		return
class TSVD:
	def __init__(self):
		self.adj = None
		self.adj_exp = None
		self.expected_overlap = None
		self.sampled = None
		return

	def importNetwork( self, adj):
		self.adj = adj

	def getTruncated(self,rank, adjust_vol=False, thresh=1e-12):
		self.u, self.s, self.vh = scipy.sparse.linalg.svds(self.adj,rank)
		self.adj_exp = self.u @ np.diag(self.s) @ self.vh
		self.adj_exp = np.clip(self.adj_exp, 0, 1)
		vol = self.adj_exp.sum()
		if adjust_vol:
			true_vol = self.adj.sum()
			#self.adj_exp = self.adj_exp * (true_vol / vol ) # Rescale
			#self.adj_exp = np.clip( self.adj_exp, 0, 1)
			scalar_shift = 0.
			tsvd_shift =  self.adj_exp + scalar_shift
			vol_recon = np.clip(self.adj_exp, a_min=0., a_max=1.).sum()
			vol_error = vol_recon - true_vol
			while np.abs(vol_error / vol) > thresh:
				grad = np.bitwise_and(0. < tsvd_shift, tsvd_shift < 1.).sum()
				scalar_shift -= vol_error / grad
				tsvd_shift = self.adj_exp + scalar_shift
				vol_recon = np.clip(tsvd_shift, a_min=0., a_max=1.).sum()
				vol_error = vol_recon - true_vol
			self.adj_exp =  np.clip(tsvd_shift, a_min=0., a_max=1.)
			vol = self.adj_exp.sum()
		try:
			self.expected_overlap = round(self.adj_exp.power(2).sum() / vol,3)
		except:
			self.expected_overlap = round(np.power(self.adj_exp, 2).sum() / vol,3)

	def _sample(self, return_sample=False):
		n,_ = self.adj_exp.shape
		adj_bin = np.zeros((n,n))
		adj_bin[np.triu_indices(n,1)] = (1. * (self.adj_exp >= np.random.uniform(size=(n,n))))[np.triu_indices(n,1)]
		adj_bin = adj_bin + adj_bin.T
		self.sampled = sp.sparse.csr_matrix(adj_bin)
		if return_sample:
			return self.sampled
		else:
			return

	def getOutput( self ):
		self._sample()
		return self.sampled, self.expected_overlap

class ModifiedChungLu:
	def __init__( self ):
		self.adj_exp = None
		self.targets = None
		self.expected_overlap = None

	def importNetwork( self, adj ):
		self.targets = adj.sum(axis=1)
	
	def getExpected( self ):
		self.adj_exp = self._probabilisticConfiguration()
		vol = self.targets.sum()
		try:
			self.expected_overlap = round(self.adj_exp.power(2).sum() / vol,3)
		except:
			self.expected_overlap = round(np.power(self.adj_exp, 2).sum() / vol,3)
		
	def _probabilisticConfiguration( self ):
		nodes = self.targets.shape[0]
		vol = self.targets.sum()
		P = self.targets @ self.targets.T / vol
		# Excess prob. mass
		e_i,e_j = np.where(np.triu(P) > 1)
		excessEdges = len(e_i)
		print("Number of edges having excess mass: {}".format( excessEdges ))
		for t in range(excessEdges):
			if t % 100 == 0:
				print("Processed {} out of {}".format(t,excessEdges))
			i,j = e_i[t], e_j[t]
			while P[i,j] > 1:
				e_ij = P[i,j] - 1
				k, l = np.random.randint(nodes, size=2)
				if i == k or i == l or j == k or j == l or k == l:
					continue
				if random.random() <= 0.5: # 1st Configuration Change
					excessToRemove = max(0,min(e_ij, 1-P[i,k],1-P[j,l],P[k,l]))
					if i == j:
						sameEdgeRemoval = excessToRemove
						excessToRemove /= 2
					else:
						sameEdgeRemoval = excessToRemove
					P[i,j] = P[i,j] - sameEdgeRemoval
					P[j,i] = P[i,j]
					P[i,k] += excessToRemove
					P[k,i] = P[i,k]
					P[j,l] += excessToRemove
					P[l,j] = P[j,l]
					P[k,l] = P[k,l] - excessToRemove
					P[l,k] = P[k,l]
				else:
					# 2nd Configuration Change
					excessToRemove = max(0,min(e_ij, 1-P[i,l],1-P[j,k],P[k,l]))
					if i == j:
						sameEdgeRemoval = excessToRemove
						excessToRemove /= 2
					else:
						sameEdgeRemoval = excessToRemove
					P[i,j] = P[i,j] - sameEdgeRemoval
					P[j,i] = P[i,j]
					P[i,l] += excessToRemove
					P[l,i] = P[i,l]
					P[j,k] += excessToRemove
		# (Try to) remove self-loops
		max_efforts = 500
		for i in range(nodes):
			efforts = 0
			while ( P[i,i] > 0 ) and (max_efforts > efforts ):
				e_ii = P[i,i]
				k, l = np.random.randint(nodes, size=2)
				if i == k or i == l or k == l:
					continue
				excessToRemove = max(0,min(e_ii, 2.0 * min(P[k,l], 1-P[i,k], 1-P[i,l]) ) )
				P[i,i] -= excessToRemove
				P[i,k] += excessToRemove / 2.0
				P[k,i] = P[i,k]
				P[i,l] += excessToRemove / 2.0
				P[l,i] = P[i,l]
				P[k,l] -= excessToRemove / 2.0
				P[l,k] = P[k,l]
				efforts += 1
			P[i,i] = 0 
		return P

	def _sample(self, return_sample=False):
		n,_ = self.adj_exp.shape
		adj_bin = np.zeros((n,n))
		adj_bin[np.triu_indices(n,1)] = (1. * (self.adj_exp >= np.random.uniform(size=(n,n))))[np.triu_indices(n,1)]
		adj_bin = adj_bin + adj_bin.T
		self.sampled = sp.sparse.csr_matrix(adj_bin)
		if return_sample:
			return self.sampled
		else:
			return

	def getOutput( self ):
		self._sample()
		return self.sampled, self.expected_overlap

class ExactEmbedding:
	def __init__( self ):
		self.adj = None
		self.adj_exp = None
		self.expected_overlap = None
		self.sampled = None
		# Useful for callback function
		self._n = None
		self._rank = None
		self._max_overlap = None
		self._iter = 0

	def importNetwork( self, adj ):
		self.adj = adj
	
	def _sample( self ):
		n,_ = self.adj_exp.shape
		adj_bin = np.zeros((n,n))
		adj_bin[np.triu_indices(n,1)] = (1. * (self.adj_exp >= np.random.uniform(size=(n,n))))[np.triu_indices(n,1)]
		adj_bin = adj_bin + adj_bin.T
		self.sampled = sp.sparse.csr_matrix(adj_bin)
		return

	def lpca_approx(self, rank, max_overlap, adjust_vol=False, n_iter=1000):
		def lpca_loss(factors, adj_s, rank):
			n = adj_s.shape[0]
			U = factors[:n*rank].reshape(n, rank)
			V = factors[n*rank:].reshape(rank, n)
			logits = U @ V
			prob_wrong = expit(-logits * adj_s)
			loss = (np.logaddexp(0,-logits*adj_s)).sum()
			U_grad = -(prob_wrong * adj_s) @ V.T
			V_grad = -U.T @ (prob_wrong * adj_s)
			return loss, np.concatenate((U_grad.flatten(), V_grad.flatten()))
		def sample( expectedM ):
			n,_ = expectedM.shape
			adj_bin = np.zeros((n,n))
			adj_bin[np.triu_indices(n,1)] = (1. * (expectedM >= np.random.uniform(size=(n,n))))[np.triu_indices(n,1)]
			adj_bin = adj_bin + adj_bin.T
			return sp.sparse.csr_matrix(adj_bin)

		def overlap_check(x_i):
			factors = x_i
			self._iter += 1
			if self._iter >= 5 and self._iter % 2 == 0:
				U = factors[:self._n*self._rank].reshape(self._n, self._rank)
				V = factors[self._n*self._rank:].reshape(self._rank, self._n)
				current_expected = expit(U@V)
				current_network = sample(current_expected)
				current_overlap = edge_overlap(current_network, self.adj)
				print("{}. Current Overlap: {}, Current Volume: {}".format( self._iter, current_overlap, current_network.sum()))
				if current_overlap >= self._max_overlap:
					self.adj_exp = current_expected.copy()
					raise Exception("Current exceeds maximum")
			
		n = self.adj.shape[0]
		vol = self.adj.sum()
		self._n = n
		self._rank = rank
		self._max_overlap = max_overlap
		self._iter = 0
		factors = (2*vol / n**2 )*( -1 + 2*np.random.random(size=(2*n*rank)) )
		try:
			res = scipy.optimize.minimize(lpca_loss, x0=factors,
				args=(-1 + 2*np.array(self.adj.todense()), rank), jac=True, method='L-BFGS-B',
				callback=overlap_check,
				options={'maxiter':n_iter}
				)
			factors = res.x
			U = res.x[:n*rank].reshape(n, rank)
			V = res.x[n*rank:].reshape(rank, n)
			self.adj_exp =  expit(U@V)
		except:
			pass # Saved in callback check
		
		if adjust_vol:
			exp_vol = self.adj_exp.sum()
			self.adj_exp = self.adj_exp * (vol /exp_vol)
			self.adj_exp = np.clip( self.adj_exp, 0 , 1)
		print("Matrix Rank: {}".format(np.linalg.matrix_rank( self.adj_exp ) ) )
	def _getExpOvrlp( self ):
		vol = self.adj_exp.sum()
		try:
			self.expected_overlap = round(self.adj_exp.power(2).sum() / vol,3)
		except:
			self.expected_overlap = round(np.power(self.adj_exp, 2).sum() / vol,3)

	def getOutput( self ):
		self._getExpOvrlp()	
		self._sample()
		print("Volume: {}".format(self.sampled.sum()))
		return self.sampled, self.expected_overlap

class ControlledOverlap:
	def __init__(self):
		self.adj = None
		self.adj_exp = None
		self.n = None
	
	def importNetwork( self, adj ):
		self.adj = adj
		self.n = adj.shape[0]
		
	def alpha_beta_adj(self, alpha):
		assert alpha <=1. and alpha >= 0.
		n = self.adj.shape[0]
		vol = self.adj.sum()
		self.adj_exp = np.array((alpha * self.adj).todense())
		num_zeros = n*n - vol - n
		self.adj_exp[self.adj_exp == 0] = (1. - alpha) * vol / num_zeros
		np.fill_diagonal(self.adj_exp, 0.)

	def _getExpOvrlp( self ):
		vol = self.adj_exp.sum()
		try:
			self.expected_overlap = round(self.adj_exp.power(2).sum() / vol,3)
		except:
			self.expected_overlap = round(np.power(self.adj_exp, 2).sum() / vol,3)
	
	def _sample( self ):
		n,_ = self.adj_exp.shape
		adj_bin = np.zeros((n,n))
		adj_bin[np.triu_indices(n,1)] = (1. * (self.adj_exp >= np.random.uniform(size=(n,n))))[np.triu_indices(n,1)]
		adj_bin = adj_bin + adj_bin.T
		self.sampled = sp.sparse.csr_matrix(adj_bin)
		return

	def getOutput( self ):
		self._getExpOvrlp()	
		self._sample()
		return self.sampled, self.expected_overlap
		
def main():
	if not sys.warnoptions:
		warnings.simplefilter("ignore")
	
	parser = argparse.ArgumentParser(description='Define filename')
	parser.add_argument('-f', '--filename', required=True, help="Relative path to dataset file")
	args = parser.parse_args()
	
	network_name = args.filename #"citeseer" # TO BE REPLACED BY COMMAND LINE ARGUMENTS
	
	# Initialize Graph Instance
	G = Graph()
	G.loadNetwork( network_name )
	originalStatistics =  G.getStatistics( *G.getAdjTriangles() )
	# Save true network
	pk.dump( originalStatistics, open( 'results/' + network_name + '_original.pk', "wb" ) )
	print(originalStatistics)

	def getStatistics( C, G ):
		GN = Graph()
		GN.importNetwork( *C.getOutput(), network_name )
		result = GN.getStatistics( *G.getAdjTriangles() )
		return result
	
	def updateDict( initial, newValues ):
		for k in initial:
			initial[k].extend( newValues[k] )
		return

	# Graph Generation Process
	option_ranges = {
		-1: [0], # Nothing
		0: np.linspace(0,0.3,8),
	}

	for option in [0]:
		#break # DELETE -- only for DEBUG
		N = NetworkGenerator( *G.getAdjG(), network_name )
		for method in ['deterministic']:
			N.initDSPeeling()
			for parameter in option_ranges[option]:
				print(" ******** Option: {}, Method: {}, Parameter: {}  ********".format(option,method,parameter))
				try:
				#if True:
					N.initGeneration(option,parameter,method)
					for r_s in range(5): # r_s repeated sampling
						N.sample_from_expected()
						# Generated
						if r_s == 0:
							initial_dict_result = getStatistics( N, G )
						else:
							new_dict_result = getStatistics( N, G )	
							updateDict( initial_dict_result, new_dict_result )
					pk.dump( initial_dict_result, open( 'results/' + network_name + '_option' + str(option) + '_method' + method + '_param'+str(parameter)+'.pk', "wb" ) )
					print( initial_dict_result )
				except:
					print("Failed for parameter: {}".format(parameter))
					break
			if option < 1: # There is no method except for k-DS
				break

	# ODDS PRODUCT BASELINE
	OPB = OddsProductBaseline(*G.getAdjG(), network_name )
	for weight in range(1,10,2):
		break # DELETE -- only for DEBUG purposes
		OPB.weightedSum( weight/10 )
		for r_s in range(5):
			OPB.sample_from_expected()
			if r_s == 0:
				initial_dict_result = getStatistics( OPB, G )
			else:
				new_dict_result = getStatistics( OPB, G )
				updateDict( initial_dict_result, new_dict_result )
		pk.dump( initial_dict_result, open( 'results/' + network_name + '_OPB_OrgWeight' + str(weight) + '.pk', "wb" ) )
		print( initial_dict_result )
	
	# CELL
	C_G = CELLHandler( G.getAdjG()[0] )
	for rank in [9,16,32]:
		for ovrlp in np.linspace(0.05,0.75,8):
			break ## DELETE DEBUGGING ONLY
			print(" ********* Train CELL, up to {} overlap ********".format(ovrlp))
			C_G.train( ovrlp, rank )
			for r_s in range(5):
				C_G.sampleScore()
				if r_s == 0:
					initial_dict_result = getStatistics( C_G, G )
				else:
					new_dict_result = getStatistics( C_G, G )
					updateDict( initial_dict_result, new_dict_result )
			pk.dump( initial_dict_result, open( 'results/' + network_name + '_CELL_Rank' + str(rank) + '_Ov_' + str(ovrlp) + '.pk', "wb" ) )
			print( initial_dict_result )




	# Truncated SVD
	T = TSVD()
	T.importNetwork( G.getAdjG()[0] )
	min_rank, max_rank = 5, int( G.G.number_of_nodes() / 2.0 )
	for adj_vol in [True,False]:
		for rank in np.linspace(min_rank,max_rank,8):
			break ## DELETE DEBUGGING ONLY
			int_rank = int(rank)
			print(" ******* Truncated SVD, Rank: {} ********* ".format( int_rank ) )
			T.getTruncated( int_rank, adj_vol )
			for r_s in range(5):
				if r_s == 0:
					initial_dict_result = getStatistics( T, G )
				else:
					new_dict_result = getStatistics( T, G )
					updateDict( initial_dict_result, new_dict_result ) 
			if adj_vol:
				pk.dump( initial_dict_result, open( 'results/' + network_name + '_TSVD_adjVol' + str(int_rank) + '.pk', "wb" ) )
			else:
				pk.dump( initial_dict_result, open( 'results/' + network_name + '_TSVD_' + str(int_rank) + '.pk', "wb" ) )
			print( initial_dict_result )
	
	# Modified Chung-Lu
	#print("Modified Chung-Lu")
	#MCL = ModifiedChungLu()
	#MCL.importNetwork( G.getAdjG()[0] )
	#MCL.getExpected()
	#for r_s in range(5):
	#	if r_s == 0:
	#		initial_dict_result = getStatistics( MCL, G )
	#	else:
	#		new_dict_result = getStatistics( MCL, G )
	#		updateDict( initial_dict_result, new_dict_result )	
	#pk.dump( initial_dict_result, open( 'results/' + network_name + '_MCL_' + '.pk', "wb" ) )
	#print( initial_dict_result )

	# "Exact" Embedding
	EE = ExactEmbedding()
	EE.importNetwork( G.getAdjG()[0] )
	for ovrlp in np.linspace(0.05,0.75,8):
		break ## DELETE DEBUGGING ONLY
		print("(Not) Exact embeddings, Overlap {}".format(ovrlp))
		EE.lpca_approx(16,ovrlp, True)
		for r_s in range(5):
			if r_s == 0:
				initial_dict_result = getStatistics( EE, G )
			else:
				new_dict_result = getStatistics( EE, G )
				updateDict( initial_dict_result, new_dict_result )
		pk.dump( initial_dict_result, open( 'results/' + network_name + '_NEE_adjVol' + str(ovrlp) + '.pk', "wb" ) )
		print( initial_dict_result )
	
	# Controlled Overlap Baseline
	CO = ControlledOverlap()
	CO.importNetwork( G.getAdjG()[0] )
	for a_parameter in range(1,10):
		#break ## DELETE DEBUGGING ONLY
		a_param = a_parameter / 10
		CO.alpha_beta_adj( a_param )
		for r_s in range(5):
			#break
			if r_s == 0:
				initial_dict_result = getStatistics( CO, G )
			else:
				new_dict_result = getStatistics( CO, G )
				updateDict( initial_dict_result, new_dict_result )
		pk.dump( initial_dict_result, open( 'results/' + network_name + '_CO_' + str(a_parameter) + '.pk', "wb" ) )
		print( initial_dict_result )
if __name__ == "__main__":
	main()
