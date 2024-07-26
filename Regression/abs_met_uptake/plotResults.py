import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import gaussian_kde

from assistFunct import check_images_dir

class plotResults:

	def __init__(self, alProcess):
		self.alProcess = alProcess

	def top_n_accuracy(self, name = '', folder = '', display = False, save = False):
		try:
			self.alProcess.member_sets
		except:
			raise Exception('(plotResults.top_n_accuracy) member_sets not initialized')
		nb_members = len(self.alProcess.member_sets)

		plt.figure()
		plt.plot(range(1, len(self.alProcess.class_set[0]) + 1), self.alProcess.class_set[0])
		plt.xlabel('Iteration')
		plt.ylabel('Accuracy (%)')
		# plt.title('Accuracy for the top ' + str(self.alProcess.n_top) + ' instances\nbatch_size : ' + str(self.alProcess.batch_size) + ' - batch_size_highest_value : ' + str(self.alProcess.batch_size_highest_value) + ' - nb_members : ' + str(nb_members))

		if display:
			plt.show()

		if save:
			check_images_dir('images/' + folder)
			path = './images/' + folder + 'plot_top_n_accuracy_' + name + '_'
			for stra in self.alProcess.reg_stra:
				if type(stra) == list:
					stra = stra[0]
				path += (stra + '_')
			plt.savefig(path + 'bs' + str(self.alProcess.batch_size) + '_m' + str(nb_members) + '_nTop' + str(self.alProcess.n_top) + '.png', dpi=300)

		plt.close()

	def r2(self, lines, columns, name = '', folder = '', display = False, save = True):
		try:
			self.alProcess.member_sets[0][3]
		except:
			raise Exception('(plotResults.r2) Model has never learned')
		nb_members = len(self.alProcess.member_sets)

		fig, axs = plt.subplots(lines, columns, figsize = (15, 12))
		l, c = 0, 0
		for idx_model in range(lines * columns):
			title = 'Model ('
			if type(self.alProcess.member_sets[idx_model][4]) == list:
				title += self.alProcess.member_sets[idx_model][4][0]
			else:
				title += self.alProcess.member_sets[idx_model][4]
			title += ') ' + str(idx_model + 1)

			if lines == 1:
				if columns == 1:
					plt.plot(range(len(self.alProcess.member_sets[idx_model][3])), self.alProcess.member_sets[idx_model][3])
					plt.title(title)

				else:
					axs[c].plot(range(len(self.alProcess.member_sets[idx_model][3])), self.alProcess.member_sets[idx_model][3])
					axs[c].set_title(title)

			else:
				axs[l, c].plot(range(len(self.alProcess.member_sets[idx_model][3])), self.alProcess.member_sets[idx_model][3])
				axs[l, c].set_title(title)

			if l == lines - 1:
				l = 0
				c += 1
			else:
				l += 1

		if display:
			plt.show()
		if save:
			check_images_dir('images/' + folder)
			path = 'images/' + folder + 'plot_r2_' + name + '_'
			for stra in self.alProcess.reg_stra:
				if type(stra) == list:
					stra = stra[0]
				path += (stra + '_')
			plt.savefig(path + 'bs' + str(self.alProcess.batch_size) + '_m' + str(nb_members) + '.png', dpi=300)

		plt.close()

	def KDE_n_top(self, name = '', folder = '', display = False, save = False):
		try:
			self.alProcess.member_sets
		except:
			raise Exception('(plotResults.KDE_n_top) alProcess not initialized')

		# Standardize the features
		std_scaler = StandardScaler()
		X_scaled = std_scaler.fit_transform(self.alProcess.X)

		# PCA
		pca = PCA(n_components = 2)
		pc_df = pca.fit_transform(X_scaled)

		# Sort the data based on absolute methane uptake in descending order (Optimization possible here : values already computed during alProcess learning)
		sorted_indices = np.argsort(self.alProcess.y)[::-1]
		pc_df_sorted = pc_df[sorted_indices]
		y_sorted = self.alProcess.y[sorted_indices]

		# n_top highest values
		n_top_pc_df_sorted = pc_df_sorted[:self.alProcess.n_top]
		n_top_y = y_sorted[:self.alProcess.n_top]

		# n_top highest predicted values
		n_top_pc_df_predicted = pc_df[self.alProcess.class_set[1]]

		# Normalize the target variable
		min_max_scaler = MinMaxScaler()
		n_top_y_normalized = min_max_scaler.fit_transform(n_top_y.reshape(-1, 1)).flatten()

		# KDE for the n_top data points
		kde = gaussian_kde(n_top_pc_df_sorted.T, weights = n_top_y_normalized, bw_method = 'scott')

		# KDE on a grid
		x_min, x_max = -6, 8
		y_min, y_max = -4, 8
		x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
		positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
		z = kde(positions)

		# Min/Max of target values
		min_target = np.min(self.alProcess.y)
		max_target = np.max(self.alProcess.y)

		# Plot
		plt.figure(figsize = (10, 8))
		scatter = plt.scatter(pc_df[:, 0], pc_df[:, 1], c = self.alProcess.y, cmap = 'coolwarm', alpha = 1, vmin = min_target, vmax = max_target)
		plt.contourf(x_grid, y_grid, z.reshape(x_grid.shape), cmap = 'coolwarm', alpha = 0.8)
		plt.scatter(n_top_pc_df_sorted[:, 0], n_top_pc_df_sorted[:, 1], c = 'red', marker = 'x', s = 50, label = 'n_top highest targets')
		plt.scatter(n_top_pc_df_predicted[:, 0], n_top_pc_df_predicted[:, 1], c = 'black', marker = 'x', s = 50, label = 'n_top highest predicted targets')
		plt.title('KDE of Principal Components with n_top highest targets')
		plt.xlabel('Principal Component 1')
		plt.ylabel('Principal Component 2')
		plt.colorbar(scatter)
		plt.xlim(-6, 8)  # Set x-axis range
		plt.ylim(-4, 8)  # Set y-axis range
		plt.legend()
		plt.grid(True)

		if display:
			plt.show()

		if save:
			check_images_dir('images/' + folder)
			path = 'images/' + folder + 'KDE_top_' + str(self.alProcess.n_top) + '_' + name + '_'
			for stra in self.alProcess.reg_stra:
				if type(stra) == list:
					stra = stra[0]
				path += (stra + '_')
			plt.savefig(path + 'bs' + str(self.alProcess.batch_size) + '_m' + str(nb_members) + '_nTop' + str(self.alProcess.n_top) + '.png', dpi=300)

		plt.close()


