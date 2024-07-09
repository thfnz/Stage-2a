import matplotlib.pyplot as plt

from assistFunct import check_images_dir

class plotResults:

	def __init__(self, alProcess):
		self.alProcess = alProcess

	def top_n_accuracy(self, name = '', folder = '', display = False, save = False):
		try:
			self.alProcess.member_sets
		except:
			raise Exception('member_sets not initialized')
		nb_members = len(self.alProcess.member_sets)

		plt.figure()
		plt.plot(range(1, len(self.alProcess.class_set[0]) + 1), self.alProcess.class_set[0])
		plt.xlabel('Iteration')
		plt.ylabel('Accuracy (%)')
		plt.title('Accuracy for the top ' + str(self.alProcess.n_top) + ' instances\nbatch_size : ' + str(self.alProcess.batch_size) + ' - batch_size_highest_value : ' + str(self.alProcess.batch_size_highest_value) + ' - nb_members : ' + str(nb_members))

		if display:
			plt.show()

		if save:
			check_images_dir(folder)
			path = './images/' + folder + 'plot_top_n_accuracy_' + name + '_'
			for stra in self.alProcess.reg_stra:
				path += (stra + '_')
			plt.savefig(path + 'bs' + str(self.alProcess.batch_size) + '_bshv' + str(self.alProcess.batch_size_highest_value) + '_m' + str(nb_members) + '.png', dpi=300)

		plt.close()

	def r2(self, lines, columns, name = '', folder = '', display = False, save = True):
		try:
			self.alProcess.member_sets[0][3]
		except:
			raise Exception('Model has never learned')
		nb_members = len(self.alProcess.member_sets)

		fix, axs = plt.subplots(lines, columns, figsize = (15, 12))
		l, c = 0, 0
		for idx_model in range(lines * columns):
			if lines == 1:
				axs[c].plot(range(len(self.alProcess.member_sets[idx_model][3])), self.alProcess.member_sets[idx_model][3])
				axs[c].set_title('Model (' + self.alProcess.member_sets[idx_model][4] + ') ' + str(idx_model + 1))
			else:
				axs[l, c].plot(range(len(self.alProcess.member_sets[idx_model][3])), self.alProcess.member_sets[idx_model][3])
				axs[l, c].set_title('Model (' + self.alProcess.member_sets[idx_model][4] + ') ' + str(idx_model + 1))

			if l == lines - 1:
				l = 0
				c += 1
			else:
				l += 1

		if display:
			plt.show()
		if save:
			check_images_dir(folder)
			path = 'images/' + folder + 'selfLabelingInde_plot_r2_' + name + '_'
			for stra in self.alProcess.reg_stra:
				path += (stra + '_')
			plt.savefig(path + 'bs' + str(self.alProcess.batch_size) + '_bshv' + str(self.alProcess.batch_size_highest_value) + '_m' + str(nb_members) + '.png', dpi=300)

		plt.close()


