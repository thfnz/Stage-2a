import matplotlib.pyplot as plt

from assistFunct import check_images_dir

class plotResults:

	def __init__(self, alProcess):
		self.alProcess = alProcess

	def top_n_accuracy(self, name = '', folder = '', display = False, save = False):
		nb_members = len(self.alProcess.member_sets)

		plt.figure()
		plt.plot(range(1, len(self.alProcess.class_set) + 1), self.alProcess.class_set)
		plt.xlabel('Iteration')
		plt.ylabel('Accuracy (%)')
		plt.title('Accuracy for the top ' + str(self.alProcess.n_top) + ' instances\nbatch_size : ' + str(self.alProcess.batch_size) + ' - batch_size_highest_value : ' + str(self.alProcess.batch_size_highest_value) + ' - nb_members : ' + str(nb_members))

		if display:
			plt.show()

		if save:
			check_images_dir(folder)
			path = './images/' + folder + '/plot_top_n_accuracy' + name + '_'
			for stra in self.alProcess.reg_stra:
				path += (stra + '_')
			plt.savefig(path + 'bs' + str(self.alProcess.batch_size) + '_bshv' + str(self.alProcess.batch_size_highest_value) + '_m' + str(nb_members) + '.png', dpi=300)

		plt.close()