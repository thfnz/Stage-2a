import matplotlib.pyplot as plt
import os

def check_images_dir(dir):
	os.makedirs('./images', exist_ok = True)
	if len(dir) > 0:
		os.makedirs(dir, exist_ok = True)

class plotResults:

	def __init__(self, alProcess):
		self.alProcess = alProcess

	def top_n_accuracy(self, display = False, save = False):
		nb_members = len(self.alProcess.member_sets)

		plt.figure()
		plt.plot(range(len(self.alProcess.class_set)), self.alProcess.class_set)
		plt.xlabel('Iteration')
		plt.ylabel('Accuracy')
		plt.title('Accuracy for the top ' + str(self.alProcess.n_top) + ' instances\nbatch_size : ' + str(self.alProcess.batch_size) + ' - batch_size_highest_value : ' + str(self.alProcess.batch_size_highest_value) + ' - nb_members : ' + str(nb_members))

		if display:
			plt.show()

		if save:
			check_images_dir()
			path = './images/plot_top_n_accuracy_'
			for stra in self.alProcess.reg_stra:
				path += (stra + '_')
			plt.savefig(path + 'bs' + str(self.alProcess.batch_size) + '_bshv' + str(self.alProcess.batch_size_highest_value) + '_m' + str(nb_members) + '.png', dpi=300)

		plt.close()