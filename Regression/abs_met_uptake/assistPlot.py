import matplotlib.pyplot as plt

from assistFunct import check_images_dir

class assistPlot:

	def __init__(self, alProcess):
		self.alProcess = alProcess

	def self_labeled_data_amount(self, idx = 2, name = '', folder = '', display = False, save = True):
		try:
			self.alProcess.class_set[idx]
		except:
			raise Exception('The model has not labeled a single data by itself')
		nb_members = len(self.alProcess.member_sets)

		plt.figure()
		plt.plot(range(1, len(self.alProcess.class_set[idx]) + 1), self.alProcess.class_set[idx])
		plt.xlabel('Iteration')
		plt.ylabel('Nubmer of =/= self labeled instances')

		if display:
			plt.show()

		if save:
			check_images_dir(folder)
			path = './images/' + folder + 'self_labeled_data_amount_' + name + '_'
			for stra in self.alProcess.reg_stra:
				if type(stra) == list:
					stra = stra[0]
				path += (stra + '_')
			plt.savefig(path + 'bs' + str(self.alProcess.batch_size) + '_bshv' + str(self.alProcess.batch_size_highest_value) + '_m' + str(nb_members) + '.png', dpi=300)

		plt.close()