import numpy as np

from assistFunct import check_images_dir

def rank_instance(y_instance, y_sorted):
	rank = 0
	found = False

	while not found and rank < len(y_sorted):
		if y_instance == y_sorted[rank]:
			found = True
		else:
			rank += 1

	if not found:
		raise Exception('(rank_instance) y_instance not found')

	return rank

def get_stra(alProcess, idx_model):
	stra = alProcess.member_sets[idx_model][4]
	if type(stra) == list:
		stra = stra[0]

	return stra

class logs:

	def __init__(self, alProcess):
		self.alProcess = alProcess

	def gen_save(self, n_init = -1, twoSteps = False, last_instance_added = False, r2 = False, n_top = False, name = '', folder = ''):
		check_images_dir('logs/' + folder)
		f = open('./logs/' + folder + name + '.txt', 'a')
		f.write('-------------------------------------------------------------------------------------\n')

		nb_instances = len(self.alProcess.y)
		y_sorted = np.sort(self.alProcess.y)[::-1]

		# Initial datasets
		if n_init > 0:
			f.write('Initial datasets :\n')
			for idx_model in range(len(self.alProcess.member_sets)):
				f.write(get_stra(self.alProcess, idx_model) + ' :\n')
				for i in range(n_init):
					f.write('\tRank #' + str(rank_instance(self.alProcess.member_sets[idx_model][1][i], y_sorted)) + '\n')
			f.write('-------------------------------------------------------------------------------------\n')
		
		# Rest
		avg_instance_added = 0
		peak_instace_addded = int(self.alProcess.y.shape[0])

		for iteration in range(self.alProcess.nb_iterations):
			f.write('Iteration n°' + str(iteration + 1) + ' :\n')

			# Last instance added
			if last_instance_added:
				### TODO : implement selfLabeling
				rank_instance_added = rank_instance(self.alProcess.member_sets[idx_model][1][n_init + iteration], y_sorted)
				f.write('Next instance to be added : Rank #' + str(rank_instance_added) + '\n')
				avg_instance_added += rank_instance_added
				if rank_instance_added < peak_instace_addded:
					peak_instace_addded = rank_instance_added

			# r2
			if r2:
				f.write('r2 :\n')
				for idx_model in range(len(self.alProcess.member_sets)):
					f.write('\t' + get_stra(self.alProcess, idx_model) + ' : ' + str(self.alProcess.member_sets[idx_model][3][iteration]) + '\n')

			# n_top
			if n_top:
				f.write('Top ' + str(self.alProcess.n_top) + ' accuracy : ' + str(self.alProcess.class_set[0][iteration]) + '\n')
				try:
					f.write('Top ' + str(self.alProcess.n_top) + ' accuracy (n_top_uncertainty) : ' + str(self.alProcess.class_set[3][iteration]) + '\n')
				except:
					pass

			f.write('-------------------------------------------------------------------------------------\n')

		# twoSteps
		if twoSteps:
			try:
				f.write(str(self.alProcess.n_top_train) + ' instances added during the second step.\n')
				f.write('Final top ' + str(self.alProcess.n_top) + ' accuracy : ' + str(self.alProcess.class_set[0][-1]) + '\n')
				f.write('-------------------------------------------------------------------------------------\n')
			except:
				pass

		if last_instance_added:
			f.write('Average of rank added : ' + str(avg_instance_added / self.alProcess.nb_iterations) + '\n')
			f.write('Peak rank added : ' + str(peak_instace_addded) + '\n')
			f.write('-------------------------------------------------------------------------------------\n')

		f.close()


