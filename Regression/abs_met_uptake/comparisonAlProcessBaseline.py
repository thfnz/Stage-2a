import copy
import gc
import matplotlib.pyplot as plt

from assistFunct import check_images_dir

def plot_hist_n_top_acc(n_top_accs, reg_stra, display = False, save = False):
	plt.figure()
	plt.hist(n_top_accs)

	if display:
		plt.show()

	if save:
		check_images_dir('')
		path = './images/plot_hist_n_top_acc_'
		for stra in reg_stra:
			path += (stra + '_')
		plt.savefig(path + '.png', dpi=300)

	plt.close()

class comparisonAlProcessBaseline:

	def __init__(self, alProcess, baseline, X, y, reg_stra, nb_members, n_init):
		self.alProcess = alProcess
		self.baseline = baseline
		self.X = X
		self.y = y
		self.reg_stra = reg_stra
		self.nb_members = nb_members
		self.n_init = n_init

	def comparison_top_n_accuracy(self, nb_processes, display = False, save = True, pbar = False):
		self.alProcess_n_top_accs = []
		self.baseline_n_top_accs = []

		# Same member_sets initialization
		member_sets = self.alProcess.member_setsInit(self.X, self.y, self.reg_stra, self.nb_members, self.n_init, display = display)
		self.alProcess.member_sets = member_sets
		self.baseline.member_sets = member_sets

		# Retrieval of the last n_top accuracy after AL
		for idx_process in range(nb_processes):
			al = copy.deepcopy(self.alProcess)
			base = copy.deepcopy(self.baseline)

			al.learn(display = display, pbar = pbar)
			base.learn(display = display, pbar = pbar)

			self.alProcess_n_top_accs.append(al.class_set[-1])
			self.baseline_n_top_accs.append(base.class_set[-1])

			del al
			del base
			gc.collect()

		# Plot
		plot_hist_n_top_acc(self.alProcess_n_top_accs, self.reg_stra, display = display, save = save)
		plot_hist_n_top_acc(self.baseline_n_top_accs, self.reg_stra, display = display, save = save)